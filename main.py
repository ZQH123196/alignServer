import os
import sys
import time as _time

# stdout 一旦被重定向到管道/文件（nohup、tee），Python 默认块缓冲：日志攒到 8KB 才出现，
# 启动阶段整体不可见。强制行缓冲让每一行即时落终端。
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_BOOT_AT = _time.monotonic()
_BOOT_TOTAL = 7
_boot_step = 0


def boot(msg: str) -> None:
    """启动期阶段进度。定义在结构化日志与业务模块之前，专门填补导入期的静默窗口。

    带 [k/N] 计数：停在 [k/6] 且不再前进，即卡在标出的那个阶段；6 步跑完后由
    uvicorn 的 `Uvicorn running on ...` 行确认端口已监听、可接收请求。
    install_async_logging() 之前的这几步只能走 stdout，因为日志文件路径还没校验过。
    """
    global _boot_step
    _boot_step += 1
    print(f"[+{_time.monotonic() - _BOOT_AT:5.1f}s] [{_boot_step}/{_BOOT_TOTAL}] {msg}", flush=True)


boot("进程已启动 | 正在导入 torch / numpy / fastapi / uvicorn（静默阶段）...")

# libgomp（torch 自带的 OpenMP 运行库）在 .so 被 dlopen 时读一次 OMP_NUM_THREADS，
# 要求正整数。系统镜像注入的 0 是无效值，libgomp 报错并回退到宿主机核数。
# 系统文件不能改，在 import torch 之前覆盖成容器配额。
# pipeline 内还会用 torch.set_num_threads 按 cpu_ratio 再收紧，这里只定上限。
from utils import get_container_cpu_count

os.environ["OMP_NUM_THREADS"] = str(get_container_cpu_count())
os.environ["MKL_NUM_THREADS"] = str(get_container_cpu_count())

import json
import asyncio
import gc
import jieba
import torch
import numpy as np
import uvicorn
import tempfile
import uuid  
import traceback  
import threading

import async_logging
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from env_config import (
    StartupConfigError,
    check_device_capability,
    check_lockfile,
    load_server_env,
)
from utils import get_container_available_memory, normalize_multipart_line_endings, with_chapter_identity, audio_defect_error_detail
import transformers
transformers.logging.set_verbosity_error()
# set_verbosity_error() 连带关掉权重加载进度条，静默只能靠阶段日志兜住。
boot("依赖导入完成 | 校验启动配置...")


# =====================================================================
# 任务注册表 + 结构化日志
#   - 任务注册表：req_id -> 文件名、模式、状态、入队与开始时刻。日志的队列视图、
#     最终报告的文件名、OOM 重试的复位都读它。
#   - 日志：带时间戳 + 级别 + req_id。启用 LOGGING_FILE 后走异步线程落盘，
#     调用线程只拼字符串入队，不碰文件句柄（见 async_logging.py）。
#   - 定义在 from pipeline import 之前：pipeline.py 与 cloud_hotword.py 都要
#     `from main import log, ...`。
# =====================================================================
_TASK_REGISTRY: dict = {}      # req_id -> {file, mode, status, enqueue_at, start_at}
_REG_LOCK = threading.Lock()

# 单次日志调用保留的任务明细上限。超过的部分只进计数不进明细，避免一次批量提交
# 把几十行拼进同一行日志里。
_QUEUE_DETAIL_LIMIT = 12

_STATUS_WAITING = "waiting"
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_FAILED = "failed"

# 当时最多允许并行的请求数。日志里报「N/M」要知道 M，故在 lifespan 里写入。
_PARALLEL_CAPACITY = 0


def register(req_id: str, file_name: str, mode: str) -> float:
    """入队登记，返回入队时刻。

    入队时刻是「排队时长」的唯一基准。前面的请求数由 task_queue_snapshot() 在
    同一次调用里给出，两处不必各算一遍。
    """
    with _REG_LOCK:
        now = _time.time()
        _TASK_REGISTRY[req_id] = {
            "file": file_name or "(未命名)",
            "mode": mode,
            "status": _STATUS_WAITING,
            "enqueue_at": now,
            "start_at": None,
        }
        return now


def _set_status(req_id: str, new_status: str) -> float | None:
    """改状态，返回本次状态切换的时刻（没有该任务时返回 None）。"""
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        if not info:
            return None
        now = _time.time()
        info["status"] = new_status
        if new_status == _STATUS_RUNNING and info["start_at"] is None:
            info["start_at"] = now
        return now


def mark_running(req_id: str) -> float | None:
    return _set_status(req_id, _STATUS_RUNNING)


def mark_done(req_id: str) -> float | None:
    return _set_status(req_id, _STATUS_DONE)


def mark_failed(req_id: str) -> float | None:
    return _set_status(req_id, _STATUS_FAILED)


def reset_for_retry(req_id: str) -> None:
    """OOM 等场景下，把任务状态拉回 WAITING 用于下一轮重试，保留原 req_id 与文件信息。"""
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        if not info:
            return
        info["status"] = _STATUS_WAITING
        info["enqueue_at"] = _time.time()
        info["start_at"] = None


def unregister(req_id: str) -> None:
    with _REG_LOCK:
        _TASK_REGISTRY.pop(req_id, None)


def _count_earlier_requests(req_id: str) -> int:
    """比本请求更早入队、且仍在运行或等待的请求数。

    口径与「排队」时长一致：都从 enqueue_at 起算。OOM 重试会把 enqueue_at 刷新，
    重试中的请求按重新排队计，与它在调度里的实际位置一致。
    """
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        if not info:
            return 0
        mine = info["enqueue_at"]
        return sum(
            1
            for other in _TASK_REGISTRY.values()
            if other["enqueue_at"] < mine
            and other["status"] in (_STATUS_RUNNING, _STATUS_WAITING)
        )


def note_audio_file(req_id: str, file_name: str) -> None:
    """计算区里发现真实音频文件名时回填到注册表。

    请求入口拿到的 `UploadFile.filename` 是调用方传的名字，可能为空或与实际落盘不符；
    流水线打开音频后知道真名，回填一次让后续的队列视图显示真名。
    """
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        if info is not None and file_name:
            info["file"] = file_name


def _ts() -> str:
    """毫秒精度本地时间戳，形如 `2026-07-22 14:23:45.123`。"""
    t = _time.time()
    lt = _time.localtime(t)
    return _time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{int((t - int(t)) * 1000):03d}"


def _fmt_dur(secs) -> str:
    if secs is None:
        return "-"
    if secs < 0.01:
        return "0s"
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m):02d}m"


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_frozen() -> bool:
    """Frozen World 是否正在暂停入计算区。事件对象定义在 CONFIG 之后，这里按名取。"""
    event = globals().get("world_frozen_event")
    return event is not None and not event.is_set()


def task_queue_snapshot() -> str:
    """当前任务视图，一行。

    形如 `任务 5 | 运行2/8 等待3 | 运行: a A 1.2s ; b B 0.4s | 等待: c C 3.0s`。
    只列当前在跑与在等的请求；已完成、失败的请求由各自的完成/失败日志给出，
    不留在快照里冒充积压。
    """
    with _REG_LOCK:
        items = [(_req_id, dict(info)) for _req_id, info in _TASK_REGISTRY.items()]

    now = _time.time()
    running = [item for item in items if item[1]["status"] == _STATUS_RUNNING]
    waiting = [item for item in items if item[1]["status"] == _STATUS_WAITING]
    # done/failed 的请求还挂在注册表里等出队，但它们已封口，不参与「还要等多久」
    # 的判断。列出来只会让「收尾 N」被误读成「N 个卡住了」。
    # Frozen World 期间所有等待都是显存清理期间的挂起，不是算力排队，必须标出来。
    frozen = _is_frozen()

    if not items:
        return "任务 0"

    def render(entries: list, is_running: bool) -> list[str]:
        entries.sort(key=lambda item: (item[1]["start_at"] or item[1]["enqueue_at"])
                     if is_running else item[1]["enqueue_at"])
        lines = []
        for req_id, info in entries[:_QUEUE_DETAIL_LIMIT]:
            since = (info["start_at"] or info["enqueue_at"]) if is_running else info["enqueue_at"]
            lines.append(
                f"{req_id} {_short(info['mode'], 24)} {_fmt_dur(now - since)} {_short(info['file'], 28)}"
            )
        if len(entries) > _QUEUE_DETAIL_LIMIT:
            lines.append(f"…其余 {len(entries) - _QUEUE_DETAIL_LIMIT} 个")
        return lines

    capacity = f"/{_PARALLEL_CAPACITY}" if _PARALLEL_CAPACITY else ""
    summary = f"运行{len(running)}{capacity} 等待{len(waiting)}"
    if frozen:
        summary += " 冻结中"

    blocks = [summary]
    if running:
        blocks.append("| 运行: " + " ; ".join(render(running, True)))
    if waiting:
        blocks.append("| 等待: " + " ; ".join(render(waiting, False)))
    return "任务 " + str(len(items)) + " | " + " ".join(blocks)


# 队列满时可以丢的只有这些级别。WARN 及以上一律不得无声消失。
_DROPPABLE_LEVELS = frozenset({"INFO", "DEBUG"})


def _emit_or_report(text: str, level: str) -> None:
    """入队；队列满且级别不可丢时同步写 stderr，绕开已经堵住的管道。"""
    if async_logging.emit(text):
        return
    if level in _DROPPABLE_LEVELS:
        return
    sys.stderr.write(text)
    sys.stderr.flush()


def log(*args, req_id: str = "", level: str = "INFO", sep: str = " ", end: str = "\n") -> None:
    """结构化日志。调用线程只拼字符串，落盘交给 async_logging 的后台线程。"""
    text = async_logging.truncate_if_needed(
        f"{_ts()} [{level:<5}] {f'[{req_id}]' if req_id else ''} {sep.join(str(a) for a in args)}{end}"
    )
    _emit_or_report(text, level)


def log_block(lines: list, req_id: str = "", level: str = "INFO") -> None:
    """多行原子输出：首行作头部（带 ts/level/req_id），其余缩进为 body。

    整块拼成一个字符串一次入队，并发时不与其他任务的行交错。
    截断按行做，不按整块：堆栈块里的正文在尾部，按整块截断会把最该看的部分砍掉。
    """
    if not lines:
        return
    head = f"{_ts()} [{level:<5}] {f'[{req_id}]' if req_id else ''} {lines[0]}"
    body = "".join(
        "\n  " + async_logging.truncate_if_needed(line + "\n").rstrip("\n")
        for line in lines[1:]
    )
    _emit_or_report(async_logging.truncate_if_needed(head + "\n").rstrip("\n") + body + "\n", level)


def log_info(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="INFO", **kw)


def log_warn(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="WARN", **kw)


def log_error(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="ERROR", **kw)


def log_fatal(*args, **kw) -> None:
    log(*args, level="FATAL", **kw)


def install_async_logging(log_path: str) -> None:
    """接管日志输出。WARN 及以上立即 flush，INFO 攒批，见 async_logging.py。"""
    async_logging.install(log_path)


def _failure_elapsed(enqueued_at: float, started_at: float | None = None) -> str:
    """失败落点统一的耗时前缀。答出「等了多久、跑了多久、总共多久」。

    排队阶段就失败的请求没有 started_at，此时只报总计。
    """
    now = _time.time()
    if started_at is None:
        return f"请求总计 {_fmt_dur(now - enqueued_at)}"
    return (
        f"排队 {_fmt_dur(started_at - enqueued_at)}"
        f" | 计算 {_fmt_dur(now - started_at)}"
        f" | 总计 {_fmt_dur(now - enqueued_at)}"
    )


def _log_if_still_waiting(req_id: str, file_name: str | None) -> None:
    """出队前兜底：状态还是 WAITING 说明请求在进入计算区之前就结束了。

    正常路径到这里状态已经是 done / failed / 校验阶段已置 failed，函数什么也不做。
    """
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        still_waiting = info is not None and info["status"] == _STATUS_WAITING
        enqueued_at = info["enqueue_at"] if info else None
    if not still_waiting:
        return
    mark_failed(req_id)
    log_error(
        f"❌ 请求在进入计算区之前结束 | {file_name or '(未命名)'} | "
        f"{_failure_elapsed(enqueued_at)}",
        req_id=req_id,
    )


def _log_rejected(req_id: str, reason: str, file_name: str | None) -> None:
    """请求在进入计算区之前被参数校验拒绝。

    拒绝发生在登记之后，所以入口行与队列快照里能看到它；这里补一行说明原因，
    并用 mark_failed 立即封口，不让它继续占用等待位。
    """
    mark_failed(req_id)
    log_warn(
        f"⛔ 请求被拒绝（未进入计算区） | {file_name or '(未命名)'} | {reason}",
        req_id=req_id,
    )


# 启动即校验环境变量、依赖锁与硬件能力：任一项不通过都直接拒绝启动，
# 不留到请求中途炸。放在业务模块导入之前 —— 校验失败时无需先付 pipeline 的导入代价。
_HARDWARE_RECEIPT = ""
try:
    SERVER_ENV = load_server_env()
    _startup_problems: list[str] = []
    check_lockfile(_startup_problems)
    _HARDWARE_RECEIPT = check_device_capability(SERVER_ENV.device, _startup_problems)
    if _startup_problems:
        raise StartupConfigError(_startup_problems)
except StartupConfigError as config_error:
    log_block(["启动检查未通过，服务拒绝启动"] + config_error.problems, level="FATAL")
    sys.exit(1)

# 校验通过才知道日志文件路径，此时才接管输出。此后的每一行都会同时进文件。
install_async_logging(SERVER_ENV.logging_file)
boot(f"启动检查通过 | 目标监听 {SERVER_ENV.host}:{SERVER_ENV.port} | 导入对齐 pipeline（静默阶段）...")

# [修改点 1]：引入我们新定义的 AlignmentThresholdError
from pipeline import LongAudioAlignPipeline, AlignmentThresholdError, build_sampling_params
from cloud_hotword import CloudHotwordAlignPipeline, AudioDefectError, resolve_language as resolve_cloud_language
from fa_timestamp_validation import FaTimestampError
boot("业务模块导入完成 | 组装 FastAPI 应用")

CONFIG = {
    "cpu_ratio": 0.7,
    "chunk_max_sec": 60,
    "vad_padding_sec": 0.3,
    "min_tail_sec": 30,         # 首尾 chunk 最短时长，过短则向前/后合并，避免残片让 FA 验证失效
    "language": "zh",
}

pipeline_instance = None
cloud_pipeline_instance = None
gpu_semaphore = None

VALID_ALIGNMENT_MODELS = [
    "Qwen/Qwen3-ASR-0.6B + Qwen/Qwen3-ForcedAligner-0.6B",
]
VALID_FA_ONLY_MODELS = [
    "Qwen/Qwen3-ForcedAligner-0.6B",
]
VALID_ASR_MODELS = [
    "Qwen/Qwen3-ASR-0.6B-hf", 
    "Qwen3-ASR-0.6B-hf", 
    "Qwen/Qwen3-ASR-0.6B", 
    "Qwen3-ASR-0.6B"
]
VALID_CLOUD_HOTWORD_MODELS = [
    "qwen-audio-3.1-asr-flash-filetrans + Qwen3-ForcedAligner-0.6B",
]
CLOUD_HOTWORD_SUPPORTED_LANGUAGES = {"zh", "en"}

world_frozen_event = asyncio.Event()
world_frozen_event.set() 

active_gpu_requests = 0
gpu_state_lock = asyncio.Lock()
stw_lock = asyncio.Lock()

async def trigger_frozen_world():
    global active_gpu_requests
    if stw_lock.locked(): return
        
    async with stw_lock:
        world_frozen_event.clear()
        log_fatal(f"检测到碎片化 OOM | 触发 Frozen World，暂停进入计算区 | {task_queue_snapshot()}")
        while active_gpu_requests > 0: await asyncio.sleep(0.5)

        log_fatal("核心计算区已清零 | 开始执行物理级显存清理...")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        log_fatal("清理完毕 | 解除 Frozen World，恢复队列处理。")
        world_frozen_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, cloud_pipeline_instance, gpu_semaphore
    
    log_info("服务初始化中...")
    log_block(["启动配置"] + SERVER_ENV.describe() + ([_HARDWARE_RECEIPT] if _HARDWARE_RECEIPT else []))
    boot("正在加载分词词典（jieba）...")
    jieba_at = _time.time()
    jieba.initialize()
    log_info(f"分词词典加载完成 | 耗时 {_time.time() - jieba_at:.1f}s")
    boot("正在加载本地模型权重（Silero VAD + Qwen3-ASR + ForcedAligner，静默阶段）...")
    load_at = _time.time()
    pipeline_instance = LongAudioAlignPipeline(CONFIG)
    log_info(f"本地模型加载完成 | 耗时 {_time.time() - load_at:.1f}s")
    cloud_pipeline_instance = CloudHotwordAlignPipeline(
        fa_model=pipeline_instance.fa_model,
        fa_lock=pipeline_instance.fa_lock,
        config=CONFIG,
    )
    
    if pipeline_instance.device == "cpu" or not torch.cuda.is_available():
        cores = get_container_cpu_count()
        max_concurrency_by_cpu = max(1, cores * CONFIG["cpu_ratio"])
        available_ram_gb = get_container_available_memory() / (1024 ** 3)
        estimated_ram_per_req_gb = 1.5 
        max_concurrency_by_ram = max(1, int(available_ram_gb // estimated_ram_per_req_gb))
        final_concurrency = min(max_concurrency_by_cpu, max_concurrency_by_ram)
        gpu_semaphore = asyncio.Semaphore(final_concurrency)
        
        ready_device = f"CPU 模式 | {cores}核 | {available_ram_gb:.1f}GB RAM"
        ready_concurrency = final_concurrency
        concurrency_basis = (
            f"CPU={max_concurrency_by_cpu} (核数×cpu_ratio)"
            f" RAM={max_concurrency_by_ram} ({available_ram_gb:.1f}GB ÷ {estimated_ram_per_req_gb}GB/请求)"
        )
    else:
        log_info("GPU 显存探针 (Warm-up)...")
        torch.cuda.reset_peak_memory_stats()
        static_vram = torch.cuda.memory_allocated()
        log_info(f"静态显存: {static_vram / (1024**3):.2f} GB")
        
        dummy_audio = np.random.randn(16000 * 30).astype(np.float32) * 0.01
        dummy_tuple = (dummy_audio, 16000)
        
        log_info("ASR 预热推理...")
        try: pipeline_instance.asr_model.transcribe(audio=dummy_tuple, language="zh")
        except Exception: pass
            
        log_info("FA 预热推理...")
        try: pipeline_instance.fa_model.align(audio=dummy_tuple, text="假", language="zh")
        except Exception: pass 
            
        peak_vram = torch.cuda.max_memory_allocated()
        dynamic_vram_per_req = peak_vram - static_vram
        estimated_dynamic_vram = max(dynamic_vram_per_req, 1.5 * 1024**3) 
        log_info(f"单请求峰值: {estimated_dynamic_vram / (1024**3):.2f} GB")
        
        total_vram = torch.cuda.get_device_properties(0).total_memory
        safe_buffer = 1 * 1024**3 
        available_vram = total_vram - static_vram - safe_buffer
        
        max_concurrency = max(1, int(available_vram // estimated_dynamic_vram))
        gpu_semaphore = asyncio.Semaphore(max_concurrency)

        ready_device = f"GPU 模式 | {torch.cuda.get_device_name(0)}"
        ready_concurrency = max_concurrency
        concurrency_basis = (
            f"显存={available_vram / (1024**3):.2f}GB ÷ 单请求峰值 "
            f"{estimated_dynamic_vram / (1024**3):.2f}GB"
        )

    global _PARALLEL_CAPACITY
    _PARALLEL_CAPACITY = ready_concurrency
    log_block([
        f"🚀 模型与并发初始化完成 | {ready_device}",
        f"并行上限 : {ready_concurrency}（取较小值：{concurrency_basis}）",
        f"日志文件 : {async_logging.file_path()}",
        f"监听地址 : http://{SERVER_ENV.host}:{SERVER_ENV.port}（本机访问用 127.0.0.1）",
    ])
    log_info("下方出现 uvicorn 的 `Uvicorn running on ...` 行即为真正启动完成，可接收请求")

    yield
    log_info(f"服务正在关闭，清理资源 | {task_queue_snapshot()}")


app = FastAPI(lifespan=lifespan, title="Long Audio Aligner API")

@app.post("/v1/audio/transcriptions")
async def openai_asr_endpoint(
    file: UploadFile = File(..., description="要处理的音频文件"),
    model: str = Form(..., description="必须指定合法的 ASR、ASR+ForcedAligner 或 ForcedAligner 模型名称"),
    prompt: str = Form(default="", description="对齐模式(FA)必填，纯ASR模式选填/忽略"),
    response_format: str = Form(default="json", description="支持 json 或 verbose_json"),
    language: str = Form(default=None, description="语言"),
    temperature: float = Form(default=None, description="采样温度（0=贪心；>0 采样；仅纯ASR模式生效）"),
    top_p: float = Form(default=None, description="top-p 采样（可选；仅纯ASR模式生效）"),
    n: int = Form(default=None, description="一次返回 n 条候选（可选；仅纯ASR模式生效）"),
):
    global active_gpu_requests
    
    req_id = str(uuid.uuid4())[:8]
    # multipart 的字段值会被调用方（Node FormData 实测如此）编码成 CRLF。参考文本是
    # 「调用方原文的逐字副本」，必须在入口还原成 LF，否则每行多一个字符，词级时间戳
    # 与 chunk 下标都会与调用方持有的原文错位。
    prompt = normalize_multipart_line_endings(prompt)
    is_alignment_mode = (model in VALID_ALIGNMENT_MODELS)
    is_fa_only_mode = (model in VALID_FA_ONLY_MODELS)
    is_cloud_hotword_mode = (model in VALID_CLOUD_HOTWORD_MODELS)
    if is_cloud_hotword_mode:
        mode_str = "CloudHotword"
    elif is_alignment_mode:
        mode_str = "ASR+ForcedAligner"
    elif is_fa_only_mode:
        mode_str = "ForcedAligner"
    else:
        mode_str = "ASR"
    
    enqueued_at = register(req_id, file.filename or "(未命名)", mode_str)
    # 只统计更早入队的请求：运行中和更早的等待中都算，口径与下一行的「排队」一致。
    ahead = _count_earlier_requests(req_id)
    log_info(
        f"📥 接收请求 | 模式: {mode_str} | 文件: {file.filename or '(未命名)'} | "
        f"排在 {ahead} 个之后 | {task_queue_snapshot()}",
        req_id=req_id,
    )

    tmp_path = None
    try:
        valid_models = (
            VALID_ALIGNMENT_MODELS
            + VALID_FA_ONLY_MODELS
            + VALID_ASR_MODELS
            + VALID_CLOUD_HOTWORD_MODELS
        )
        if model not in valid_models:
            _log_rejected(req_id, "Unsupported model", file.filename)
            raise HTTPException(status_code=400, detail={"error": {"message": f"Bad Request: Unsupported model '{model}'.", "type": "invalid_request_error"}})

        if is_cloud_hotword_mode:
            try:
                resolve_cloud_language(language)
            except ValueError as lang_err:
                _log_rejected(req_id, "Unsupported language", file.filename)
                raise HTTPException(status_code=400, detail={"error": {"message": str(lang_err), "type": "invalid_request_error"}})

        if is_alignment_mode or is_fa_only_mode or is_cloud_hotword_mode:
            if not prompt or not prompt.strip():
                _log_rejected(req_id, "prompt 为空", file.filename)
                raise HTTPException(status_code=400, detail={"error": {"message": "The 'prompt' parameter is required.", "type": "invalid_request_error"}})
            if response_format != "verbose_json":
                _log_rejected(req_id, f"response_format={response_format}", file.filename)
                raise HTTPException(status_code=400, detail={"error": {"message": "Only 'verbose_json' response_format is supported.", "type": "invalid_request_error"}})
            if temperature is not None or top_p is not None or n is not None:
                _log_rejected(req_id, "对齐模式不接受采样参数", file.filename)
                raise HTTPException(status_code=400, detail={"error": {"message": "Sampling params (temperature/top_p/n) are only valid in ASR mode.", "type": "invalid_request_error"}})

        if temperature is not None and not (0.0 <= temperature <= 2.0):
            _log_rejected(req_id, f"temperature={temperature}", file.filename)
            raise HTTPException(status_code=400, detail={"error": {"message": "temperature must be in [0, 2].", "type": "invalid_request_error"}})
        if top_p is not None and not (0.0 < top_p <= 1.0):
            _log_rejected(req_id, f"top_p={top_p}", file.filename)
            raise HTTPException(status_code=400, detail={"error": {"message": "top_p must be in (0, 1].", "type": "invalid_request_error"}})
        if n is not None and not (1 <= n <= 10):
            _log_rejected(req_id, f"n={n}", file.filename)
            raise HTTPException(status_code=400, detail={"error": {"message": "n must be in [1, 10].", "type": "invalid_request_error"}})

        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            max_retries = 3
            for attempt in range(max_retries):
                await world_frozen_event.wait()
                try:
                    async with gpu_semaphore:
                        # 排队时长取 mark_running 的返回值，不再回查注册表。重试轮次的
                        # 排队从 reset_for_retry 刷新的 enqueue_at 起算，故每轮重新取。
                        started_at = mark_running(req_id)
                        assert started_at is not None, "请求已登记，mark_running 必须返回时刻"
                        log_info(
                            f"▶️ 开始计算 | 模式: {mode_str} | 排队 {_fmt_dur(started_at - enqueued_at)} | "
                            f"排在 {_count_earlier_requests(req_id)} 个之后 | 尝试 {attempt + 1}/{max_retries} | "
                            f"{task_queue_snapshot()}",
                            req_id=req_id,
                        )

                        async with gpu_state_lock: active_gpu_requests += 1
                            
                        try:
                            if is_cloud_hotword_mode:
                                result = await run_in_threadpool(
                                    cloud_pipeline_instance.process_audio, audio_path=tmp_path, reference_text=prompt, language_code=language, req_id=req_id
                                )
                            elif is_alignment_mode:
                                result = await run_in_threadpool(
                                    pipeline_instance.process_audio, audio_path=tmp_path, reference_text=prompt, language_code=language, req_id=req_id
                                )
                            elif is_fa_only_mode:
                                result = await run_in_threadpool(
                                    pipeline_instance.process_audio_fa_only,
                                    audio_path=tmp_path,
                                    reference_text=prompt,
                                    language_code=language,
                                    req_id=req_id,
                                )
                            else:
                                sampling_params = build_sampling_params(
                                    pipeline_instance.asr_model,
                                    temperature=temperature,
                                    top_p=top_p,
                                    n=n,
                                )
                                if sampling_params is not None:
                                    log_info(
                                        f"🎛️ 采样参数 | temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, n={sampling_params.n}",
                                        req_id=req_id,
                                    )
                                result = await run_in_threadpool(
                                    pipeline_instance.process_audio_asr_only,
                                    audio_path=tmp_path,
                                    language_code=language,
                                    req_id=req_id,
                                    sampling_params=sampling_params,
                                )
                                
                            done_at = mark_done(req_id) or _time.time()
                            log_info(
                                f"✅ 计算完成 | 模式: {mode_str} | 排队 {_fmt_dur(started_at - enqueued_at)} | "
                                f"计算 {_fmt_dur(done_at - started_at)} | 总计 {_fmt_dur(done_at - enqueued_at)}",
                                req_id=req_id,
                            )
                            
                            if not is_alignment_mode and not is_fa_only_mode and not is_cloud_hotword_mode and response_format == "json":
                                return {"text": result.get("text", "")}
                            return result
                        finally:
                            async with gpu_state_lock: active_gpu_requests -= 1
                                
                except AlignmentThresholdError as e:
                    # 首尾对齐失败：打印诊断 + 透传详细报错给前端
                    mark_failed(req_id)
                    log_warn(
                        f"⚠️ 对齐匹配熔断（首尾 chunk 对齐失败） | {_failure_elapsed(enqueued_at, started_at)}",
                        req_id=req_id,
                    )
                    log_warn(str(e), req_id=req_id)
                    raise HTTPException(status_code=400, detail={
                        "error": {
                            "message": with_chapter_identity(str(e), file.filename),
                            "type": "alignment_threshold_error",
                            "file": file.filename or "(未命名)",
                        }
                    })

                except FaTimestampError as e:
                    # 运行期 FA 对齐失败（时间戳校验未通过）：不是请求参数问题，
                    # 但同一输入确定性复现，重试无意义，故返回 400 并给专属 type。
                    # 与 AlignmentThresholdError 同处置，不再走通用 ValueError 分支
                    # 被误标成「前置校验」。
                    mark_failed(req_id)
                    first_line = str(e).split("\n", 1)[0]
                    log_error(
                        f"⚠️ FA 对齐失败（时间戳校验未通过） | {_failure_elapsed(enqueued_at, started_at)} | {first_line}",
                        req_id=req_id,
                    )
                    raise HTTPException(status_code=400, detail={
                        "error": {
                            "message": with_chapter_identity(str(e), file.filename),
                            "type": "fa_timestamp_error",
                            "file": file.filename or "(未命名)",
                        }
                    })

                except Exception as e:
                    is_oom = False
                    try:
                        if isinstance(e, torch.cuda.OutOfMemoryError): is_oom = True
                    except AttributeError: pass
                    if "out of memory" in str(e).lower(): is_oom = True

                    if is_oom:
                        if attempt == max_retries - 1:
                            mark_failed(req_id)
                            log_error(
                                f"严重错误 | GPU OOM 彻底耗尽 | {_failure_elapsed(enqueued_at, started_at)}",
                                req_id=req_id,
                            )
                            raise HTTPException(status_code=500, detail={"error": {"message": "GPU 资源耗尽。", "type": "server_error"}})

                        mark_failed(req_id)
                        reset_for_retry(req_id)
                        enqueued_at = _time.time()
                        log_warn(
                            f"拦截 OOM 发生 | 当前请求转入挂起重试状态 ({attempt + 1}/{max_retries}) | "
                            f"{_failure_elapsed(enqueued_at, started_at)} | {task_queue_snapshot()}",
                            req_id=req_id,
                        )
                        del e
                        asyncio.create_task(trigger_frozen_world())
                        await asyncio.sleep(2)
                    else:
                        mark_failed(req_id)
                        if isinstance(e, AudioDefectError):
                            # 音频缺陷：裁决的是这份音频（覆盖率不足等），不是请求参数。
                            # 必须与普通 400 分开类型 —— 调用方据此作废本章产物并重生成音频；
                            # 混进 invalid_request_error 会被当成参数问题直接记失败，
                            # 回退重跑额度永远用不上（2026-10-28 第 34 章）。
                            first_line = str(e).split("\n", 1)[0]
                            log_error(
                                f"请求被前置校验拒绝（音频缺陷） | {file.filename or '(未命名)'} | "
                                f"reason={e.reason} | {_failure_elapsed(enqueued_at, started_at)} | {first_line}",
                                req_id=req_id,
                            )
                            raise HTTPException(status_code=400, detail=audio_defect_error_detail(
                                str(e), file.filename, e.reason,
                            ))
                        if isinstance(e, ValueError):
                            # ValueError 一律是业务前置条件不满足（覆盖率校验、热词不足等），
                            # 属可预期的 400，打堆栈只会让人误判为代码 bug；
                            # 详细诊断已由抛错处 log_block 过，这里只补一行落点指引。
                            first_line = str(e).split("\n", 1)[0]
                            log_error(
                                f"请求被前置校验拒绝 | {file.filename or '(未命名)'} | {type(e).__name__}: {first_line} | "
                                f"{_failure_elapsed(enqueued_at, started_at)}",
                                req_id=req_id,
                            )
                            raise HTTPException(status_code=400, detail={
                                "error": {
                                    "message": with_chapter_identity(str(e), file.filename),
                                    "type": "invalid_request_error",
                                    "file": file.filename or "(未命名)",
                                }
                            })

                        # 其他是真正的代码层面 Bug。堆栈必须和这一行走同一条落盘通道，
                        # 否则文件里只有一行「下方为代码堆栈」，堆栈却散在 stderr。
                        log_error(
                            f"运行时发生未捕获异常 | {_failure_elapsed(enqueued_at, started_at)} | {type(e).__name__}: {e}",
                            req_id=req_id,
                        )
                        log_block(
                            ["堆栈"] + traceback.format_exc().rstrip().split("\n"),
                            req_id=req_id, level="ERROR",
                        )

                        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        # 排队阶段就抛错（读文件失败、临时文件建不出来）的请求不会进上面那个
        # try/except，缺了这一行日志就彻底没有痕迹。
        _log_if_still_waiting(req_id, file.filename)
        # 无论成功/失败/校验未通过，最终出队
        unregister(req_id)

if __name__ == "__main__":
    boot("进入 uvicorn.run | 模型加载完成后，`Uvicorn running on ...` 一行即真正启动")
    # log_level=info：保留 uvicorn 自己的 startup 行（Waiting for application startup /
    # Application startup complete / Uvicorn running on）——只有它能在端口 bind 成功后才打印，
    # 是"服务可接收请求"的唯一权威标记。warning 会把这几行连同启动期一起吞掉，表现为进程无反应。
    # 请求级日志仍由本文件的结构化日志承担，故关掉 uvicorn 的 access log。
    uvicorn.run(app, host=SERVER_ENV.host, port=SERVER_ENV.port, log_level="info", access_log=False)
