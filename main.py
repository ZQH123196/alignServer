import os
import json
import asyncio
import gc
import torch
import numpy as np
import uvicorn
import tempfile
import uuid  
import traceback  
import sys
import time as _time
import threading
from dotenv import load_dotenv

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from utils import get_container_cpu_count, get_container_available_memory
import transformers
transformers.logging.set_verbosity_error()


# =====================================================================
# [并发可视化] 任务注册表 + 结构化日志
#   - 目的：每次输出日志时附正规时间戳 + 日志级别 + req_id 上下文 +
#           "任务队列快照"，一目了然当前在排队的、正在并行的、以及它们各自的文件。
#   - 由 main.py / pipeline.py 共同使用；必须定义在 from pipeline import 之前，
#     以便 pipeline.py 也能 `from main import log, register, ...`。
# =====================================================================
_TASK_REGISTRY: dict = {}      # req_id -> {file, mode, status, enqueue_at, start_at}
_REG_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()

_STATUS_WAITING = "waiting"
_STATUS_RUNNING = "running"
_STATUS_DONE    = "done"
_STATUS_FAILED  = "failed"
_EMOJI = {
    _STATUS_WAITING: "⏳",
    _STATUS_RUNNING: "🔄",
    _STATUS_DONE:    "✅",
    _STATUS_FAILED:  "❌",
}
_STATUS_LABEL_CN = {
    _STATUS_WAITING: "等待",
    _STATUS_RUNNING: "运行",
    _STATUS_DONE:    "完成",
    _STATUS_FAILED:  "失败",
}


def register(req_id: str, file_name: str, mode: str) -> None:
    with _REG_LOCK:
        _TASK_REGISTRY[req_id] = {
            "file": file_name or "(未命名)",
            "mode": mode,
            "status": _STATUS_WAITING,
            "enqueue_at": _time.time(),
            "start_at": None,
        }


def _set_status(req_id: str, new_status: str) -> None:
    with _REG_LOCK:
        info = _TASK_REGISTRY.get(req_id)
        if not info:
            return
        info["status"] = new_status
        if new_status == _STATUS_RUNNING and info["start_at"] is None:
            info["start_at"] = _time.time()


def mark_running(req_id: str) -> None: _set_status(req_id, _STATUS_RUNNING)
def mark_done(req_id: str) -> None:    _set_status(req_id, _STATUS_DONE)
def mark_failed(req_id: str) -> None:  _set_status(req_id, _STATUS_FAILED)


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


def _ts() -> str:
    """毫秒精度本地时间戳，形如 `2026-07-22 14:23:45.123`。"""
    t = _time.time()
    lt = _time.localtime(t)
    return _time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{int((t - int(t)) * 1000):03d}"


def _fmt_dur(secs):
    if secs is None:
        return "-"
    if secs < 0.01:
        return "0s"
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(secs, 60)
    return f"{int(m)}m{s:.1f}s"


def _queue_snapshot() -> str:
    """单行可读队列快照：`队列 N | 运行X 等待Y 完成Z 失败W | [运行] req | 模式 | 时长 | 文件名 ...`"""
    with _REG_LOCK:
        items = list(_TASK_REGISTRY.items())
    if not items:
        return "队列为空"

    now = _time.time()
    counters = {"running": 0, "waiting": 0, "done": 0, "failed": 0}
    enriched = []
    for rid, info in items:
        counters[info["status"]] = counters.get(info["status"], 0) + 1
        base = (
            info["start_at"]
            if info["status"] == _STATUS_RUNNING and info["start_at"]
            else info["enqueue_at"]
        )
        enriched.append((rid, info, now - base))

    # 运行中优先，其余按 req_id 字母序
    enriched.sort(key=lambda x: (x[1]["status"] != _STATUS_RUNNING, x[0]))

    def _short(s: str, n: int) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    parts = []
    for rid, info, dur in enriched:
        label = _STATUS_LABEL_CN.get(info["status"], "?")
        parts.append(
            f"[{label}] {rid} | {_short(info['mode'], 16)} | {_fmt_dur(dur):>7} | {_short(info['file'], 32)}"
        )
    total = len(items)
    summary = (
        f"队列 {total} | 运行{counters.get('running', 0)} "
        f"等待{counters.get('waiting', 0)} "
        f"完成{counters.get('done', 0)} "
        f"失败{counters.get('failed', 0)}"
    )
    return summary + " | " + " | ".join(parts)


def log_queue_snapshot(req_id: str = "", level: str = "INFO") -> None:
    """独立打印一行队列快照，不再追加到其他日志行。"""
    log(_queue_snapshot(), req_id=req_id, level=level)


def get_task_info(req_id: str) -> dict | None:
    """从任务注册表取任务信息（含 file 文件名），供 pipeline 最终报告使用。"""
    with _REG_LOCK:
        return dict(_TASK_REGISTRY.get(req_id)) if req_id in _TASK_REGISTRY else None


def log(*args, req_id: str = "", level: str = "INFO", sep: str = " ",
        end: str = "\n", flush: bool = True, queue: bool = False) -> None:
    """结构化日志。queue 参数已废弃（保留签名兼容），队列快照改用 log_queue_snapshot 独占一行。"""
    msg = sep.join(str(a) for a in args)
    ts = _ts()
    ctx = f"[{req_id}]" if req_id else ""
    level_str = f"[{level.upper():<5}]"
    with _PRINT_LOCK:
        try:
            sys.stdout.write(f"{ts} {level_str} {ctx} {msg}{end}")
            if flush:
                sys.stdout.flush()
        except Exception:
            pass


def log_block(lines: list, req_id: str = "", level: str = "INFO") -> None:
    """多行原子输出：首行作头部（带 ts/level/req_id），其余缩进为 body。
    整块在 _PRINT_LOCK 内一次写入，并发时不与其他任务交错。"""
    if not lines:
        return
    ts = _ts()
    ctx = f"[{req_id}]" if req_id else ""
    level_str = f"[{level.upper():<5}]"
    header = lines[0]
    body = "\n".join(f"  {l}" for l in lines[1:])
    with _PRINT_LOCK:
        try:
            if body:
                sys.stdout.write(f"{ts} {level_str} {ctx} {header}\n{body}\n")
            else:
                sys.stdout.write(f"{ts} {level_str} {ctx} {header}\n")
            sys.stdout.flush()
        except Exception:
            pass


def log_info(*args, req_id: str = "", queue: bool = False, **kw) -> None:
    log(*args, req_id=req_id, level="INFO", queue=queue, **kw)


def log_warn(*args, req_id: str = "", queue: bool = False, **kw) -> None:
    log(*args, req_id=req_id, level="WARN", queue=queue, **kw)


def log_error(*args, req_id: str = "", queue: bool = False, **kw) -> None:
    log(*args, req_id=req_id, level="ERROR", queue=queue, **kw)


def log_fatal(*args, queue: bool = False, **kw) -> None:
    log(*args, level="FATAL", queue=queue, **kw)


# [修改点 1]：引入我们新定义的 AlignmentThresholdError
from pipeline import LongAudioAlignPipeline, AlignmentThresholdError, build_sampling_params
from cloud_hotword import CloudHotwordAlignPipeline

load_dotenv()

CONFIG = {
    "device": "cpu",            
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
    "Qwen/Qwen3-ForcedAligner-0.6B-hf", 
    "Qwen/Qwen3-ForcedAligner-0.6B", 
    "Qwen3-ForcedAligner-0.6B-hf", 
    "Qwen3-ForcedAligner-0.6B"
]
VALID_ASR_MODELS = [
    "Qwen/Qwen3-ASR-0.6B-hf", 
    "Qwen3-ASR-0.6B-hf", 
    "Qwen/Qwen3-ASR-0.6B", 
    "Qwen3-ASR-0.6B"
]
VALID_CLOUD_HOTWORD_MODELS = [
    "qwen-audio-3.0-asr-flash-filetrans + Qwen3-ForcedAligner-0.6B",
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
        log_fatal("检测到碎片化 OOM | 触发 Frozen World！暂停进入计算区，后续请求进入排队队列...")
        log_queue_snapshot(level="FATAL")
        while active_gpu_requests > 0: await asyncio.sleep(0.5)

        log_fatal("核心计算区已清零 | 开始执行物理级显存清理...")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        log_fatal("清理完毕 | 解除 Frozen World，恢复队列处理。")
        log_queue_snapshot(level="FATAL")
        world_frozen_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, cloud_pipeline_instance, gpu_semaphore
    
    log_info("服务初始化中...")
    pipeline_instance = LongAudioAlignPipeline(CONFIG)
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
    else:
        log_info("GPU 显存探针 (Warm-up)...")
        torch.cuda.reset_peak_memory_stats()
        static_vram = torch.cuda.memory_allocated()
        log_info(f"静态显存: {static_vram / (1024**3):.2f} GB")
        
        dummy_audio = np.random.randn(16000 * 30).astype(np.float32) * 0.01
        dummy_tuple = (dummy_audio, 16000)
        
        try: pipeline_instance.asr_model.transcribe(audio=dummy_tuple, language="zh")
        except Exception: pass
            
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

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 6008))
    log_info(f"🚀 服务已就绪 | {ready_device} | 并发={ready_concurrency} | 监听 http://{host}:{port}")

    yield 
    log_info("服务正在关闭，清理资源...")


app = FastAPI(lifespan=lifespan, title="Long Audio Aligner API")

@app.post("/v1/audio/transcriptions")
async def openai_asr_endpoint(
    file: UploadFile = File(..., description="要处理的音频文件"),
    model: str = Form(..., description="强制要求：必须指定合法的 ForcedAligner 或 ASR 模型名称"),
    prompt: str = Form(default="", description="对齐模式(FA)必填，纯ASR模式选填/忽略"),
    response_format: str = Form(default="json", description="支持 json 或 verbose_json"),
    language: str = Form(default=None, description="语言"),
    temperature: float = Form(default=None, description="采样温度（0=贪心；>0 采样；仅纯ASR模式生效）"),
    top_p: float = Form(default=None, description="top-p 采样（可选；仅纯ASR模式生效）"),
    n: int = Form(default=None, description="一次返回 n 条候选（可选；仅纯ASR模式生效）"),
):
    global active_gpu_requests
    
    req_id = str(uuid.uuid4())[:8]
    is_alignment_mode = (model in VALID_ALIGNMENT_MODELS)
    is_cloud_hotword_mode = (model in VALID_CLOUD_HOTWORD_MODELS)
    if is_cloud_hotword_mode:
        mode_str = "CloudHotword"
    elif is_alignment_mode:
        mode_str = "ForcedAligner"
    else:
        mode_str = "ASR"
    
    # 先入册，再打印"接收请求"，这样第一行日志就已经能看到队列里多了一条 WAITING
    register(req_id, file.filename or "(未命名)", mode_str)
    log_info(
        f"📥 接收请求 | 模式: {mode_str} | 文件: {file.filename or '(未命名)'}",
        req_id=req_id,
    )
    log_queue_snapshot(req_id=req_id)

    tmp_path = None
    try:
        valid_models = VALID_ALIGNMENT_MODELS + VALID_ASR_MODELS + VALID_CLOUD_HOTWORD_MODELS
        if model not in valid_models:
            mark_failed(req_id)
            raise HTTPException(status_code=400, detail={"error": {"message": f"Bad Request: Unsupported model '{model}'.", "type": "invalid_request_error"}})

        if is_cloud_hotword_mode:
            lang_key = (language or "").lower().strip().split("-")[0].split("_")[0]
            if lang_key not in CLOUD_HOTWORD_SUPPORTED_LANGUAGES:
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": f"Cloud hotword 分支仅支持中文(zh)和英文(en)，当前语言 '{language}' 不支持。", "type": "invalid_request_error"}})

        if is_alignment_mode or is_cloud_hotword_mode:
            if not prompt or not prompt.strip():
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": "The 'prompt' parameter is required.", "type": "invalid_request_error"}})
            if response_format != "verbose_json":
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": "Only 'verbose_json' response_format is supported.", "type": "invalid_request_error"}})
            if temperature is not None or top_p is not None or n is not None:
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": "Sampling params (temperature/top_p/n) are only valid in ASR mode.", "type": "invalid_request_error"}})

        if temperature is not None and not (0.0 <= temperature <= 2.0):
            mark_failed(req_id)
            raise HTTPException(status_code=400, detail={"error": {"message": "temperature must be in [0, 2].", "type": "invalid_request_error"}})
        if top_p is not None and not (0.0 < top_p <= 1.0):
            mark_failed(req_id)
            raise HTTPException(status_code=400, detail={"error": {"message": "top_p must be in (0, 1].", "type": "invalid_request_error"}})
        if n is not None and not (1 <= n <= 10):
            mark_failed(req_id)
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
                        # 进入并行计算区：标记为 RUNNING 并打印队列快照
                        mark_running(req_id)
                        log_info(
                            f"进入并行计算区 | 尝试 {attempt + 1}/{max_retries}",
                            req_id=req_id,
                        )
                        log_queue_snapshot(req_id=req_id)

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
                                
                            mark_done(req_id)
                            log_info(
                                f"✅ 计算完成 | 模式: {mode_str}",
                                req_id=req_id,
                            )
                            log_queue_snapshot(req_id=req_id)
                            
                            if not is_alignment_mode and not is_cloud_hotword_mode and response_format == "json":
                                return {"text": result.get("text", "")}
                            return result
                        finally:
                            async with gpu_state_lock: active_gpu_requests -= 1
                                
                except AlignmentThresholdError as e:
                    # 首尾对齐失败：打印诊断 + 透传详细报错给前端
                    mark_failed(req_id)
                    log_warn(
                        f"⚠️ 对齐匹配熔断（首尾 chunk 对齐失败）",
                        req_id=req_id,
                    )
                    log_queue_snapshot(req_id=req_id, level="WARN")
                    log_warn(str(e), req_id=req_id)
                    raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "alignment_threshold_error"}})

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
                                "严重错误 | GPU OOM 彻底耗尽。",
                                req_id=req_id,
                            )
                            log_queue_snapshot(req_id=req_id, level="ERROR")
                            raise HTTPException(status_code=500, detail={"error": {"message": "GPU 资源耗尽。", "type": "server_error"}})

                        mark_failed(req_id)
                        reset_for_retry(req_id)
                        log_warn(
                            f"拦截 OOM 发生 | 当前请求转入挂起重试状态 ({attempt + 1}/{max_retries})",
                            req_id=req_id,
                        )
                        log_queue_snapshot(req_id=req_id, level="WARN")
                        del e
                        asyncio.create_task(trigger_frozen_world())
                        await asyncio.sleep(2)
                    else:
                        mark_failed(req_id)
                        # 对于其他的真正的代码层面的 Bug，依旧保留堆栈打印方便排查
                        log_error(
                            "运行时发生未捕获异常（下方为代码堆栈）",
                            req_id=req_id,
                        )
                        traceback.print_exc()

                        if isinstance(e, ValueError): raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
                        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        # 无论成功/失败/校验未通过，最终出队
        unregister(req_id)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 6008))
    uvicorn.run(app, host=host, port=port, log_level="warning")
