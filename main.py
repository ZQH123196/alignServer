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
    """单行紧凑队列快照：`[queue N: rX wY dZ fW] [🔄req|mode|dur|file] ...`"""
    with _REG_LOCK:
        items = list(_TASK_REGISTRY.items())
    if not items:
        return "[queue empty]"

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
        emoji = _EMOJI.get(info["status"], "•")
        parts.append(
            f"{emoji}[{rid}|{_short(info['mode'], 8)}|{_fmt_dur(dur):>7}|{_short(info['file'], 32)}]"
        )
    total = len(items)
    summary = (
        f"[queue {total}: r={counters.get('running', 0)} "
        f"w={counters.get('waiting', 0)} "
        f"done={counters.get('done', 0)} "
        f"fail={counters.get('failed', 0)}]"
    )
    return summary + " " + " ".join(parts)


def log(*args, req_id: str = "", level: str = "INFO", sep: str = " ",
        end: str = "\n", flush: bool = True) -> None:
    """替代 print 的结构化日志。

    自动为每条日志附加：
      - 毫秒精度时间戳（`2026-07-22 14:23:45.123`）
      - 日志级别（INFO / WARN / ERROR / FATAL）
      - 可选 req_id 上下文（调用时 `req_id=req_id`）
      - 当前任务队列紧凑快照（每个 req 的状态、模式、耗时、文件名）

    便捷包装：
      log_info(msg, req_id=...) / log_warn(...) / log_error(...) / log_fatal(...)
    """
    msg = sep.join(str(a) for a in args)
    ts = _ts()
    ctx = f"[req={req_id}]" if req_id else ""
    queue = _queue_snapshot()
    level_str = f"[{level.upper():<5}]"
    with _PRINT_LOCK:
        try:
            sys.stdout.write(
                f"{ts} {level_str} {ctx} {queue} | {msg}{end}"
            )
            if flush:
                sys.stdout.flush()
        except Exception:
            try:
                sys.stdout.write(f"{ts} {level_str} {msg}{end}")
                if flush:
                    sys.stdout.flush()
            except Exception:
                pass


def log_info(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="INFO", **kw)


def log_warn(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="WARN", **kw)


def log_error(*args, req_id: str = "", **kw) -> None:
    log(*args, req_id=req_id, level="ERROR", **kw)


def log_fatal(*args, **kw) -> None:
    log(*args, level="FATAL", **kw)


# [修改点 1]：引入我们新定义的 AlignmentThresholdError
from pipeline import LongAudioAlignPipeline, AlignmentThresholdError

load_dotenv()

CONFIG = {
    "device": "cpu",            
    "cpu_ratio": 0.7,             
    "chunk_max_sec": 60,        
    "vad_padding_sec": 0.3,     
    "language": "zh",
}

pipeline_instance = None
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
        log_fatal("🛑 检测到碎片化 OOM | 触发 Frozen World！暂停进入计算区，后续请求进入排队队列...")
        while active_gpu_requests > 0: await asyncio.sleep(0.5)

        log_fatal("🧹 核心计算区已清零 | 开始执行物理级显存清理...")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        log_fatal("✅ 清理完毕 | 解除 Frozen World，恢复队列处理。")
        world_frozen_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, gpu_semaphore
    
    print("\n🚀 开始 Server 生命周期初始化...")
    pipeline_instance = LongAudioAlignPipeline(CONFIG)
    
    if pipeline_instance.device == "cpu" or not torch.cuda.is_available():
        cores = get_container_cpu_count()
        max_concurrency_by_cpu = max(1, cores * CONFIG["cpu_ratio"])
        available_ram_gb = get_container_available_memory() / (1024 ** 3)
        estimated_ram_per_req_gb = 1.5 
        max_concurrency_by_ram = max(1, int(available_ram_gb // estimated_ram_per_req_gb))
        final_concurrency = min(max_concurrency_by_cpu, max_concurrency_by_ram)
        gpu_semaphore = asyncio.Semaphore(final_concurrency)
        
        print(f"[*] 运行在 CPU 模式")
        print(f"  -> 系统总核心数: {cores}，物理内存可用: {available_ram_gb:.2f} GB")
        print(f"  -> 算力限制并发: {max_concurrency_by_cpu} | 内存限制并发: {max_concurrency_by_ram}")
        print(f"✅ 综合短板效应，已自动设置安全并发数为: {final_concurrency}\n")
    else:
        print("\n🤖 正在执行 GPU 动态显存探针测试 (Warm-up)...")
        torch.cuda.reset_peak_memory_stats()
        static_vram = torch.cuda.memory_allocated()
        print(f"  -> 模型静态显存底座占用: {static_vram / (1024**3):.2f} GB")
        
        dummy_audio = np.random.randn(16000 * 30).astype(np.float32) * 0.01
        dummy_tuple = (dummy_audio, 16000)
        
        try: pipeline_instance.asr_model.transcribe(audio=dummy_tuple, language="zh")
        except Exception: pass
            
        try: pipeline_instance.fa_model.align(audio=dummy_tuple, text="假", language="zh")
        except Exception: pass 
            
        peak_vram = torch.cuda.max_memory_allocated()
        dynamic_vram_per_req = peak_vram - static_vram
        estimated_dynamic_vram = max(dynamic_vram_per_req, 1.5 * 1024**3) 
        print(f"  -> 预估单次请求动态显存峰值: {estimated_dynamic_vram / (1024**3):.2f} GB")
        
        total_vram = torch.cuda.get_device_properties(0).total_memory
        safe_buffer = 1 * 1024**3 
        available_vram = total_vram - static_vram - safe_buffer
        
        max_concurrency = max(1, int(available_vram // estimated_dynamic_vram))
        print(f"✅ 探针计算完毕！当前硬件 ({torch.cuda.get_device_name(0)}) 最佳安全并发数为: {max_concurrency}\n")
        gpu_semaphore = asyncio.Semaphore(max_concurrency)

    yield 
    print("🛑 Server 正在关闭，清理资源...")


app = FastAPI(lifespan=lifespan, title="Long Audio Aligner API")

@app.post("/v1/audio/transcriptions")
async def openai_asr_endpoint(
    file: UploadFile = File(..., description="要处理的音频文件"),
    model: str = Form(..., description="强制要求：必须指定合法的 ForcedAligner 或 ASR 模型名称"),
    prompt: str = Form(default="", description="对齐模式(FA)必填，纯ASR模式选填/忽略"),
    response_format: str = Form(default="json", description="支持 json 或 verbose_json"),
    language: str = Form(default=None, description="语言")
):
    global active_gpu_requests
    
    req_id = str(uuid.uuid4())[:8]
    is_alignment_mode = (model in VALID_ALIGNMENT_MODELS)
    mode_str = "ForcedAligner" if is_alignment_mode else "ASR"
    
    # 先入册，再打印"接收请求"，这样第一行日志就已经能看到队列里多了一条 WAITING
    register(req_id, file.filename or "(未命名)", mode_str)
    log_info(
        f"📥 接收请求 | 模式: {mode_str} | 文件: {file.filename or '(未命名)'}",
        req_id=req_id,
    )

    tmp_path = None
    try:
        valid_models = VALID_ALIGNMENT_MODELS + VALID_ASR_MODELS
        if model not in valid_models:
            mark_failed(req_id)
            raise HTTPException(status_code=400, detail={"error": {"message": f"Bad Request: Unsupported model '{model}'.", "type": "invalid_request_error"}})

        if is_alignment_mode:
            if not prompt or not prompt.strip():
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": "The 'prompt' parameter is required.", "type": "invalid_request_error"}})
            if response_format != "verbose_json":
                mark_failed(req_id)
                raise HTTPException(status_code=400, detail={"error": {"message": "Only 'verbose_json' response_format is supported.", "type": "invalid_request_error"}})

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
                            f"▶️ 进入并行计算区 | 尝试 {attempt + 1}/{max_retries}",
                            req_id=req_id,
                        )

                        async with gpu_state_lock: active_gpu_requests += 1
                            
                        try:
                            if is_alignment_mode:
                                result = await run_in_threadpool(
                                    pipeline_instance.process_audio, audio_path=tmp_path, reference_text=prompt, language_code=language, req_id=req_id
                                )
                            else:
                                result = await pipeline_instance.process_audio_asr_only(
                                    audio_path=tmp_path, language_code=language, req_id=req_id
                                )
                                
                            mark_done(req_id)
                            log_info(
                                f"✅ 计算完成 | 模式: {mode_str}",
                                req_id=req_id,
                            )
                            
                            if not is_alignment_mode and response_format == "json":
                                return {"text": result.get("text", "")}
                            return result
                        finally:
                            async with gpu_state_lock: active_gpu_requests -= 1
                                
                except AlignmentThresholdError as e:
                    # [核心修改 2]：精准拦截我们自定义的匹配失败错误，只打印有用的排查信息，不打印长长的代码报错栈
                    mark_failed(req_id)
                    log_warn(
                        f"⚠️ 对齐匹配熔断（音频与给定文本相似度过低）",
                        req_id=req_id,
                    )
                    log_warn(str(e), req_id=req_id)
                    raise HTTPException(status_code=400, detail={"error": {"message": "音频与给定文本相似度过低，无法完成对齐", "type": "alignment_threshold_error"}})

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
                                "🔴 严重错误 | GPU OOM 彻底耗尽。",
                                req_id=req_id,
                            )
                            raise HTTPException(status_code=500, detail={"error": {"message": "GPU 资源耗尽。", "type": "server_error"}})

                        mark_failed(req_id)
                        # 重置回 WAITING 状态以便下一轮重试时再次进入 RUNNING（保留原 req_id 与文件信息）
                        reset_for_retry(req_id)
                        log_warn(
                            f"→ 拦截 OOM 发生 | 当前请求转入挂起重试状态 ({attempt + 1}/{max_retries})",
                            req_id=req_id,
                        )
                        del e
                        asyncio.create_task(trigger_frozen_world())
                        await asyncio.sleep(2)
                    else:
                        mark_failed(req_id)
                        # 对于其他的真正的代码层面的 Bug，依旧保留堆栈打印方便排查
                        log_error(
                            "🔴 运行时发生未捕获异常（下方为代码堆栈）",
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
    print("="*50)
    print(f"🌐 Server 将在 http://{host}:{port} 启动")
    print("="*50)
    uvicorn.run(app, host=host, port=port, log_level="warning")