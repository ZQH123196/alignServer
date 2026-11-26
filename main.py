import os
import json
import asyncio
import gc
import torch
import numpy as np
import uvicorn
import tempfile
from dotenv import load_dotenv

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from utils import get_container_cpu_count, get_container_available_memory
import transformers
transformers.logging.set_verbosity_error()
from pipeline import LongAudioAlignPipeline

load_dotenv()

CONFIG = {
    "device": "cpu",            
    "cpu_ratio": 0.7,             
    "chunk_max_sec": 60,        # [核心优化 1]：切片降至 60 秒，完美平衡匹配熵与 O(N^2) 算力消耗
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
    """触发 STW，执行物理清场"""
    global active_gpu_requests
    
    if stw_lock.locked():
        return
        
    async with stw_lock:
        world_frozen_event.clear()
        print("\n[FATAL] 🛑 检测到碎片化 OOM，触发 Frozen World! 暂停进入计算区，后续请求进入排队队列...")
        
        while active_gpu_requests > 0:
            await asyncio.sleep(0.5)
            
        print("[FATAL] 🧹 核心计算区已清零，开始执行物理级显存清理...")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("[FATAL] ✅ 清理完毕，解除 Frozen World，恢复队列处理。\n")
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
        
        # [核心优化 2]：探针预热直接使用内存数据，不再生成临时文件
        dummy_audio = np.random.randn(16000 * 30).astype(np.float32) * 0.01
        dummy_tuple = (dummy_audio, 16000)
        
        try:
            pipeline_instance.asr_model.transcribe(audio=dummy_tuple, language="zh")
        except Exception:
            pass
            
        try:
            pipeline_instance.fa_model.align(audio=dummy_tuple, text="假", language="zh")
        except Exception:
            pass 
            
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
    
    valid_models = VALID_ALIGNMENT_MODELS + VALID_ASR_MODELS
    if model not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail={"error": {"message": f"Bad Request: Unsupported model '{model}'. Valid models are {valid_models}.", "type": "invalid_request_error"}}
        )

    is_alignment_mode = (model in VALID_ALIGNMENT_MODELS)

    if is_alignment_mode:
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400, 
                detail={"error": {"message": "Bad Request: The 'prompt' parameter is strictly required for ForcedAligner mode.", "type": "invalid_request_error"}}
            )
        if response_format != "verbose_json":
            raise HTTPException(
                status_code=400, 
                detail={"error": {"message": "Bad Request: Only 'verbose_json' response_format is supported in ForcedAligner mode.", "type": "invalid_request_error"}}
            )

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
                    async with gpu_state_lock:
                        active_gpu_requests += 1
                        
                    try:
                        if is_alignment_mode:
                            result = await run_in_threadpool(
                                pipeline_instance.process_audio, 
                                audio_path=tmp_path, 
                                reference_text=prompt,
                                language_code=language
                            )
                        else:
                            result = await pipeline_instance.process_audio_asr_only(
                                audio_path=tmp_path,
                                language_code=language
                            )
                            
                        if not is_alignment_mode and response_format == "json":
                            return {"text": result.get("text", "")}
                        return result
                        
                    finally:
                        async with gpu_state_lock:
                            active_gpu_requests -= 1
                            
            except Exception as e:
                is_oom = False
                try:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        is_oom = True
                except AttributeError:
                    pass
                if "out of memory" in str(e).lower():
                    is_oom = True

                if is_oom:
                    if attempt == max_retries - 1:
                        raise HTTPException(status_code=500, detail={"error": {"message": "GPU 资源耗尽，请求经历排队重试后依然失败。", "type": "server_error"}})
                    
                    print(f"  -> [拦截] OOM 发生！当前请求转入挂起重试状态 ({attempt+1}/{max_retries})...")
                    del e 
                    asyncio.create_task(trigger_frozen_world())
                    await asyncio.sleep(2) 
                else:
                    if isinstance(e, ValueError):
                        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
                    raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})
                    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 6008))
    
    print("="*50)
    print(f"🌐 Server 将在 http://{host}:{port} 启动")
    print(f"👉 API 接口端点: http://{host}:{port}/v1/audio/transcriptions")
    print("="*50)
    
    uvicorn.run(app, host=host, port=port, log_level="warning")