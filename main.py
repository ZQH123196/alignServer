import os
import json
import asyncio
import gc
import torch
import tempfile
import numpy as np
import soundfile as sf
import uvicorn
import gradio as gr
from dotenv import load_dotenv

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from utils import get_container_cpu_count, get_container_available_memory
from pipeline import LongAudioAlignPipeline

load_dotenv()

# ==========================================
# 2. 全局状态与探针配置 (新增 STW 控制器)
# ==========================================
CONFIG = {
    "device": "cpu",            
    "cpu_ratio": 0.8,             
    "chunk_max_sec": 60*4,      
    "vad_padding_sec": 0.3,     
    "language": "zh",
}

pipeline_instance = None
gpu_semaphore = None

# 定义支持的模型列表
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

# --- STW (Stop The World) 状态控制器 ---
world_frozen_event = asyncio.Event()
world_frozen_event.set() # 默认放行状态

active_gpu_requests = 0
gpu_state_lock = asyncio.Lock()
stw_lock = asyncio.Lock()

async def trigger_frozen_world():
    """触发 STW，执行物理清场"""
    global active_gpu_requests
    
    if stw_lock.locked():
        # 如果已经有其他并发请求触发了清场，直接跳过
        return
        
    async with stw_lock:
        # 1. 关门：阻止所有新请求获取 GPU 锁（使其在 wait() 处挂起排队）
        world_frozen_event.clear()
        print("\n[FATAL] 🛑 检测到碎片化 OOM，触发 Frozen World! 暂停进入计算区，后续请求进入排队队列...")
        
        # 2. 等待已经在处理中的其余任务安全退出
        while active_gpu_requests > 0:
            await asyncio.sleep(0.5)
            
        print("[FATAL] 🧹 核心计算区已清零，开始执行物理级显存清理...")
        
        # 3. 强制清零（回收显存及幽灵对象）
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("[FATAL] ✅ 清理完毕，解除 Frozen World，恢复队列处理。\n")
        
        # 4. 开门：唤醒所有正在排队的请求继续执行
        world_frozen_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, gpu_semaphore
    
    print("\n🚀 开始 Server 生命周期初始化...")
    pipeline_instance = LongAudioAlignPipeline(CONFIG)
    
    if pipeline_instance.device == "cpu" or not torch.cuda.is_available():
        cores = get_container_cpu_count()
        max_concurrency_by_cpu = max(1, cores // 2)
        
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
        
        dummy_audio_path = "temp_dummy_warmup.wav"
        sf.write(dummy_audio_path, np.random.randn(16000 * 5).astype(np.float32) * 0.01, 16000)
        
        try:
            pipeline_instance.asr_model.transcribe(audio=dummy_audio_path, language="zh")
        except Exception:
            pass
            
        try:
            pipeline_instance.fa_model.align(audio=dummy_audio_path, text="假", language="zh")
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
        
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)

    yield 
    print("🛑 Server 正在关闭，清理资源...")


# ==========================================
# 3. FastAPI 主网关与排队路由
# ==========================================
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
            # [STW 屏障]：如果世界被冻结，新来的和重试的请求都在这里安静排队
            await world_frozen_event.wait()
            
            try:
                async with gpu_semaphore:
                    # 登记进入计算区
                    async with gpu_state_lock:
                        active_gpu_requests += 1
                        
                    try:
                        if is_alignment_mode:
                            # 对齐路由：采用原有线程池安全串行分块
                            result = await run_in_threadpool(
                                pipeline_instance.process_audio, 
                                audio_path=tmp_path, 
                                reference_text=prompt,
                                language_code=language
                            )
                        else:
                            # 纯 ASR 路由：直接调用全新升级的异步内部分块并发函数
                            result = await pipeline_instance.process_audio_asr_only(
                                audio_path=tmp_path,
                                language_code=language
                            )
                            
                        # 执行成功，跳出重试循环
                        if not is_alignment_mode and response_format == "json":
                            return {"text": result.get("text", "")}
                        return result
                        
                    finally:
                        # 离开计算区
                        async with gpu_state_lock:
                            active_gpu_requests -= 1
                            
            except Exception as e:
                # 判断是否为 OOM 错误 (支持跨版本 PyTorch 判断)
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
                        # 如果穷尽重试仍旧 OOM，抛出异常。
                        raise HTTPException(status_code=500, detail={"error": {"message": "GPU 资源耗尽，请求经历排队重试后依然失败。", "type": "server_error"}})
                    
                    print(f"  -> [拦截] OOM 发生！当前请求转入挂起重试状态 ({attempt+1}/{max_retries})...")
                    # 触发 STW
                    asyncio.create_task(trigger_frozen_world())
                    # 强迫当前任务交出协程控制权，确保大门能关上
                    await asyncio.sleep(1) 
                else:
                    # 非内存类错误，属于正常业务失败，直接抛出
                    if isinstance(e, ValueError):
                        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
                    raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})
                    
    finally:
        # 文件必须在所有的排队与执行彻底结束后才能删
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================
# 4. Gradio 可视化 UI 挂载 (同步加入排队保护)
# ==========================================
async def gradio_bridge(audio_filepath, ref_text, selected_model):
    global active_gpu_requests
    
    if not audio_filepath:
        return "❌ 错误: 请先上传音频。"
    
    is_alignment = (selected_model in VALID_ALIGNMENT_MODELS)
    
    if is_alignment and (not ref_text or not ref_text.strip()):
        return "❌ 错误: ForcedAligner 对齐模式下，必须提供参考原文 (prompt)。"
    
    max_retries = 3
    for attempt in range(max_retries):
        await world_frozen_event.wait()
        try:
            async with gpu_semaphore:
                async with gpu_state_lock:
                    active_gpu_requests += 1
                try:
                    if is_alignment:
                        result = await run_in_threadpool(
                            pipeline_instance.process_audio, 
                            audio_path=audio_filepath, 
                            reference_text=ref_text,
                            language_code="zh" 
                        )
                    else:
                        result = await pipeline_instance.process_audio_asr_only(
                            audio_path=audio_filepath, 
                            language_code="zh" 
                        )
                    return json.dumps(result, ensure_ascii=False, indent=2)
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
                    return "🚨 发生错误: GPU OOM，多次重试失败。"
                asyncio.create_task(trigger_frozen_world())
                await asyncio.sleep(1)
            else:
                return f"🚨 发生错误: {str(e)}"

with gr.Blocks(title="Audio Processing UI") as demo:
    gr.Markdown("## 🎙️ Audio Processing Server (支持 ASR 与 强制对齐)")
    gr.Markdown(f"✅ **API 地址**: `http://<YOUR_IP>:<PORT>/v1/audio/transcriptions`  \n✅ **并发锁保护**: 已开启 | **防碎片 STW 队列**: 已开启")
    
    with gr.Row():
        with gr.Column(scale=1):
            ui_model = gr.Dropdown(
                choices=VALID_ALIGNMENT_MODELS + VALID_ASR_MODELS, 
                value="Qwen/Qwen3-ForcedAligner-0.6B", 
                label="0. 选择处理模式 (Model)"
            )
            ui_audio = gr.Audio(type="filepath", label="1. 上传测试音频")
            ui_text = gr.Textbox(lines=5, label="2. 参考原文 (纯 ASR 模式下此项会被忽略)", placeholder="对齐模式请在此粘贴带标点的完整参考文本...")
            ui_btn = gr.Button("开始处理", variant="primary")
        with gr.Column(scale=1):
            ui_output = gr.JSON(label="响应结果")

    ui_btn.click(fn=gradio_bridge, inputs=[ui_audio, ui_text, ui_model], outputs=ui_output)

app = gr.mount_gradio_app(app, demo, path="/ui")


# ==========================================
# 程序启动入口
# ==========================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8100))
    
    print("="*50)
    print(f"🌐 Server 将在 http://{host}:{port} 启动")
    print(f"👉 API 接口端点: http://{host}:{port}/v1/audio/transcriptions")
    print(f"👉 调试 UI 界面: http://{host}:{port}/ui")
    print("="*50)
    
    uvicorn.run(app, host=host, port=port, log_level="warning")