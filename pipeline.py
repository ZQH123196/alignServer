import os
import re
import time
import torch
import asyncio
import Levenshtein
import soundfile as sf
import numpy as np
import threading

from fastapi.concurrency import run_in_threadpool
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner 
from utils import get_container_cpu_count
from silero_vad import load_silero_vad, get_speech_timestamps 

try:
    import zhconv
except ImportError:
    print("[ERROR] 未找到 zhconv 库，请运行: pip install zhconv")
    raise

class LongAudioAlignPipeline:
    def __init__(self, config):
        self.config = config
        self.device = os.environ.get("DEVICE", config.get("device", "cpu"))
        
        # 线程锁：保护单进程内底层的 TorchScript / CUDA 上下文免于冲突崩溃
        self.infer_lock = threading.Lock()
        
        model_kwargs = {
            "device_map": self.device,
        }
        
        is_gpu_mode = "cuda" in str(self.device).lower() or (self.device != "cpu" and torch.cuda.is_available())
        
        if is_gpu_mode:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
                
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"    [⚡] 确认运行于 GPU 模式，已自动切换半精度类型: {self.dtype} 并激活：Flash Attention 2")
            except ImportError:
                model_kwargs["attn_implementation"] = "sdpa"
                print(f"    [💡] 提示：未检测到完全兼容的 Flash Attention 2，已自动切换至 PyTorch 原生加速：SDPA (性能几乎无损)")

            total_cores = get_container_cpu_count()
            cpu_ratio = config.get("cpu_ratio", 0.5)
            allocated_cores = max(1, int(total_cores * cpu_ratio))
            torch.set_num_threads(allocated_cores)
        else:
            self.dtype = torch.float32
            print("    [ℹ️] 确认运行于 CPU 模式，强制关闭算子多线程。")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1) 
            
        model_kwargs["torch_dtype"] = self.dtype

        asr_path = os.environ.get("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
        fa_path = os.environ.get("FA_MODEL_PATH", "Qwen/Qwen3-ForcedAligner-0.6B")

        print(f"[*] 正在加载模型...\n    运行设备: {self.device}\n    VAD: silero-vad (Pip 托管版)\n    ASR: {asr_path}\n    FA:  {fa_path}")
        
        self.vad_model = load_silero_vad() 
        self.get_speech_timestamps = get_speech_timestamps

        self.asr_model = Qwen3ASRModel.from_pretrained(asr_path, **model_kwargs)
        self.fa_model = Qwen3ForcedAligner.from_pretrained(fa_path, **model_kwargs)
        print("[*] 模型加载完成！")

    def _normalize_text(self, text):
        text_sim = zhconv.convert(text, 'zh-cn')
        text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text_sim)
        return text_clean

    def _build_index_map(self, original_text):
        norm_text = ""
        idx_map = []
        sim_text = zhconv.convert(original_text, 'zh-cn')
        for i, char in enumerate(sim_text):
            if re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]', char):
                norm_text += char
                idx_map.append(i)
        return norm_text, idx_map

    def _group_vad_timestamps(self, timestamps, total_audio_duration, max_sec=None):
        if not timestamps:
            return []

        if max_sec is None:
            max_sec = self.config["chunk_max_sec"]
        chunks = []
        current_chunk = []
        start_time = 0
        
        for ts in timestamps:
            if not current_chunk:
                current_chunk.append(ts)
                start_time = ts['start']
                continue
            
            potential_end_time = ts['end']
            duration = potential_end_time - start_time
            
            if duration > max_sec:
                chunks.append({'start': current_chunk[0]['start'], 'end': current_chunk[-1]['end']})
                current_chunk = [ts]
                start_time = ts['start']
            else:
                current_chunk.append(ts)
                
        if current_chunk:
            chunks.append({'start': current_chunk[0]['start'], 'end': current_chunk[-1]['end']})

        desired_pad = self.config.get("vad_padding_sec", 0.3)
        padded_chunks = []
        for i, chunk in enumerate(chunks):
            start = chunk['start']
            end = chunk['end']
            if i == 0:
                front_pad = min(start, desired_pad)
            else:
                gap = start - chunks[i-1]['end']
                front_pad = min(desired_pad, gap / 2.0)
            if i == len(chunks) - 1:
                back_pad = min(total_audio_duration - end, desired_pad)
            else:
                gap = chunks[i+1]['start'] - end
                back_pad = min(desired_pad, gap / 2.0)
            padded_chunks.append({'start': start - front_pad, 'end': end + back_pad})
        return padded_chunks

    def _macro_route_end_cursor(self, norm_asr, norm_ref, current_norm_cursor):
        if not norm_asr:
            return current_norm_cursor
        asr_len = len(norm_asr)
        lookahead_size = max(300, asr_len * 3) 
        search_window = norm_ref[current_norm_cursor : current_norm_cursor + lookahead_size]
        if not search_window:
            print(f"[WARNING] 原文已耗尽，但音频仍有多余识别。将安全忽略此段音频。")
            return current_norm_cursor

        highest_ratio = 0.0
        best_match_text = ""
        best_match_start = 0
        window_size = int(asr_len * 1.2)
        
        for i in range(0, max(1, len(search_window) - window_size // 2)):
            sub_text = search_window[i : i + window_size]
            ratio = Levenshtein.ratio(norm_asr, sub_text)
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match_text = sub_text
                best_match_start = i

        MATCH_THRESHOLD = 0.30
        if highest_ratio < MATCH_THRESHOLD:
            raise RuntimeError(f"[FATAL ERROR] 识别结果严重偏离原文！相似度: {highest_ratio:.2f}")

        opcodes = Levenshtein.opcodes(norm_asr, best_match_text)
        exact_match_len = len(best_match_text)
        for tag, i1, i2, j1, j2 in reversed(opcodes):
            if tag == 'insert':
                exact_match_len = j1
            else:
                break
        best_match_end_idx = best_match_start + exact_match_len
        return current_norm_cursor + best_match_end_idx

    def _generate_diff_report(self, clean_reference, final_result):
        norm_ref = self._normalize_text(clean_reference)
        norm_result = self._normalize_text(final_result)
        opcodes = Levenshtein.opcodes(norm_ref, norm_result)
        missing_blocks = []
        extra_blocks = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag in ('delete', 'replace'):
                missing_blocks.append(norm_ref[i1:i2])
            if tag in ('insert', 'replace'):
                extra_blocks.append(norm_result[j1:j2])
        ratio = Levenshtein.ratio(norm_ref, norm_result)
        return {
            "accuracy_ratio": f"{ratio * 100:.2f}%",
            "ref_length": len(norm_ref),
            "result_length": len(norm_result),
            "missing_from_result": missing_blocks,  
            "extra_in_result": extra_blocks         
        }

    def process_audio(self, audio_path, reference_text, language_code=None):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        clean_reference_text = re.sub(r'\s+', ' ', reference_text).strip()
        norm_ref, ref_idx_map = self._build_index_map(clean_reference_text)
        
        print(f"[*] 正在读取音频文件 (对齐模式)...")
        audio_data, sample_rate = sf.read(audio_path)
        wav = torch.from_numpy(audio_data).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sample_rate != 16000:
            import torchaudio.functional as F
            wav = F.resample(wav, sample_rate, 16000)
            sample_rate = 16000

        total_audio_duration = len(wav) / sample_rate
        print("[*] 正在进行 VAD 物理切片...")
        with self.infer_lock:
            raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
        if not raw_speech_timestamps:
            raise ValueError("VAD 未检测到任何有效人声。")

        physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration)
        print(f"[*] 音频已切分为 {len(physical_chunks)} 个物理长片段。")

        # 🚀【核心重构 1】：一次性构建所有切片的 Batch 内存数据，消灭 ASR 循环
        batch_audios = []
        for chunk_info in physical_chunks:
            start_sample = max(0, int(chunk_info['start'] * sample_rate))
            end_sample = min(len(wav), int(chunk_info['end'] * sample_rate))
            chunk_wav = wav[start_sample:end_sample]
            batch_audios.append((chunk_wav.numpy(), sample_rate))

        print(f"    [🚀] 启动对齐模式 - 原生 ASR Batch 推理 (共 {len(batch_audios)} 个切片)...")
        with self.infer_lock:
            asr_results = self.asr_model.transcribe(audio=batch_audios, language=run_lang)

        # 🚀【核心重构 2】：在 CPU 纯内存中快速批量处理路由匹配，并组装 FA 推理对
        fa_batch_audios = []
        fa_batch_texts = []
        chunk_mapping_meta = []  # 用于保留有效切片的原始信息

        current_norm_cursor = 0 
        prev_orig_end = 0 
        
        for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
            norm_asr = self._normalize_text(res.text)
            if not norm_asr:
                continue
                
            abs_end = self._macro_route_end_cursor(norm_asr, norm_ref, current_norm_cursor)
            orig_start = prev_orig_end
            if abs_end < len(norm_ref):
                orig_end = ref_idx_map[abs_end]
            else:
                orig_end = len(clean_reference_text)
                
            ground_truth = clean_reference_text[orig_start : orig_end]
            current_norm_cursor = abs_end
            prev_orig_end = orig_end
            if not ground_truth.strip():
                continue

            # 加入真正的 FA 推理批次
            fa_batch_audios.append(batch_audios[idx])
            fa_batch_texts.append(ground_truth)
            chunk_mapping_meta.append(chunk_info)

        verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "report": {}, "text": "", "words": []}

        # 🚀【核心重构 3】：如果存在有效对齐文本，一次性发起 FA Batch 计算，消灭 FA 循环
        if fa_batch_audios:
            print(f"    [⚡] 启动对齐模式 - 原生 Forced Aligner Batch 推理 (共 {len(fa_batch_audios)} 个切片)...")
            with self.infer_lock:
                fa_results = self.fa_model.align(audio=fa_batch_audios, text=fa_batch_texts, language=run_lang)

            # 组装最终时间戳报告
            for chunk_info, single_fa_res in zip(chunk_mapping_meta, fa_results):
                chunk_start = chunk_info['start']
                for word_align in single_fa_res:
                    verbose_json["words"].append({
                        "word": word_align.text,
                        "start": round(chunk_start + word_align.start_time, 3),
                        "end": round(chunk_start + word_align.end_time, 3)
                    })
                    verbose_json["text"] += word_align.text

        verbose_json["report"] = self._generate_diff_report(clean_reference_text, verbose_json["text"])
        total_pipeline_time = time.time() - pipeline_start_time
        print(f"\n[*] ================== 统计报告 (对齐模式) ==================")
        print(f"    - 音频总时长         : {total_audio_duration:.2f} 秒")
        print(f"    - 流水线整体总运行时间: {total_pipeline_time:.2f} 秒")
        print(f"    - RTF (实时率)       : {total_pipeline_time / total_audio_duration:.3f}")
        return verbose_json

    async def process_audio_asr_only(self, audio_path, language_code=None):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        
        print(f"[*] 正在读取音频文件 (纯ASR模式)...")
        audio_data, sample_rate = sf.read(audio_path)
        wav = torch.from_numpy(audio_data).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sample_rate != 16000:
            import torchaudio.functional as F
            wav = F.resample(wav, sample_rate, 16000)
            sample_rate = 16000

        total_audio_duration = len(wav) / sample_rate
        print("[*] 正在进行 VAD 物理切片...")
        with self.infer_lock:
            raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
            
        if not raw_speech_timestamps:
            raise ValueError("VAD 未检测到任何有效人声。")

        asr_max_sec = self.config.get("chunk_max_sec", 60)
        physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration, max_sec=asr_max_sec)
        print(f"[*] 音频已切分为 {len(physical_chunks)} 个片段。")

        verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "text": "", "segments": []}

        batch_audios = []
        for chunk_info in physical_chunks:
            start_sample = max(0, int(chunk_info['start'] * sample_rate))
            end_sample = min(len(wav), int(chunk_info['end'] * sample_rate))
            chunk_wav = wav[start_sample:end_sample]
            batch_audios.append((chunk_wav.numpy(), sample_rate))

        print(f"    [🚀] 启动高并发原生 Batch 推理 (共 {len(batch_audios)} 个切片)...")
        asr_stage_start = time.time()

        def _sync_batch_infer():
            with self.infer_lock:
                return self.asr_model.transcribe(audio=batch_audios, language=run_lang)

        asr_results = await run_in_threadpool(_sync_batch_infer)
        asr_stage_duration = time.time() - asr_stage_start

        for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
            asr_text = res.text
            if asr_text:
                verbose_json["text"] += asr_text
                verbose_json["segments"].append({
                    "id": idx, "start": round(chunk_info['start'], 3), "end": round(chunk_info['end'], 3), "text": asr_text
                })

        total_pipeline_time = time.time() - pipeline_start_time
        print(f"\n[*] ================== 统计报告 (纯ASR模式) ==================")
        print(f"    - 音频总时长         : {total_audio_duration:.2f} 秒")
        print(f"    - ASR 阶段内部耗时   : {asr_stage_duration:.2f} 秒")
        print(f"    - 流水线整体总运行时间: {total_pipeline_time:.2f} 秒")
        print(f"    - RTF (实时率)       : {total_pipeline_time / total_audio_duration:.3f}")
        return verbose_json