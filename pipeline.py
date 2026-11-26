import os
import re
import time
import torch
import asyncio
import tempfile
import Levenshtein
import soundfile as sf
import numpy as np

from fastapi.concurrency import run_in_threadpool
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner 
from utils import get_container_cpu_count

try:
    import zhconv
except ImportError:
    print("[ERROR] 未找到 zhconv 库，请运行: pip install zhconv")
    raise

# ==========================================
# 1. 核心 Pipeline
# ==========================================
class LongAudioAlignPipeline:
    def __init__(self, config):
        self.config = config
        self.device = os.environ.get("DEVICE", config.get("device", "cpu"))
        self.dtype = torch.float32
        
        total_cores = get_container_cpu_count()
        cpu_ratio = config.get("cpu_ratio", 0.5)
        allocated_cores = max(1, int(total_cores * cpu_ratio))
        torch.set_num_threads(allocated_cores)
        print(f"[*] 系统总核心数: {total_cores}, 已分配计算核心数: {allocated_cores}")

        vad_repo = os.environ.get("VAD_REPO", "snakers4/silero-vad")
        vad_model_name = os.environ.get("VAD_MODEL_NAME", "silero_vad")
        asr_path = os.environ.get("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
        fa_path = os.environ.get("FA_MODEL_PATH", "Qwen/Qwen3-ForcedAligner-0.6B")

        print(f"[*] 正在加载模型...\n    运行设备: {self.device}\n    VAD: {vad_model_name}\n    ASR: {asr_path}\n    FA:  {fa_path}")
        
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir=vad_repo, 
            model=vad_model_name,
            verbose=False
        )
        self.get_speech_timestamps = self.vad_utils[0] 

        # --- 🛠️ 动态构建模型加载参数，严格区分 CPU 和 GPU 模式 ---
        model_kwargs = {
            "dtype": self.dtype,
            "device_map": self.device,
        }
        
        # 判断标准：检查设备字符串中是否包含 'cuda'，或者设备不为 'cpu' 且当前 CUDA 确实可用
        is_gpu_mode = "cuda" in str(self.device).lower() or (self.device != "cpu" and torch.cuda.is_available())
        
        if is_gpu_mode:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("    [⚡] 确认运行于 GPU 模式，已自动激活动态加速：Flash Attention 2")
        else:
            print("    [ℹ️] 确认运行于 CPU 模式，已自动停用 Flash Attention 2 以防报错")
        # --------------------------------------------------------

        # 传入解包后的参数字典
        self.asr_model = Qwen3ASRModel.from_pretrained(
            asr_path,
            **model_kwargs
        )

        self.fa_model = Qwen3ForcedAligner.from_pretrained(
            fa_path,
            **model_kwargs
        )
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
                
            padded_chunks.append({
                'start': start - front_pad, 
                'end': end + back_pad
            })
            
        return padded_chunks

    def _macro_route_end_cursor(self, norm_asr, norm_ref, current_norm_cursor):
        if not norm_asr:
            return current_norm_cursor

        asr_len = len(norm_asr)
        lookahead_size = max(300, asr_len * 3) 
        search_window = norm_ref[current_norm_cursor : current_norm_cursor + lookahead_size]

        if not search_window:
            print(f"[WARNING] 原文已耗尽，但音频仍有多余识别: '{norm_asr}'。将安全忽略此段音频。")
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
            raise RuntimeError(
                f"[FATAL ERROR] 识别结果严重偏离原文或匹配脱轨！\n"
                f"相似度: {highest_ratio:.2f} (低于安全阈值 {MATCH_THRESHOLD})"
            )

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
        """对齐模式流水线：由于宏观寻路上下文强依赖，内部保持严格串行以确保对齐不紊乱"""
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
        raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
        
        if not raw_speech_timestamps:
            raise ValueError("VAD 未检测到任何有效人声，音频可能为空白或仅有底噪，停止处理。")

        physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration)
        total_chunks = len(physical_chunks)
        print(f"[*] 音频已根据 VAD 贪婪切分为 {total_chunks} 个带有静音缓冲的物理长片段。")

        verbose_json = {
            "task": "transcribe",
            "language": run_lang,
            "duration": round(total_audio_duration, 2),
            "report": {}, 
            "text": "", 
            "words": []
        }
        
        current_norm_cursor = 0 
        prev_orig_end = 0 
        
        # 用于精简版最终报告的时间累加器
        total_asr_time = 0.0
        total_fa_time = 0.0
        
        for idx, chunk_info in enumerate(physical_chunks):
            chunk_start = chunk_info['start']
            chunk_end = chunk_info['end']
            
            start_sample = max(0, int(chunk_start * sample_rate))
            end_sample = min(len(wav), int(chunk_end * sample_rate))
            chunk_wav = wav[start_sample:end_sample]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, chunk_wav.numpy(), sample_rate)
            
            try:
                asr_start = time.time()
                asr_results = self.asr_model.transcribe(audio=tmp_path, language=run_lang)
                asr_text = asr_results[0].text
                total_asr_time += (time.time() - asr_start)
                
                norm_asr = self._normalize_text(asr_text)
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

                fa_start = time.time()
                fa_results = self.fa_model.align(audio=tmp_path, text=ground_truth, language=run_lang)
                total_fa_time += (time.time() - fa_start)
                
                for word_align in fa_results[0]:
                    w = {
                        "word": word_align.text,
                        "start": round(chunk_start + word_align.start_time, 3),
                        "end": round(chunk_start + word_align.end_time, 3)
                    }
                    verbose_json["words"].append(w)
                    verbose_json["text"] += word_align.text
                    
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        verbose_json["report"] = self._generate_diff_report(clean_reference_text, verbose_json["text"])
        pipeline_end_time = time.time()
        total_pipeline_time = pipeline_end_time - pipeline_start_time
        rtf = total_pipeline_time / total_audio_duration
        
        # 统一输出整套请求的最终结果报告
        print(f"\n[*] ================== 统计报告 (对齐模式) ==================")
        print(f"    - 音频总时长         : {total_audio_duration:.2f} 秒")
        print(f"    - ASR 阶段总耗时     : {total_asr_time:.2f} 秒")
        print(f"    - 对齐 (FA) 阶段总耗时: {total_fa_time:.2f} 秒")
        print(f"    - 流水线整体总运行时间: {total_pipeline_time:.2f} 秒")
        print(f"    - RTF (实时率)       : {rtf:.3f}")
        print(f"    =========================================================\n")
        
        return verbose_json

    async def process_audio_asr_only(self, audio_path, language_code=None):
        """纯 ASR 模式流水线：支持增强级 VAD 分割限制（最大10分钟），内部分块全并发异步化，最终保序合并"""
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
        raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
        
        if not raw_speech_timestamps:
            raise ValueError("VAD 未检测到任何有效人声，音频可能为空白或仅有底噪，停止处理。")

        asr_max_sec = min(self.config.get("chunk_max_sec", 240), 600)
        physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration, max_sec=asr_max_sec)
        total_chunks = len(physical_chunks)
        print(f"[*] 音频已根据 ASR 增强规则切分为 {total_chunks} 个长片段（单片上限 {asr_max_sec} 秒 / {asr_max_sec / 60:.1f} 分钟）。")

        verbose_json = {
            "task": "transcribe",
            "language": run_lang,
            "duration": round(total_audio_duration, 2),
            "text": "", 
            "segments": []
        }

        # 内部并发任务：每个分块的任务高度隔离，确保文件、计算相互不污染
        async def process_single_chunk(idx, chunk_info):
            chunk_start = chunk_info['start']
            chunk_end = chunk_info['end']
            
            start_sample = max(0, int(chunk_start * sample_rate))
            end_sample = min(len(wav), int(chunk_end * sample_rate))
            chunk_wav = wav[start_sample:end_sample]
            
            def _sync_inference():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, chunk_wav.numpy(), sample_rate)
                try:
                    asr_results = self.asr_model.transcribe(audio=tmp_path, language=run_lang)
                    return asr_results[0].text
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            # 将同步推理委托至线程池，实现内部分块的无锁真并发
            asr_text = await run_in_threadpool(_sync_inference)
            return idx, chunk_start, chunk_end, asr_text

        # 启动全局并发调度
        asr_stage_start = time.time()
        tasks = [process_single_chunk(i, chunk) for i, chunk in enumerate(physical_chunks)]
        concurrent_results = await asyncio.gather(*tasks)
        asr_stage_duration = time.time() - asr_stage_start

        # 【保序重组】：按照原始的分块 idx 进行严格物理升序排序，确保文本与时间戳顺序绝对正确
        concurrent_results.sort(key=lambda x: x[0])

        for idx, chunk_start, chunk_end, asr_text in concurrent_results:
            if asr_text:
                verbose_json["text"] += asr_text
                verbose_json["segments"].append({
                    "id": idx,
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "text": asr_text
                })

        pipeline_end_time = time.time()
        total_pipeline_time = pipeline_end_time - pipeline_start_time
        rtf = total_pipeline_time / total_audio_duration
        
        print(f"\n[*] ================== 统计报告 (纯ASR模式) ==================")
        print(f"    - 音频总时长         : {total_audio_duration:.2f} 秒")
        print(f"    - ASR 阶段并发总耗时 : {asr_stage_duration:.2f} 秒")
        print(f"    - 流水线整体总运行时间: {total_pipeline_time:.2f} 秒")
        print(f"    - RTF (实时率)       : {rtf:.3f}")
        print(f"    =========================================================\n")
        
        return verbose_json