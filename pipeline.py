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

# [修改点 1]：定义专属异常类
class AlignmentThresholdError(Exception):
    pass

class LongAudioAlignPipeline:
    def __init__(self, config):
        self.config = config
        self.device = os.environ.get("DEVICE", config.get("device", "cpu"))
        
        self.infer_lock = threading.Lock()
        model_kwargs = {"device_map": self.device}
        
        is_gpu_mode = "cuda" in str(self.device).lower() or (self.device != "cpu" and torch.cuda.is_available())
        
        if is_gpu_mode:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): self.dtype = torch.bfloat16
            else: self.dtype = torch.float16
                
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
        if not timestamps: return []
        if max_sec is None: max_sec = self.config["chunk_max_sec"]
        
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
            if i == 0: front_pad = min(start, desired_pad)
            else: gap = start - chunks[i-1]['end']; front_pad = min(desired_pad, gap / 2.0)
            
            if i == len(chunks) - 1: back_pad = min(total_audio_duration - end, desired_pad)
            else: gap = chunks[i+1]['start'] - end; back_pad = min(desired_pad, gap / 2.0)
                
            padded_chunks.append({'start': start - front_pad, 'end': end + back_pad})
        return padded_chunks

    # [修改点 2]：将切片序号和切片信息注入进来，方便拼装诊断报告
    def _macro_route_end_cursor(self, norm_asr, norm_ref, current_norm_cursor, chunk_idx, chunk_info):
        if not norm_asr:
            return current_norm_cursor
        asr_len = len(norm_asr)
        lookahead_size = max(300, asr_len * 3) 
        search_window = norm_ref[current_norm_cursor : current_norm_cursor + lookahead_size]
        if not search_window:
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
            # [修改点 3]：组装详尽的诊断信息，包含切片位置、音频时间以及原文预览（防刷屏，截断显示前后文）
            window_preview = search_window[:80] + ("..." if len(search_window) > 80 else "")
            asr_preview = norm_asr[:80] + ("..." if len(norm_asr) > 80 else "")
            
            err_msg = (
                f"    -> 异常片段: 第 {chunk_idx + 1} 个物理切片 (音频时间: {chunk_info['start']:.2f}秒 - {chunk_info['end']:.2f}秒)\n"
                f"    -> ASR 识别结果: '{asr_preview}'\n"
                f"    -> 系统比对原文: '{window_preview}'\n"
                f"    -> 最高相似度: {highest_ratio:.2f} (及格线: {MATCH_THRESHOLD})"
            )
            # 抛出专属的异常供 main.py 精准捕捉
            raise AlignmentThresholdError(err_msg)

        opcodes = Levenshtein.opcodes(norm_asr, best_match_text)
        exact_match_len = len(best_match_text)
        for tag, i1, i2, j1, j2 in reversed(opcodes):
            if tag == 'insert': exact_match_len = j1
            else: break
                
        best_match_end_idx = best_match_start + exact_match_len
        return current_norm_cursor + best_match_end_idx

    def _generate_diff_report(self, clean_reference, final_result):
        norm_ref = self._normalize_text(clean_reference)
        norm_result = self._normalize_text(final_result)
        opcodes = Levenshtein.opcodes(norm_ref, norm_result)
        missing_blocks = []
        extra_blocks = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag in ('delete', 'replace'): missing_blocks.append(norm_ref[i1:i2])
            if tag in ('insert', 'replace'): extra_blocks.append(norm_result[j1:j2])
        ratio = Levenshtein.ratio(norm_ref, norm_result)
        return {
            "accuracy_ratio": f"{ratio * 100:.2f}%",
            "ref_length": len(norm_ref),
            "result_length": len(norm_result),
            "missing_from_result": missing_blocks,  
            "extra_in_result": extra_blocks         
        }

    def process_audio(self, audio_path, reference_text, language_code=None, req_id="Unknown"):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        
        try:
            clean_reference_text = re.sub(r'\s+', ' ', reference_text).strip()
            norm_ref, ref_idx_map = self._build_index_map(clean_reference_text)
            
            try: audio_data, sample_rate = sf.read(audio_path)
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: 音频读取失败 | 详情: {err}")
                raise
                
            wav = torch.from_numpy(audio_data).float()
            if wav.dim() > 1: wav = wav.mean(dim=1)
            if sample_rate != 16000:
                import torchaudio.functional as F
                wav = F.resample(wav, sample_rate, 16000)
                sample_rate = 16000

            total_audio_duration = len(wav) / sample_rate
            
            try:
                with self.infer_lock:
                    raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
                if not raw_speech_timestamps:
                    raise ValueError("VAD 未检测到任何有效人声。")
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: VAD 物理切片失败 | 详情: {err}")
                raise

            physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration)

            batch_audios = []
            for chunk_info in physical_chunks:
                start_sample = max(0, int(chunk_info['start'] * sample_rate))
                end_sample = min(len(wav), int(chunk_info['end'] * sample_rate))
                chunk_wav = wav[start_sample:end_sample]
                batch_audios.append((chunk_wav.numpy(), sample_rate))

            try:
                with self.infer_lock:
                    asr_results = self.asr_model.transcribe(audio=batch_audios, language=run_lang)
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: ASR 批量模型推理崩溃 | 详情: {err}")
                raise

            fa_batch_audios = []
            fa_batch_texts = []
            chunk_mapping_meta = []  

            current_norm_cursor = 0 
            prev_orig_end = 0 
            
            try:
                for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
                    norm_asr = self._normalize_text(res.text)
                    if not norm_asr:
                        continue
                        
                    # [修改点 4]：调用时把 idx 和 chunk_info 传进去
                    abs_end = self._macro_route_end_cursor(norm_asr, norm_ref, current_norm_cursor, idx, chunk_info)
                    
                    orig_start = prev_orig_end
                    if abs_end < len(norm_ref): orig_end = ref_idx_map[abs_end]
                    else: orig_end = len(clean_reference_text)
                        
                    ground_truth = clean_reference_text[orig_start : orig_end]
                    current_norm_cursor = abs_end
                    prev_orig_end = orig_end
                    if not ground_truth.strip():
                        continue

                    fa_batch_audios.append(batch_audios[idx])
                    fa_batch_texts.append(ground_truth)
                    chunk_mapping_meta.append(chunk_info)
            except AlignmentThresholdError as threshold_err:
                # 原封不动抛出，不套别的壳子，让 main.py 直接截获
                raise threshold_err
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: 文本对齐路由匹配失败 | 详情: {err}")
                raise

            verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "report": {}, "text": "", "words": []}

            if fa_batch_audios:
                try:
                    with self.infer_lock:
                        fa_results = self.fa_model.align(audio=fa_batch_audios, text=fa_batch_texts, language=run_lang)
                except Exception as err:
                    print(f"[🔴 错误] ID: {req_id} | 阶段: FA ForcedAligner 批量推理对齐失败 | 详情: {err}")
                    raise

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
            
            accuracy_str = verbose_json["report"].get("accuracy_ratio", "0.00%")
            accuracy_val = float(accuracy_str.strip('%'))
            align_status = "✅ 成功" if accuracy_val > 50 else "⚠️ 相似度低"
            ref_len = verbose_json["report"].get("ref_length", 0)
            res_len = verbose_json["report"].get("result_length", 0)

            print(f"\n[📊] =============== 最终报告 (对齐模式) ===============")
            print(f"    - 请求标识       : {req_id}")
            print(f"    - 执行状态       : {align_status} (文本相似度: {accuracy_str})")
            print(f"    - 物理切片数量   : 产生 {len(physical_chunks)} 段 -> 有效对齐 {len(fa_batch_audios)} 段")
            print(f"    - 字符匹配统计   : 原文共 {ref_len} 字，最终成功对齐 {res_len} 字")
            print(f"    - 音频总时长     : {total_audio_duration:.2f} 秒")
            print(f"    - 流水线总耗时   : {total_pipeline_time:.2f} 秒")
            print(f"    - RTF (实时率)   : {total_pipeline_time / total_audio_duration:.3f}")
            print(f"[=======================================================]\n")
            
            return verbose_json

        except Exception as outer_err:
            raise outer_err

    async def process_audio_asr_only(self, audio_path, language_code=None, req_id="Unknown"):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        
        try:
            try: audio_data, sample_rate = sf.read(audio_path)
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: 音频读取失败 | 详情: {err}")
                raise
                
            wav = torch.from_numpy(audio_data).float()
            if wav.dim() > 1: wav = wav.mean(dim=1)
            if sample_rate != 16000:
                import torchaudio.functional as F
                wav = F.resample(wav, sample_rate, 16000)
                sample_rate = 16000

            total_audio_duration = len(wav) / sample_rate
            
            try:
                with self.infer_lock:
                    raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
                if not raw_speech_timestamps:
                    raise ValueError("VAD 未检测到任何有效人声。")
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: VAD 物理切片失败 | 详情: {err}")
                raise

            asr_max_sec = self.config.get("chunk_max_sec", 60)
            physical_chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration, max_sec=asr_max_sec)

            verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "text": "", "segments": []}

            batch_audios = []
            for chunk_info in physical_chunks:
                start_sample = max(0, int(chunk_info['start'] * sample_rate))
                end_sample = min(len(wav), int(chunk_info['end'] * sample_rate))
                chunk_wav = wav[start_sample:end_sample]
                batch_audios.append((chunk_wav.numpy(), sample_rate))

            asr_stage_start = time.time()
            def _sync_batch_infer():
                with self.infer_lock: return self.asr_model.transcribe(audio=batch_audios, language=run_lang)

            try: asr_results = await run_in_threadpool(_sync_batch_infer)
            except Exception as err:
                print(f"[🔴 错误] ID: {req_id} | 阶段: 纯ASR模型批量推理崩溃 | 详情: {err}")
                raise
                
            asr_stage_duration = time.time() - asr_stage_start

            for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
                asr_text = res.text
                if asr_text:
                    verbose_json["text"] += asr_text
                    verbose_json["segments"].append({
                        "id": idx, "start": round(chunk_info['start'], 3), "end": round(chunk_info['end'], 3), "text": asr_text
                    })

            total_pipeline_time = time.time() - pipeline_start_time
            
            seg_count = len(verbose_json["segments"])
            char_count = len(verbose_json["text"])
            
            print(f"\n[📊] =============== 最终报告 (纯ASR模式) ===============")
            print(f"    - 请求标识       : {req_id}")
            print(f"    - 执行状态       : ✅ 成功完成")
            print(f"    - 切片与分段     : VAD物理切片 {len(physical_chunks)} 段 -> 生成 {seg_count} 个识别段落")
            print(f"    - 识别字符统计   : 共计生成 {char_count} 个字符")
            print(f"    - 音频总时长     : {total_audio_duration:.2f} 秒")
            print(f"    - ASR 推理耗时   : {asr_stage_duration:.2f} 秒")
            print(f"    - 流水线总耗时   : {total_pipeline_time:.2f} 秒")
            print(f"    - RTF (实时率)   : {total_pipeline_time / total_audio_duration:.3f}")
            print(f"[=======================================================]\n")
            
            return verbose_json
            
        except Exception as outer_err:
            raise outer_err