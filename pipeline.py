import os
import re
import time
import torch
import Levenshtein
import soundfile as sf
import numpy as np
import threading

from fastapi.concurrency import run_in_threadpool
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from utils import get_container_cpu_count
from silero_vad import load_silero_vad, get_speech_timestamps

# 共享 main.py 的并发可视化日志（log/register/mark_*）—— 任务执行过程打印时
# 会自动附上时间戳、级别、req_id 与"当前任务队列快照"，方便实时看到并行任务与各自处理的文件。
#
# [修复] main.py 作为入口脚本运行时，Python 把它注册为 sys.modules['__main__']
# 而不是 'main'。下面这一行 `from main import ...` 会让 Python 找不到 'main'
# 而把 main.py 当成独立模块重新执行，从而触发循环导入。
# 这里在导入之前把当前 __main__ 同步到 'main'，让 Python 使用部分加载的模块。
import sys as _sys
if 'main' not in _sys.modules and '__main__' in _sys.modules:
    _sys.modules['main'] = _sys.modules['__main__']
from main import (
    log, log_info, log_warn, log_error,
    register, mark_running, mark_done, mark_failed, unregister, reset_for_retry,
) 

try:
    import zhconv
except ImportError:
    print("[ERROR] 未找到 zhconv 库，请运行: pip install zhconv")
    raise

# 动态载入 Epitran
try:
    import epitran
except ImportError:
    print("[ERROR] 未找到 epitran 库，请运行: pip install epitran")
    raise

class AlignmentThresholdError(Exception):
    pass

class LongAudioAlignPipeline:
    def _load_audio(self, audio_path, req_id="Unknown"):
        """
        健壮音频解码：ffmpeg 子进程解码为 16kHz mono PCM，再读 numpy。

        libsndfile (soundfile) 对 opus 文件返回垃圾 frames（如 -1），导致
        np.empty(shape) 字节数溢出报 'array is too big'，autodl 上对所有 opus 必现。
        torchaudio 的 sox 后端虽能读但解码出的音频内容损坏（ASR 识别为空，相似度 0）。
        ffmpeg 子进程直接解码到 PCM raw，绕开所有 Python 音频库的后端问题，最可靠。

        Returns: (wav: torch.Tensor[N] float32 @16000, sample_rate=16000)
        """
        import subprocess, tempfile as _tempfile
        try:
            # ffmpeg 解码为 16kHz mono f32le PCM raw，输出到临时文件
            tmp_pcm = _tempfile.NamedTemporaryFile(delete=False, suffix='.pcm')
            tmp_pcm.close()
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', audio_path,
                     '-ac', '1', '-ar', '16000', '-f', 'f32le', tmp_pcm.name],
                    check=True, capture_output=True, timeout=600,
                )
                audio_data = np.fromfile(tmp_pcm.name, dtype=np.float32)
            finally:
                try: os.unlink(tmp_pcm.name)
                except OSError: pass
            if audio_data.size == 0:
                raise ValueError("ffmpeg 解码输出为空")
            wav = torch.from_numpy(audio_data).float()
            sample_rate = 16000
        except Exception as ff_err:
            log_error(f"阶段: 音频读取失败 | ffmpeg: {ff_err}", req_id=req_id)
            raise

        return wav, sample_rate

    def __init__(self, config):
        self.config = config
        self.device = os.environ.get("DEVICE", config.get("device", "cpu"))
        
        self.infer_lock = threading.Lock()
        model_kwargs = {"device_map": self.device}
        
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
        
        # 缓存已实例化的 Epitran 转换器以复用
        self._epitran_cache = {}
        self._cache_lock = threading.Lock() # [新增] 专门用于保护字典的并发锁
        print("[*] 模型及核心算子加载完成！")

    def _get_epitran(self, lang):
        """
        跨语言 Epitran 适配转换器
        
        支持两阶段解析：
        1. 别名转换阶段 (Alias Stage): 将 "Chinese", "English" 等常用名称安全映射到标准码。
        2. 标准码解析阶段 (Standard Stage): 匹配 RFC 5646 / ISO 639-1 标准。
        
        不支持的语言将直接 Fail-Fast 抛出异常。
        """
        # 统一归一化：去空格并转小写
        lang_input = (lang or "zh").lower().strip()

        # =================【数据定义：分层隔离】=================
        
        # 1. 常用自然语言/别名 到 标准代码的映射 (Alias Map)
        ALIAS_MAP = {
            "chinese": "zh",
            "english": "en",
            "japanese": "ja",
            "korean": "ko",
            "french": "fr",
            "german": "de",
            "russian": "ru",
        }

        # 2. RFC 5646 标准代码 到 Epitran 内部代码的映射 (Standard Map)
        RFC_5646_MAP = {
            # 中文语系变体
            "zh": "cmn-Hans",
            "zh-cn": "cmn-Hans",
            "zh-hans": "cmn-Hans",
            "zh-sg": "cmn-Hans",
            "cmn-hans": "cmn-Hans",
            "zh-tw": "cmn-Hant",
            "zh-hk": "cmn-Hant",
            "zh-hant": "cmn-Hant",
            "cmn-hant": "cmn-Hant",
            
            # 英文语系
            "en": "eng-Latn",
            "en-us": "eng-Latn",
            "en-gb": "eng-Latn",
            "eng": "eng-Latn",
            "eng-latn": "eng-Latn",
            
            # 日语
            "ja": "jpn-Latn",
            "ja-jp": "jpn-Latn",
            "jpn": "jpn-Latn",
            
            # 韩文
            "ko": "kor-Hang",
            "ko-kr": "kor-Hang",
            "kor": "kor-Hang",
            
            # 法文
            "fr": "fra-Latn",
            "fr-fr": "fra-Latn",
            "fra": "fra-Latn",
            
            # 德文
            "de": "deu-Latn",
            "de-de": "deu-Latn",
            "deu": "deu-Latn",
            
            # 俄文
            "ru": "rus-Cyrl",
            "ru-ru": "rus-Cyrl",
            "rus": "rus-Cyrl",
        }

        # =================【业务逻辑：两阶段解析】=================

        # 第一阶段：尝试将“别名/全称”转换为“标准代码”
        standard_key = ALIAS_MAP.get(lang_input, lang_input)

        # 第二阶段：从标准映射表中获取 Epitran 对应的具体配置
        if standard_key not in RFC_5646_MAP:
            raise ValueError(
                f"Unsupported language input '{lang}' (parsed as '{standard_key}').\n"
                f"Please use standard RFC 5646 codes (e.g., 'en-US', 'zh-CN') "
                f"or supported names (e.g., 'English', 'Chinese')."
            )
            
        epitran_code = RFC_5646_MAP[standard_key]

        # =================【实例化与缓存】=================

        # 第一层检查：如果已经缓存，直接返回（无锁，保证绝大多数情况下的极速访问）
        if epitran_code in self._epitran_cache:
            return self._epitran_cache[epitran_code]
            
        # 如果没缓存，加锁让多线程排队排查
        with self._cache_lock:
            # 第二层检查：拿到锁后，再查一次。防止在排队等待期间，别的线程已经初始化好了
            if epitran_code in self._epitran_cache:
                return self._epitran_cache[epitran_code]
                
            try:
                print(f"    [🌐] 正在初始化 Epitran 转换器 (Input: '{lang}' -> Target: '{epitran_code}')...")
                epi = epitran.Epitran(epitran_code)
                self._epitran_cache[epitran_code] = epi
                return epi
            except Exception as e:
                print(f"    [🔴] 错误：Epitran 无法载入代码 {epitran_code}。异常: {e}")
                raise ValueError(f"Failed to initialize Epitran for code '{epitran_code}': {e}")

    def _normalize_text(self, text, lang="zh"):
        """跨语言字符级基础过滤归一化"""
        lang = (lang or "zh").lower().strip()
        is_chinese = any(kw in lang for kw in ["zh", "chinese", "cmn", "hans", "hant"])
        
        if is_chinese:
            text_sim = zhconv.convert(text, 'zh-cn')
            text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text_sim)
        else:
            # 西方/非中文语系保留空格，移除绝大多数标点，归一化小写
            text_clean = text.lower()
            text_clean = re.sub(r'[^a-z0-9\s]', '', text_clean)
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        return text_clean

    def _build_index_map(self, original_text, lang="zh"):
        """
        三位一体的索引映射构建器（新增 IPA 编译层）
        建立 `IPA 下标 -> norm 字符下标 -> 原始字符下标` 复合映射链
        """
        norm_text = ""
        idx_map = [] # 记录 norm_text 的每个字在 original_text 中的索引
        
        lang = (lang or "zh").lower().strip()
        is_chinese = any(kw in lang for kw in ["zh", "chinese", "cmn", "hans", "hant"])
        
        # 1. 物理对齐 norm_text 与 original_text 的索引
        if is_chinese:
            sim_text = zhconv.convert(original_text, 'zh-cn')
            for i, char in enumerate(sim_text):
                if re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]', char):
                    norm_text += char
                    idx_map.append(i)
        else:
            orig_lower = original_text.lower()
            for i, char in enumerate(orig_lower):
                if re.match(r'[a-z0-9\s]', char):
                    norm_text += char
                    idx_map.append(i)

        # 2. 调用 Epitran 对 norm_text 全文编译为 IPA (Fail-Fast 会在这里被触发)
        epi = self._get_epitran(lang)
        
        ipa_text = ""
        ipa_to_norm_map = [] # 用于从 ipa_text 下标查回 norm_text 对应的字符下标
        
        if epi is None:
            # 如果 Epitran 加载失败，无缝降级回纯文本
            return norm_text, norm_text, list(range(len(norm_text))), idx_map

# Epitran 以词为单位进行音素编译效率最高
        words = norm_text.split(" ") if not is_chinese else list(norm_text)
        
        current_search_offset = 0  # [新增] 维护一个全局向前推进的搜索游标
        
        for word_idx, word in enumerate(words):
            # 将单词编译为 IPA 音素
            word_ipa = epi.transliterate(word)
            # 记录当前单词在原 norm_text 中的字符索引区间
            if not is_chinese:
                # [修改] 强制从 current_search_offset 之后开始寻找，防止高频词导致时光倒流
                norm_word_start = norm_text.find(word, current_search_offset) 
                if norm_word_start == -1: 
                    norm_word_start = current_search_offset
                
                # [新增] 更新游标，跳过当前单词长度
                current_search_offset = norm_word_start + len(word)
            else:
                norm_word_start = word_idx
            
            # 为该单词生成的每一个 IPA 字符映射回它原属的 norm 字符起点
            for _ in range(len(word_ipa)):
                ipa_text += word_ipa[_]
                ipa_to_norm_map.append(norm_word_start)
                
            # 西方文字在 IPA 中加空格作为边界
            if not is_chinese and word_idx < len(words) - 1:
                ipa_text += " "
                # 空格对应原 norm 中的空格
                ipa_to_norm_map.append(norm_word_start + len(word))

        return ipa_text, norm_text, ipa_to_norm_map, idx_map

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

    def _global_teleport_anchor(self, ipa_asr, ipa_ref, current_ipa_cursor):
        """
        【绝对寻址】跨语言 IPA N-gram 全局自愈器
        """
        if len(ipa_asr) < 8:
            return None
            
        anchor_len = min(20, len(ipa_asr) // 2)
        start_idx = len(ipa_asr) // 4
        ipa_anchor = ipa_asr[start_idx : start_idx + anchor_len]
        
        search_range = ipa_ref[current_ipa_cursor:]
        if not search_range or len(search_range) < len(ipa_anchor):
            return None
            
        best_ratio = 0.0
        best_offset = -1
        window_size = int(len(ipa_anchor) * 1.3)
        
        for i in range(0, len(search_range) - window_size):
            sub_ipa = search_range[i : i + window_size]
            ratio = Levenshtein.ratio(ipa_anchor, sub_ipa)
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = i
                
        GLOBAL_CONFIDENCE_THRESHOLD = 0.85
        if best_ratio >= GLOBAL_CONFIDENCE_THRESHOLD:
            new_ipa_cursor = current_ipa_cursor + best_offset + len(ipa_anchor)
            return new_ipa_cursor, best_ratio
            
        return None

    def _macro_route_end_cursor(self, ipa_asr, ipa_ref, current_ipa_cursor, chunk_idx, chunk_info, retry_level=0):
        """
        【IPA + 相对寻址机制】
        """
        if not ipa_asr:
            return current_ipa_cursor, 1.0, None
            
        asr_len = len(ipa_asr)
        
        if retry_level == 0:
            search_start = current_ipa_cursor
            lookahead_size = max(250, asr_len * 3) 
            MATCH_THRESHOLD = 0.35
        elif retry_level == 1:
            search_start = current_ipa_cursor
            lookahead_size = max(800, asr_len * 10) 
            MATCH_THRESHOLD = 0.20
        elif retry_level == 2:
            search_start = max(0, current_ipa_cursor - 60)
            lookahead_size = max(500, asr_len * 5)
            MATCH_THRESHOLD = 0.20
        else:
            teleport_res = self._global_teleport_anchor(ipa_asr, ipa_ref, current_ipa_cursor)
            if teleport_res is not None:
                new_cursor, score = teleport_res
                log_warn(f"🌟 全局自愈成功 | 物理切片 {chunk_idx + 1} 定位崩溃！已抓取核心 IPA 绝对锚点，游标由 {current_ipa_cursor} 瞬移至 -> {new_cursor} (锚点匹配度: {score:.2f})", req_id=req_id)
                return new_cursor, score, None
            return None, 0.0, None 

        search_window = ipa_ref[search_start : search_start + lookahead_size]
        if not search_window:
            if retry_level == 0: return current_ipa_cursor, 0.0, None
            return None, 0.0, None

        highest_ratio = 0.0
        best_match_start = 0
        window_size = int(asr_len * 1.2)
        
        for i in range(0, max(1, len(search_window) - window_size // 2)):
            sub_text = search_window[i : i + window_size]
            ratio = Levenshtein.ratio(ipa_asr, sub_text)
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match_start = i

        if highest_ratio < MATCH_THRESHOLD:
            return self._macro_route_end_cursor(ipa_asr, ipa_ref, current_ipa_cursor, chunk_idx, chunk_info, retry_level + 1)

        sub_match = search_window[best_match_start : best_match_start + window_size]
        opcodes = Levenshtein.opcodes(ipa_asr, sub_match)
        exact_match_len = len(sub_match)
        for tag, i1, i2, j1, j2 in reversed(opcodes):
            if tag == 'insert': exact_match_len = j1
            else: break
                
        best_match_end_idx = best_match_start + exact_match_len
        return search_start + best_match_end_idx, highest_ratio, None

    def _evaluate_fa_health(self, word_alignments):
        if not word_alignments: return 0.0
        squashed_words = 0
        for word in word_alignments:
            duration = word.end_time - word.start_time
            if duration < 0.02: 
                squashed_words += 1
        squash_ratio = squashed_words / len(word_alignments)
        return 1.0 - squash_ratio

    def _generate_diff_report(self, clean_reference, final_result, lang="zh"):
        norm_ref = self._normalize_text(clean_reference, lang)
        norm_result = self._normalize_text(final_result, lang)
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
            # 记录最原始未经任何清理的 Prompt 字符数
            raw_prompt_length = len(reference_text)
            
            clean_reference_text = re.sub(r'\s+', ' ', reference_text).strip()
            
            # 1. 前置一键全量编译为 IPA，并产出对应的映射数组 (若不支持语言，此处将直接被 Fail-Fast 阻断)
            ipa_ref, norm_ref, ipa_to_norm_map, ref_idx_map = self._build_index_map(clean_reference_text, run_lang)

            try:
                wav, sample_rate = self._load_audio(audio_path, req_id=req_id)
            except Exception as err:
                log_error(f"阶段: 音频读取失败 | 详情: {err}", req_id=req_id)
                raise

            total_audio_duration = len(wav) / sample_rate
            
            try:
                with self.infer_lock:
                    raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
                if not raw_speech_timestamps:
                    raise ValueError("VAD 未检测到任何有效人声。")
            except Exception as err:
                log_error(f"阶段: VAD 物理切片失败 | 详情: {err}", req_id=req_id)
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
                log_error(f"阶段: ASR 批量模型推理崩溃 | 详情: {err}", req_id=req_id)
                raise

            fa_batch_audios = []
            fa_batch_texts = []
            chunk_mapping_meta = []  

            current_ipa_cursor = 0 
            prev_orig_end = 0 
            
            epi = self._get_epitran(run_lang)

            try:
                for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
                    norm_asr = self._normalize_text(res.text, run_lang)
                    if not norm_asr:
                        continue
                    
                    # 2. 对当前切片 ASR 识别出的干净文本进行 IPA 音标编译
                    if epi is not None:
                        ipa_asr = epi.transliterate(norm_asr)
                    else:
                        ipa_asr = norm_asr

                    # 3. 接入基于国际音标与绝对寻址自愈的定位游标计算
                    abs_ipa_end, ratio, fail_msg = self._macro_route_end_cursor(ipa_asr, ipa_ref, current_ipa_cursor, idx, chunk_info)
                    orig_start = prev_orig_end

                    if abs_ipa_end is None:
                        log_warn(f"⚠️ 触发预警 | 第 {idx+1} 段 ASR 匹配极度混乱，启动 FA 强制纠偏...", req_id=req_id)
                        candidate_lengths = [100, 200, 300]
                        best_candidate_cursor = None
                        best_fa_score = -1
                        
                        current_chunk_audio = batch_audios[idx]

                        for length in candidate_lengths:
                            c_end = min(orig_start + length, len(clean_reference_text))
                            candidate_text = clean_reference_text[orig_start : c_end]
                            if not candidate_text.strip(): continue
                                
                            try:
                                with self.infer_lock:
                                    fa_res = self.fa_model.align(audio=[current_chunk_audio], text=[candidate_text], language=run_lang)[0]
                                score = self._evaluate_fa_health(fa_res)
                                if score > best_fa_score:
                                    best_fa_score = score
                                    best_candidate_cursor = orig_start + len(candidate_text)
                            except Exception:
                                pass

                        FA_HEALTH_THRESHOLD = 0.50
                        
                        if best_fa_score > FA_HEALTH_THRESHOLD and best_candidate_cursor is not None:
                            log_warn(f"✅ 纠偏成功 | FA 模型强力介入！置信度得分: {best_fa_score:.2f}，强行修正文本边界。", req_id=req_id)
                            orig_end = best_candidate_cursor
                            
                            new_ipa_cursor = current_ipa_cursor
                            for n_idx in range(current_ipa_cursor, len(ipa_ref)):
                                norm_pos = ipa_to_norm_map[n_idx]
                                if norm_pos < len(ref_idx_map) and ref_idx_map[norm_pos] >= orig_end:
                                    new_ipa_cursor = n_idx
                                    break
                            else:
                                new_ipa_cursor = len(ipa_ref)
                                
                            abs_ipa_end = new_ipa_cursor
                        else:
                            final_error = f"[🔴 彻底失败] 段落 {idx+1} 匹配极低，音标对齐与 FA 反哺纠偏均未通过。详细特征:\n    -> ASR: '{norm_asr[:50]}...'\n    -> IPA: '{ipa_asr[:50]}...'"
                            raise AlignmentThresholdError(final_error)
                    else:
                        if abs_ipa_end < len(ipa_ref): 
                            norm_pos = ipa_to_norm_map[abs_ipa_end]
                            if norm_pos < len(ref_idx_map):
                                orig_end = ref_idx_map[norm_pos]
                            else:
                                orig_end = len(clean_reference_text)
                        else: 
                            orig_end = len(clean_reference_text)
                        
                    ground_truth = clean_reference_text[orig_start : orig_end]
                    current_ipa_cursor = abs_ipa_end
                    prev_orig_end = orig_end
                    
                    if not ground_truth.strip():
                        continue

                    fa_batch_audios.append(batch_audios[idx])
                    fa_batch_texts.append(ground_truth)
                    chunk_mapping_meta.append(chunk_info)

            except AlignmentThresholdError as threshold_err:
                raise threshold_err
            except Exception as err:
                log_error(f"阶段: 文本对齐路由匹配失败 | 详情: {err}", req_id=req_id)
                raise

            verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "report": {}, "text": "", "words": []}

            if fa_batch_audios:
                try:
                    with self.infer_lock:
                        fa_results = self.fa_model.align(audio=fa_batch_audios, text=fa_batch_texts, language=run_lang)
                except Exception as err:
                    log_error(f"阶段: FA ForcedAligner 批量推理对齐失败 | 详情: {err}", req_id=req_id)
                    raise

# 提前判断是否为中文语系
                is_chinese_lang = any(kw in run_lang.lower() for kw in ["zh", "chinese", "cmn", "hans", "hant"])

                for chunk_info, single_fa_res in zip(chunk_mapping_meta, fa_results):
                    chunk_start = chunk_info['start']
                    for word_align in single_fa_res:
                        verbose_json["words"].append({
                            "word": word_align.text,
                            "start": round(chunk_start + word_align.start_time, 3),
                            "end": round(chunk_start + word_align.end_time, 3)
                        })
                        
                        # 核心修复：根据语系决定是否加空格
                        if is_chinese_lang:
                            verbose_json["text"] += word_align.text
                        else:
                            verbose_json["text"] += word_align.text + " "
                            
                # 循环结束后，剔除英文尾部多余的一个空格
                verbose_json["text"] = verbose_json["text"].strip()

            verbose_json["report"] = self._generate_diff_report(clean_reference_text, verbose_json["text"], run_lang)
            total_pipeline_time = time.time() - pipeline_start_time
            
            accuracy_str = verbose_json["report"].get("accuracy_ratio", "0.00%")
            accuracy_val = float(accuracy_str.strip('%'))
            align_status = "✅ 成功" if accuracy_val > 50 else "⚠️ 相似度低"
            ref_len = verbose_json["report"].get("ref_length", 0)
            res_len = verbose_json["report"].get("result_length", 0)

            log_info(f"📊 最终报告 (对齐模式 - 已升级国际音标架构) ============", req_id=req_id)
            log_info(f"  • 请求标识       : {req_id}", req_id=req_id)
            log_info(f"  • 执行状态       : {align_status} (文本相似度: {accuracy_str})", req_id=req_id)
            log_info(f"  • 物理切片数量   : 产生 {len(physical_chunks)} 段 -> 有效对齐 {len(fa_batch_audios)} 段", req_id=req_id)
            log_info(f"  • 字符匹配统计   : 原始 Prompt={raw_prompt_length}字 | 正则化后原文={ref_len}字 | 最终成功对齐={res_len}字", req_id=req_id)
            log_info(f"  • 音频总时长     : {total_audio_duration:.2f} 秒", req_id=req_id)
            log_info(f"  • 流水线总耗时   : {total_pipeline_time:.2f} 秒", req_id=req_id)
            log_info(f"  • RTF (实时率)   : {total_pipeline_time / total_audio_duration:.3f}", req_id=req_id)
            log_info("=" * 60, req_id=req_id)
            
            return verbose_json

        except Exception as outer_err:
            raise outer_err

    async def process_audio_asr_only(self, audio_path, language_code=None, req_id="Unknown"):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]

        try:
            try:
                wav, sample_rate = self._load_audio(audio_path, req_id=req_id)
            except Exception as err:
                log_error(f"阶段: 音频读取失败 | 详情: {err}", req_id=req_id)
                raise

            total_audio_duration = len(wav) / sample_rate
            
            try:
                with self.infer_lock:
                    raw_speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, return_seconds=True)
                if not raw_speech_timestamps:
                    raise ValueError("VAD 未检测到任何有效人声。")
            except Exception as err:
                log_error(f"阶段: VAD 物理切片失败 | 详情: {err}", req_id=req_id)
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
                log_error(f"阶段: 纯ASR模型批量推理崩溃 | 详情: {err}", req_id=req_id)
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

            log_info("📊 最终报告 (纯ASR模式) ============", req_id=req_id)
            log_info(f"  • 请求标识       : {req_id}", req_id=req_id)
            log_info(f"  • 执行状态       : ✅ 成功完成", req_id=req_id)
            log_info(f"  • 切片与分段     : VAD物理切片 {len(physical_chunks)} 段 -> 生成 {seg_count} 个识别段落", req_id=req_id)
            log_info(f"  • 识别字符统计   : 共计生成 {char_count} 个字符", req_id=req_id)
            log_info(f"  • 音频总时长     : {total_audio_duration:.2f} 秒", req_id=req_id)
            log_info(f"  • ASR 推理耗时   : {asr_stage_duration:.2f} 秒", req_id=req_id)
            log_info(f"  • 流水线总耗时   : {total_pipeline_time:.2f} 秒", req_id=req_id)
            log_info(f"  • RTF (实时率)   : {total_pipeline_time / total_audio_duration:.3f}", req_id=req_id)
            log_info("=" * 55, req_id=req_id)

            return verbose_json
            
        except Exception as outer_err:
            raise outer_err