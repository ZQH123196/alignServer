import os
import re
import time
import torch
import Levenshtein
import soundfile as sf
import numpy as np
import threading
import unicodedata

from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from env_config import server_env
from utils import get_container_cpu_count
from silero_vad import load_silero_vad, get_speech_timestamps
from fa_timestamp_validation import (
    FaTimestampError,
    validate_absolute_timestamps,
    validate_fa_timestamps_in_chunk,
)

# 共享 main.py 的并发可视化日志（log/register/mark_*）—— 任务执行过程打印时
# 会自动附上时间戳、级别、req_id 与"当前任务队列快照"，方便实时看到并行任务与各自处理的文件。
# [修复] main.py 作为入口脚本运行时，Python 把它注册为 sys.modules['__main__']
# 而不是 'main'。下面这一行 `from main import ...` 会让 Python 找不到 'main'
# 而把 main.py 当成独立模块重新执行，从而触发循环导入。
# 这里在导入之前把当前 __main__ 同步到 'main'，让 Python 使用部分加载的模块。
import sys as _sys
if 'main' not in _sys.modules and '__main__' in _sys.modules:
    _sys.modules['main'] = _sys.modules['__main__']
from main import (
    log, log_info, log_warn, log_error, log_block,
    register, mark_running, mark_done, mark_failed, unregister, reset_for_retry,
    get_task_info,
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
    """首尾对齐彻底失败。携带 diag_lines 供调用方聚合打印。"""
    def __init__(self, message, diag_lines=None):
        super().__init__(message)
        self.diag_lines = diag_lines or []


class AsrBatchScheduler:
    """
    ASR 单实例动态批处理调度器。

    背景：Qwen3ASRModel.generate() 非线程安全（HF generate 维护实例级 position_ids /
    KV cache / generation_config 状态），多线程并发调用同一实例必崩（见
    modeling_qwen3_asr.py:1227 `position_ids.add(delta)` 的 batch 维度串台）。
    Qwen3-ASR 的 transcribe() 本身就是 batch 接口（内部 padding=True + 自动切分 +
    按 orig_index 合并），单线程调用 + 大 batch 即可吃满 GPU，无需多实例。

    工作机制：
      - submit(): 多个并发请求把 audios 列表丢进 pending 池后阻塞等待结果
      - worker: 单线程，等 max_wait_ms 攒批 -> 拼成一次 transcribe 调用 ->
        按长度切回 -> 唤醒各请求
      - max_batch_size: 每次 flush 最多取多少个 audio（沿用模型 max_inference_batch_size=32）
      - max_pending_chunks: pending 池软上限，防内存爆（背压，Fail-Fast 抛错）

    独立性保证：transcribe 内部 padding 用静音零值 + attention_mask 屏蔽，batch 内
    各样本 attention/KV cache/generate 严格 per-sample 隔离，结果与单独调用数学等价。
    """

    def __init__(self, asr_model, max_batch_size=32, max_wait_ms=50, max_pending_chunks=256):
        self.asr_model = asr_model
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = int(max_wait_ms)
        self.max_pending_chunks = int(max_pending_chunks)

        self._pending = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._flush_loop, name="AsrBatchScheduler", daemon=True)
        self._worker.start()

    def submit(self, audios, language, req_id="Unknown", sampling_params=None):
        """提交本请求的所有 audio chunks（带采样参数），阻塞直到结果返回。

        sampling_params：可选 vLLM SamplingParams（「生成定调文本（calibration）」温度采样用）；
        None = 贪心解码。flush 时按采样签名分组，贪心请求不与采样请求混批。
        超过 max_pending_chunks 时分批排队（等前一批消化完再提交下一批），不拒绝。
        """
        if not audios:
            return []

        results = [None] * len(audios)
        offset = 0
        while offset < len(audios):
            batch = audios[offset:offset + self.max_pending_chunks]
            batch_results = self._submit_batch(batch, language, req_id, sampling_params)
            results[offset:offset + len(batch)] = batch_results
            offset += len(batch)
        return results

    def _submit_batch(self, audios, language, req_id="Unknown", sampling_params=None):
        """提交一批 audios（不超过 max_pending_chunks），阻塞等待结果。"""
        if not audios:
            return []

        count = len(audios)
        event = threading.Event()
        holder = [None]

        with self._lock:
            self._pending.append((audios, language, sampling_params, holder, event, count))
            self._cv.notify()

        event.wait()

        result = holder[0]
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def _sampling_signature(sampling_params):
        """采样参数签名：贪心请求与不同采样参数的请求不得混批。"""
        if sampling_params is None:
            return "greedy"
        return (sampling_params.temperature, sampling_params.top_p, sampling_params.n)

    def _flush_loop(self):
        """单线程 worker：攒批 -> 按采样参数分组 -> 调用 transcribe -> 切分结果 -> 唤醒等待者。"""
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                self._cv.wait(timeout=self.max_wait_ms / 1000.0)

                buckets = {}
                remaining = []
                for item in self._pending:
                    audios, lang, sampling, holder, event, count = item
                    key = self._sampling_signature(sampling)
                    bucket = buckets.get(key)
                    if bucket is None:
                        bucket = {"sampling": sampling, "items": [], "taken": 0}
                        buckets[key] = bucket
                    if bucket["taken"] == 0 or bucket["taken"] + count <= self.max_batch_size:
                        bucket["items"].append(item)
                        bucket["taken"] += count
                    else:
                        remaining.append(item)
                self._pending = remaining

            if not buckets:
                continue

            for bucket in buckets.values():
                batch = bucket["items"]
                all_audios = []
                all_langs = []
                for audios, lang, _, _, _, count in batch:
                    all_audios.extend(audios)
                    all_langs.extend([lang] * count)

                try:
                    results = self.asr_model.transcribe(
                        audio=all_audios,
                        language=all_langs,
                        sampling_params=bucket["sampling"],
                    )
                except Exception as err:
                    for _, _, _, holder, event, _ in batch:
                        holder[0] = err
                        event.set()
                    continue

                i = 0
                for _, _, _, holder, event, count in batch:
                    holder[0] = results[i:i + count]
                    event.set()
                    i += count


class Qwen3ASRSamplingModel(Qwen3ASRModel):
    """支持 per-call 解码参数的 Qwen3-ASR 包装。

    背景：qwen_asr v0.0.6 的 transcribe() 不暴露解码参数，vLLM SamplingParams
    在 from_pretrained() 时写死 temperature=0.0（贪心）。「生成定调文本（calibration）」
    需要温度采样产生多样候选做共识投票；对齐模式必须保持贪心。
    AsrBatchScheduler 保证单线程调用 generate()（见其类文档），因此临时替换
    sampling_params 是安全的。
    """

    def transcribe(
        self,
        audio,
        context="",
        language=None,
        return_time_stamps=False,
        sampling_params=None,
    ):
        if sampling_params is None:
            return super().transcribe(audio, context, language, return_time_stamps)
        prev = self.sampling_params
        self.sampling_params = sampling_params
        try:
            return super().transcribe(audio, context, language, return_time_stamps)
        finally:
            self.sampling_params = prev


def build_sampling_params(asr_model, temperature=None, top_p=None, n=None):
    """构造 vLLM SamplingParams；任一解码参数给定才构造，全 None 返回 None（贪心）。

    max_tokens 沿用模型初始化时的采样参数，避免新参数覆盖生成长度限制。
    """
    if temperature is None and top_p is None and n is None:
        return None
    from vllm import SamplingParams

    base = getattr(asr_model, "sampling_params", None)
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if n is not None:
        kwargs["n"] = n
    max_tokens = getattr(base, "max_tokens", None)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return SamplingParams(**kwargs)


class LongAudioAlignPipeline:
    def __init__(self, config):
        self.config = config
        # DEVICE / 模型路径已在启动期校验完毕，这里只取校验后的结果，不再读环境变量
        self.device = server_env().device
        
        # 并发模型说明（Qwen3-ASR/FA 实测非线程安全，见 AsrBatchScheduler 文档）：
        #   - VAD: silero-vad 内部维护 LSTM 隐状态 -> vad_lock 串行
        #   - FA:  generate() 非线程安全 -> fa_lock 串行（FA 非吞吐主路径）
        #   - ASR: generate() 非线程安全 -> AsrBatchScheduler 单线程 worker 动态批处理
        #          （单实例 + 大 batch 吃满 GPU，跨请求合并，无并发污染）
        self.vad_lock = threading.Lock()
        self.fa_lock = threading.Lock()
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
                print(f"  GPU 模式 | dtype={self.dtype} | Flash Attention 2")
            except ImportError:
                model_kwargs["attn_implementation"] = "sdpa"
                print(f"  GPU 模式 | dtype={self.dtype} | SDPA")

            total_cores = get_container_cpu_count()
            cpu_ratio = config.get("cpu_ratio", 0.5)
            allocated_cores = max(1, int(total_cores * cpu_ratio))
            torch.set_num_threads(allocated_cores)
        else:
            self.dtype = torch.float32
            print("  CPU 模式 | 单线程")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1) 

        model_kwargs["torch_dtype"] = self.dtype
        asr_path = server_env().asr_model_path
        fa_path = server_env().fa_model_path

        print(f"  设备: {self.device} | ASR: {asr_path} | FA: {fa_path}")

        self.vad_model = load_silero_vad() 
        self.get_speech_timestamps = get_speech_timestamps
        self.asr_model = Qwen3ASRSamplingModel.from_pretrained(asr_path, **model_kwargs)
        # Qwen3-ForcedAligner 边缘情况：FA 是强制对齐器，不做文本识别。
        # fa_res.text 永远是输入文本的回声；输入过量/不匹配音频的文本时，
        # FA 仍会给所有词分配时间戳（伪插值），结果毫无价值。
        # 实测：在原音频文本前后各加一段非音频文本，FA 仍输出全部文本并给时间戳。
        # 因此不能用 fa_res.text 做相似度判定，只能靠上游分割保证 ground_truth 长度合理。
        self.fa_model = Qwen3ForcedAligner.from_pretrained(fa_path, **model_kwargs)
        
        # ASR 动态批处理调度器：单线程串行调用 generate()（消除并发污染），
        # 跨请求合并 chunks 攒大 batch 提 GPU 利用率。max_batch_size 沿用模型默认 32。
        self.asr_scheduler = AsrBatchScheduler(
            asr_model=self.asr_model,
            max_batch_size=self.asr_model.max_inference_batch_size if self.asr_model.max_inference_batch_size > 0 else 32,
            max_wait_ms=config.get("asr_batch_wait_ms", 50),
            max_pending_chunks=config.get("asr_max_pending_chunks", 512),
        )
        
        self._epitran_cache = {}
        self._cache_lock = threading.Lock()
        print("  模型加载完成")

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
                print(f"  Epitran: '{lang}' -> '{epitran_code}'")
                epi = epitran.Epitran(epitran_code)
                self._epitran_cache[epitran_code] = epi
                return epi
            except Exception as e:
                print(f"  Epitran 载入失败 {epitran_code}: {e}")
                raise ValueError(f"Failed to initialize Epitran for code '{epitran_code}': {e}")

    @staticmethod
    def _uses_character_tokens(lang):
        """Return whether the language should be aligned character by character."""
        key = (lang or "zh").lower().strip().replace("_", "-")
        return key.startswith(("zh", "cmn", "chinese", "ja", "jpn", "japanese"))

    @staticmethod
    def _is_chinese_language(lang):
        key = (lang or "zh").lower().strip().replace("_", "-")
        return key.startswith(("zh", "cmn", "chinese"))

    @staticmethod
    def _is_text_character(char):
        """Keep Unicode letters/numbers while dropping punctuation and symbols."""
        return char.isalnum() or unicodedata.category(char).startswith("L")

    def _normalize_text(self, text, lang="zh"):
        """Normalize text without assuming that every non-Chinese language is Latin."""
        lang = (lang or "zh").lower().strip()
        uses_character_tokens = self._uses_character_tokens(lang)

        if uses_character_tokens:
            text_sim = zhconv.convert(text, 'zh-cn') if self._is_chinese_language(lang) else text
            text_clean = "".join(char for char in text_sim if self._is_text_character(char))
        else:
            # Whitespace-delimited languages retain spaces; Unicode scripts such
            # as Cyrillic must not be reduced to ASCII before matching.
            text_clean = text.lower()
            text_clean = "".join(char for char in text_clean if self._is_text_character(char) or char.isspace())
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        return text_clean

    def _build_index_map(self, original_text, lang="zh"):
        """
        三位一体的索引映射构建器（新增 IPA 编译层）
        建立 `IPA 下标 -> norm 字符下标 -> 原始字符下标` 复合映射链
        """
        norm_chars = []  # 用 list 累积，最后 join，避免 += 的 O(n²) 拷贝
        idx_map = [] # 记录 norm_text 的每个字在 original_text 中的索引
        
        lang = (lang or "zh").lower().strip()
        uses_character_tokens = self._uses_character_tokens(lang)
        
        # 1. 物理对齐 norm_text 与 original_text 的索引
        if uses_character_tokens:
            sim_text = zhconv.convert(original_text, 'zh-cn') if self._is_chinese_language(lang) else original_text
            for i, char in enumerate(sim_text):
                if self._is_text_character(char):
                    norm_chars.append(char)
                    idx_map.append(i)
        else:
            orig_lower = original_text.lower()
            for i, char in enumerate(orig_lower):
                if self._is_text_character(char) or char.isspace():
                    norm_chars.append(char)
                    idx_map.append(i)

        norm_text = "".join(norm_chars)

        # 2. 调用 Epitran 对 norm_text 全文编译为 IPA (Fail-Fast 会在这里被触发)
        epi = self._get_epitran(lang)
        
        ipa_parts = []  # 用 list 累积 IPA 片段，最后 join，避免 += 的 O(n²) 拷贝
        ipa_to_norm_map = [] # 用于从 ipa_text 下标查回 norm_text 对应的字符下标
        
        if epi is None:
            # 如果 Epitran 加载失败，无缝降级回纯文本
            return norm_text, norm_text, list(range(len(norm_text))), idx_map

        # Epitran 以词为单位进行音素编译效率最高
        words = list(norm_text) if uses_character_tokens else norm_text.split()
        
        current_search_offset = 0  # [新增] 维护一个全局向前推进的搜索游标
        
        for word_idx, word in enumerate(words):
            # 将单词编译为 IPA 音素
            word_ipa = epi.transliterate(word)
            # 记录当前单词在原 norm_text 中的字符索引区间
            if not uses_character_tokens:
                # [修改] 强制从 current_search_offset 之后开始寻找，防止高频词导致时光倒流
                norm_word_start = norm_text.find(word, current_search_offset) 
                if norm_word_start == -1: 
                    norm_word_start = current_search_offset
                
                # [新增] 更新游标，跳过当前单词长度
                current_search_offset = norm_word_start + len(word)
            else:
                norm_word_start = word_idx
            
            # 为该单词生成的每一个 IPA 字符映射回它原属的 norm 字符起点
            if word_ipa:
                ipa_parts.append(word_ipa)
                ipa_to_norm_map.extend([norm_word_start] * len(word_ipa))
                
            # Whitespace-delimited languages retain word boundaries in IPA.
            if not uses_character_tokens and word_idx < len(words) - 1:
                ipa_parts.append(" ")
                # 空格对应原 norm 中的空格
                ipa_to_norm_map.append(norm_word_start + len(word))

        ipa_text = "".join(ipa_parts)
        return ipa_text, norm_text, ipa_to_norm_map, idx_map

    def _hard_match_asr_to_ref(self, asr_text, is_forward, norm_ref, ref_idx_map,
                                 min_words=1, max_words=8, language=None):
        """
        ASR 末尾/开头词在归一化原文(norm_ref)里找唯一匹配位置（硬匹配）。
        对称模型：is_forward=True 取 ASR 末尾词前向找（首/中间 chunk），
                  is_forward=False 取 ASR 开头词反向找（尾 chunk）。

        匹配策略：从 min_words 个词开始，在 norm_ref 里找匹配；
                  唯一匹配 -> 成功；不唯一 -> 加词到 max_words；
                  耗尽仍不唯一或无匹配 -> 失败。

        用 norm_ref（已剥标点/空白）比对，命中后通过 ref_idx_map 映射回 clean_reference_text 坐标。

        Returns: (orig_pos, matched_words, matched_str) | 失败 (None, 0, "")
          - is_forward=True: orig_pos = ASR 末尾词在原文中的结束位置（clean_reference_text 下标）
          - is_forward=False: orig_pos = ASR 开头词在原文中的起始位置（clean_reference_text 下标）
          - matched_str: 匹配到的词串（用于日志打印）
        """
        if not asr_text or not norm_ref:
            return None, 0, ""

        uses_character_tokens = self._uses_character_tokens(
            language or self.config.get("language", "zh")
        )
        if uses_character_tokens:
            asr_words = list(asr_text)
            ref_words = list(norm_ref)
        else:
            asr_words = asr_text.split()
            ref_words = norm_ref.split()

        if not asr_words or not ref_words:
            return None, 0, ""

        for n_words in range(min_words, max_words + 1):
            if len(asr_words) < n_words:
                continue
            if is_forward:
                query = asr_words[-n_words:]
            else:
                query = asr_words[:n_words]
            query_str = "".join(query) if uses_character_tokens else " ".join(query)

            # 词边界匹配：query 在 norm_ref 里必须出现在词边界处
            # Character-token languages use direct substring matching.
            # Word-token languages require whitespace boundaries (防止 "ined" 匹配到 "combined" 中间).
            if uses_character_tokens:
                count = norm_ref.count(query_str)
                if count == 1:
                    norm_pos = norm_ref.find(query_str)
                    if is_forward:
                        norm_end = norm_pos + len(query_str)
                        if norm_end - 1 < len(ref_idx_map):
                            return ref_idx_map[norm_end - 1] + 1, n_words, query_str
                        elif len(ref_idx_map) > 0:
                            return ref_idx_map[-1] + 1, n_words, query_str
                    else:
                        if norm_pos < len(ref_idx_map):
                            return ref_idx_map[norm_pos], n_words, query_str
            else:
                # 英文：用正则确保词边界
                import re as _re
                # 转义 query 里的正则特殊字符
                escaped = _re.escape(query_str)
                # query 前面必须是行首或空格，后面必须是行尾或空格
                pattern = r'(?:^|\s)' + escaped + r'(?:$|\s)'
                matches = list(_re.finditer(pattern, norm_ref))
                if len(matches) == 1:
                    m = matches[0]
                    # 匹配到的 query 在 norm_ref 中的起始位置（去掉前导空格）
                    norm_pos = m.start() + (1 if m.group().startswith(' ') or m.group().startswith('\n') else 0)
                    if is_forward:
                        norm_end = norm_pos + len(query_str)
                        if norm_end - 1 < len(ref_idx_map):
                            return ref_idx_map[norm_end - 1] + 1, n_words, query_str
                        elif len(ref_idx_map) > 0:
                            return ref_idx_map[-1] + 1, n_words, query_str
                    else:
                        if norm_pos < len(ref_idx_map):
                            return ref_idx_map[norm_pos], n_words, query_str
        return None, 0, ""

    def _group_vad_by_asr_hardmatch(self, raw_speech_timestamps, total_audio_duration, wav, sample_rate,
                                     clean_reference_text, norm_ref, ref_idx_map, run_lang, req_id="Unknown"):
        """
        ASR 硬匹配合并（对称模型，首尾优先）：
          - 首 chunk：锚点 start=0 固定，沿 text_len 方向贪婪合并到 max_sec，
            ASR 末尾词硬匹配原文，匹配不到则回退一块细分片重试，取最大可匹配 chunk。
          - 尾 chunk：锚点 end=text_len 固定，沿 0 方向贪婪合并到 max_sec，
            ASR 开头词硬匹配原文，匹配不到则回退一块细分片重试，取最大可匹配 chunk。
          - 中间 chunk：首尾定界后，在剩余细分片区间内前向贪婪合并，
            ASR 末尾词硬匹配原文，回退找最大可匹配 chunk。

        失败兜底：硬匹配不到则 fallback 到现有时长切片（_group_vad_timestamps），
                  并打印警告。

        Returns: (physical_chunks, asr_results, orig_boundaries, fallback_used)
          - physical_chunks: 合并后的 chunk 列表
          - asr_results: 每个 chunk 的 ASR 结果（复用，不重复推理）
          - orig_boundaries: 每个 chunk 的 (orig_start, orig_end)，硬匹配命中则为硬锚点，None 为未命中
          - fallback_used: 是否 fallback 到时长切片
        """
        if not raw_speech_timestamps:
            return [], [], [], True

        max_sec = self.config["chunk_max_sec"]
        min_edge_sec = self.config.get("min_tail_sec", 30)
        text_len = len(clean_reference_text)
        uses_character_tokens = self._uses_character_tokens(run_lang)

        # 先按细分片跑 ASR（一次性全跑，后续合并时拼接文本复用）
        seg_audios = []
        for ts in raw_speech_timestamps:
            s = max(0, int(ts['start'] * sample_rate))
            e = min(len(wav), int(ts['end'] * sample_rate))
            seg_audios.append((wav[s:e].numpy(), sample_rate))

        try:
            seg_asr_results = self.asr_scheduler.submit(audios=seg_audios, language=run_lang, req_id=req_id)
        except Exception as err:
            log_error(f"阶段: ASR 硬匹配预推理失败 | 详情: {err}", req_id=req_id)
            # fallback 到时长切片
            chunks = self._group_vad_timestamps(raw_speech_timestamps, total_audio_duration)
            return chunks, [], [(None, None)] * len(chunks), True

        # 归一化每个细分片 ASR 文本
        seg_norm_texts = []
        for res in seg_asr_results:
            norm = self._normalize_text(res.text, run_lang) if res and res.text else ""
            seg_norm_texts.append(norm)

        n_segs = len(raw_speech_timestamps)
        warning_lines = []

        def _combine_asr(segs_slice):
            parts = []
            for t in segs_slice:
                if t: parts.append(t)
            return "".join(parts) if uses_character_tokens else " ".join(parts)

        def _fa_check_chunk(seg_start, seg_end_exclusive, text_start, text_end):
            """对候选 chunk 跑 FA 健康度，>= 阈值则通过。"""
            chunk_s = max(0, int(raw_speech_timestamps[seg_start]['start'] * sample_rate))
            chunk_e = min(len(wav), int(raw_speech_timestamps[seg_end_exclusive - 1]['end'] * sample_rate))
            chunk_audio = (wav[chunk_s:chunk_e].numpy(), sample_rate)
            chunk_dur = raw_speech_timestamps[seg_end_exclusive - 1]['end'] - raw_speech_timestamps[seg_start]['start']
            gt = clean_reference_text[text_start:text_end]
            if not gt.strip():
                return False
            try:
                with self.fa_lock:
                    fa_res = self.fa_model.align(audio=[chunk_audio], text=[gt], language=run_lang)[0]
                    self._assert_fa_timestamps_in_chunk(
                        fa_res, raw_speech_timestamps[seg_start]['start'],
                        raw_speech_timestamps[seg_end_exclusive - 1]['end'], req_id,
                    )
                    score = self._evaluate_fa_health(fa_res, chunk_dur, run_lang=run_lang)
                return score >= self.ASR_FA_HEALTH_THRESHOLD
            except FaTimestampError:
                raise
            except Exception:
                return False

        def _greedy_match_forward(seg_start, seg_end_max, cursor_orig):
            """
            前向贪婪：从 seg_start 合并到 seg_end_max，取 ASR 末尾词硬匹配。
            策略：先累加到 max（或 seg_end_max），匹配不到则回退一块重试。
            硬匹配命中后追加 FA 健康度校验，不通过则继续回退。
            Returns: (seg_end, orig_end, n_words, matched_str) | 失败 (None, None, 0, "")
            """
            # 先累加到 max_sec 或 seg_end_max
            j = seg_start + 1
            while j < seg_end_max:
                chunk_dur = raw_speech_timestamps[j]['end'] - raw_speech_timestamps[seg_start]['start']
                if chunk_dur > max_sec: break
                j += 1
            # j 是第一个超 max_sec 或 seg_end_max 的位置，候选范围 [seg_start, j)
            # 从最大开始回退找匹配
            for end_idx in range(j, seg_start, -1):
                if end_idx <= seg_start: continue
                combined = _combine_asr(seg_norm_texts[seg_start:end_idx])
                if not combined.strip(): continue
                orig_pos, n_words, matched = self._hard_match_asr_to_ref(
                    combined, is_forward=True, norm_ref=norm_ref, ref_idx_map=ref_idx_map,
                    language=run_lang,
                )
                if orig_pos is not None and orig_pos > cursor_orig:
                    if _fa_check_chunk(seg_start, end_idx, cursor_orig, orig_pos):
                        return end_idx, orig_pos, n_words, matched
            return None, None, 0, ""

        def _greedy_match_backward(seg_end_min, seg_end, text_len, text_lower_bound=0):
            """
            反向贪婪：从 seg_end 向首部合并到 seg_end_min，取 ASR 开头词硬匹配。
            策略：先累加到 max（或 seg_end_min），匹配不到则回退一块重试。
            硬匹配命中后追加 FA 健康度校验，不通过则继续回退。
            text_lower_bound: 匹配到的 orig_pos 必须 > 此值（防止尾 chunk 匹配到首 chunk 已覆盖区域）。
            Returns: (seg_start, orig_start, n_words, matched_str) | 失败 (None, None, 0, "")
            """
            # 先向首部累加到 max_sec 或 seg_end_min
            k = seg_end - 1
            while k > seg_end_min:
                chunk_dur = raw_speech_timestamps[seg_end - 1]['end'] - raw_speech_timestamps[k]['start']
                if chunk_dur > max_sec: break
                k -= 1
            # 候选范围 [k, seg_end)，从最大开始回退找匹配
            for start_idx in range(k, seg_end):
                if start_idx >= seg_end: continue
                combined = _combine_asr(seg_norm_texts[start_idx:seg_end])
                if not combined.strip(): continue
                orig_pos, n_words, matched = self._hard_match_asr_to_ref(
                    combined, is_forward=False, norm_ref=norm_ref, ref_idx_map=ref_idx_map,
                    language=run_lang,
                )
                if orig_pos is not None and orig_pos < text_len and orig_pos > text_lower_bound:
                    if _fa_check_chunk(start_idx, seg_end, orig_pos, text_len):
                        return start_idx, orig_pos, n_words, matched
            return None, None, 0, ""

        # ── 首 chunk：从 0 前向贪婪合并，ASR 末尾词硬匹配原文 ──
        # 估算尾 chunk 至少需要的细分片数
        avg_seg_dur = total_audio_duration / n_segs if n_segs > 0 else 1.0
        min_tail_segs = max(2, int(min_edge_sec / max(avg_seg_dur, 1.0)) + 1)

        head_seg_end_max = n_segs - min_tail_segs  # 首 chunk 最多用到这
        # 首 chunk 至少留 min_head_segs 给中间 + 尾
        head_end_idx, head_orig_end, head_n_words, head_matched = _greedy_match_forward(0, head_seg_end_max, 0)

        # ── 尾 chunk：从末尾反向贪婪合并，ASR 开头词硬匹配原文 ──
        tail_seg_start_min = head_end_idx if head_end_idx else 0
        tail_text_lower_bound = head_orig_end if head_orig_end else 0
        tail_start_idx, tail_orig_start, tail_n_words, tail_matched = _greedy_match_backward(
            tail_seg_start_min, n_segs, text_len, text_lower_bound=tail_text_lower_bound
        )

        # ── 组装结果 ──
        chunks = []
        asr_results = []
        orig_boundaries = []

        # 首 chunk
        if head_end_idx is not None:
            chunks.append({'start': raw_speech_timestamps[0]['start'], 'end': raw_speech_timestamps[head_end_idx - 1]['end']})
            asr_results.append(self._combine_seg_asr(seg_asr_results[0:head_end_idx], run_lang))
            orig_boundaries.append((0, head_orig_end))
            warning_lines.append(
                f"  首 chunk 硬匹配成功 | segs[0:{head_end_idx}] | 原文[0:{head_orig_end}] | "
                f"末尾{head_n_words}词唯一匹配: \"{head_matched}\""
            )
            cursor_seg = head_end_idx
            cursor_orig = head_orig_end
        else:
            # 首 chunk fallback：切到 max_sec 或 head_seg_end_max
            j = 1
            while j < head_seg_end_max:
                chunk_dur = raw_speech_timestamps[j]['end'] - raw_speech_timestamps[0]['start']
                if chunk_dur > max_sec: break
                j += 1
            if j <= 0: j = 1
            chunks.append({'start': raw_speech_timestamps[0]['start'], 'end': raw_speech_timestamps[j - 1]['end']})
            asr_results.append(self._combine_seg_asr(seg_asr_results[0:j], run_lang))
            orig_boundaries.append((0, None))
            warning_lines.append(
                f"  ⚠️ 首 chunk 硬匹配失败 | segs[0:{j}] | 音频{raw_speech_timestamps[j-1]['end'] - raw_speech_timestamps[0]['start']:.1f}s | fallback，下游 IPA 兜底"
            )
            cursor_seg = j
            cursor_orig = 0  # fallback 时 cursor_orig 保持 0，下游寻路自己算

        # 中间 chunk：在 [cursor_seg, tail_start_idx) 区间内前向贪婪合并
        mid_end = tail_start_idx if tail_start_idx is not None else n_segs
        while cursor_seg < mid_end:
            mid_end_idx, mid_orig_end, mid_n_words, mid_matched = _greedy_match_forward(cursor_seg, mid_end, cursor_orig)
            if mid_end_idx is not None:
                chunks.append({'start': raw_speech_timestamps[cursor_seg]['start'], 'end': raw_speech_timestamps[mid_end_idx - 1]['end']})
                asr_results.append(self._combine_seg_asr(seg_asr_results[cursor_seg:mid_end_idx], run_lang))
                orig_boundaries.append((cursor_orig, mid_orig_end))
                warning_lines.append(
                    f"  chunk[{len(chunks)-1}] 硬匹配成功 | segs[{cursor_seg}:{mid_end_idx}] | "
                    f"原文[{cursor_orig}:{mid_orig_end}] | 末尾{mid_n_words}词唯一匹配: \"{mid_matched}\""
                )
                cursor_seg = mid_end_idx
                cursor_orig = mid_orig_end
            else:
                # fallback：切到 max_sec
                j = cursor_seg + 1
                while j < mid_end:
                    chunk_dur = raw_speech_timestamps[j]['end'] - raw_speech_timestamps[cursor_seg]['start']
                    if chunk_dur > max_sec: break
                    j += 1
                if j <= cursor_seg: j = cursor_seg + 1
                chunks.append({'start': raw_speech_timestamps[cursor_seg]['start'], 'end': raw_speech_timestamps[j - 1]['end']})
                asr_results.append(self._combine_seg_asr(seg_asr_results[cursor_seg:j], run_lang))
                orig_boundaries.append((cursor_orig, None))
                warning_lines.append(
                    f"  ⚠️ chunk[{len(chunks)-1}] 硬匹配失败 | segs[{cursor_seg}:{j}] | "
                    f"音频{raw_speech_timestamps[j-1]['end'] - raw_speech_timestamps[cursor_seg]['start']:.1f}s | fallback，下游 IPA 兜底"
                )
                cursor_seg = j

        # 尾 chunk
        if tail_start_idx is not None:
            chunks.append({'start': raw_speech_timestamps[tail_start_idx]['start'], 'end': raw_speech_timestamps[n_segs - 1]['end']})
            asr_results.append(self._combine_seg_asr(seg_asr_results[tail_start_idx:n_segs], run_lang))
            orig_boundaries.append((tail_orig_start, text_len))
            warning_lines.append(
                f"  尾 chunk 硬匹配成功 | segs[{tail_start_idx}:{n_segs}] | "
                f"原文[{tail_orig_start}:{text_len}] | 开头{tail_n_words}词唯一匹配: \"{tail_matched}\""
            )
        elif cursor_seg < n_segs:
            # 尾 fallback：剩余全给一个 chunk
            chunks.append({'start': raw_speech_timestamps[cursor_seg]['start'], 'end': raw_speech_timestamps[n_segs - 1]['end']})
            asr_results.append(self._combine_seg_asr(seg_asr_results[cursor_seg:n_segs], run_lang))
            orig_boundaries.append((None, text_len))
            warning_lines.append(
                f"  ⚠️ 尾 chunk 硬匹配失败 | segs[{cursor_seg}:{n_segs}] | "
                f"音频{raw_speech_timestamps[n_segs-1]['end'] - raw_speech_timestamps[cursor_seg]['start']:.1f}s | fallback，下游反向 IPA 兜底"
            )

        # 打印硬匹配结果汇总
        if warning_lines:
            log_block(
                ["━━━ ASR 硬匹配合并诊断 ━━━"] + warning_lines,
                req_id=req_id, level="INFO"
            )

        # VAD padding
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

        return padded_chunks, asr_results, orig_boundaries, False

    def _combine_seg_asr(self, seg_asr_results, run_lang):
        """合并多个细分片 ASR 结果为一个（拼接 text，保留 language 字段）。"""
        if not seg_asr_results:
            return None
        uses_character_tokens = self._uses_character_tokens(run_lang)
        parts = []
        for res in seg_asr_results:
            if res and getattr(res, 'text', None):
                parts.append(res.text)
            combined_text = "".join(parts) if uses_character_tokens else " ".join(parts)
        first = seg_asr_results[0]
        return type(first)(
            language=getattr(first, 'language', run_lang),
            text=combined_text,
            time_stamps=None,
        )

    def _group_vad_timestamps(self, timestamps, total_audio_duration, max_sec=None):
        if not timestamps: return []
        if max_sec is None: max_sec = self.config["chunk_max_sec"]

        # FA (Qwen3-ForcedAligner-0.6B) 单次对齐上限 5 分钟，留 30s 保险余量
        FA_MAX_SEC = 270

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

        # 首尾优先足量：过短则合并，避免残片让 FA 验证失效
        # 整个 pipeline 假设每 chunk 能承载足够文本；首尾时间戳最关键，必须足量
        min_edge_sec = self.config.get("min_tail_sec", 30)

        def _try_split_at_boundary(c_left, c_right):
            """合并会超 FA_MAX_SEC 时，按 speech 边界把两 chunk 时长均分。"""
            split_point = (c_left['start'] + c_right['end']) / 2.0
            ts_starts = [t['start'] for t in timestamps
                         if c_left['start'] < t['start'] < c_right['end']]
            if not ts_starts:
                return None
            closest = min(ts_starts, key=lambda t: abs(t - split_point))
            if c_left['start'] < closest < c_right['end']:
                left_dur = closest - c_left['start']
                right_dur = c_right['end'] - closest
                # 均分后两侧都得满足 min_edge_sec 且不超 FA_MAX_SEC
                if left_dur >= min_edge_sec and right_dur >= min_edge_sec \
                   and left_dur <= FA_MAX_SEC and right_dur <= FA_MAX_SEC:
                    return {'start': c_left['start'], 'end': closest}
            return None

        # 尾 chunk：过短则与前一 chunk 合并或均分
        while len(chunks) >= 2 and (chunks[-1]['end'] - chunks[-1]['start']) < min_edge_sec:
            prev = chunks[-2]
            tail = chunks[-1]
            merged = {'start': prev['start'], 'end': tail['end']}
            merged_dur = tail['end'] - prev['start']
            if merged_dur <= FA_MAX_SEC:
                chunks[-2] = merged
                chunks.pop()
            else:
                split = _try_split_at_boundary(prev, tail)
                if split is None:
                    break  # 无法均分，保留原状（避免无限循环）
                chunks[-2] = split
                chunks[-1] = {'start': split['end'], 'end': tail['end']}

        # 首 chunk：过短则与后一 chunk 合并或均分
        while len(chunks) >= 2 and (chunks[0]['end'] - chunks[0]['start']) < min_edge_sec:
            head = chunks[0]
            nxt = chunks[1]
            merged = {'start': head['start'], 'end': nxt['end']}
            merged_dur = nxt['end'] - head['start']
            if merged_dur <= FA_MAX_SEC:
                chunks[0] = merged
                chunks.pop(1)
            else:
                split = _try_split_at_boundary(head, nxt)
                if split is None:
                    break
                chunks[0] = split
                chunks[1] = {'start': split['end'], 'end': nxt['end']}

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
                log_warn(f"全局自愈成功 | 物理切片 {chunk_idx + 1} 定位崩溃！已抓取核心 IPA 绝对锚点，游标由 {current_ipa_cursor} 瞬移至 -> {new_cursor} (锚点匹配度: {score:.2f})", req_id=req_id)
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

    def _global_teleport_anchor_reverse(self, ipa_asr, ipa_ref, current_ipa_cursor):
        """
        【绝对寻址·反向】对称于 _global_teleport_anchor。
        end 版在 ipa_ref[cursor:] 往后找锚点并返回锚点结束位置；
        reverse 版在 ipa_ref[:cursor] 往前找锚点并返回锚点起始位置。
        用于尾 chunk 全局自愈起点定位。
        """
        if len(ipa_asr) < 8:
            return None

        anchor_len = min(20, len(ipa_asr) // 2)
        start_idx = len(ipa_asr) // 4
        ipa_anchor = ipa_asr[start_idx : start_idx + anchor_len]

        search_range = ipa_ref[:current_ipa_cursor]
        if not search_range or len(search_range) < len(ipa_anchor):
            return None

        best_ratio = 0.0
        best_offset = -1
        window_size = int(len(ipa_anchor) * 1.3)

        for i in range(0, max(0, len(search_range) - window_size)):
            sub_ipa = search_range[i : i + window_size]
            ratio = Levenshtein.ratio(ipa_anchor, sub_ipa)
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = i

        GLOBAL_CONFIDENCE_THRESHOLD = 0.85
        if best_ratio >= GLOBAL_CONFIDENCE_THRESHOLD:
            # 返回锚点起始位置（end 版返回锚点结束位置，对称）
            new_ipa_cursor = best_offset
            return new_ipa_cursor, best_ratio

        return None

    def _macro_route_start_cursor(self, ipa_asr, ipa_ref, current_ipa_cursor, chunk_idx, chunk_info, retry_level=0):
        """
        【IPA 反向相对寻址】对称于 _macro_route_end_cursor。
        end 版从 cursor 往后找 ASR 文本末尾位置（用于首/中间 chunk 终点定位）；
        start 版从 cursor 往前找 ASR 文本开头位置（用于尾 chunk 起点定位）。
        Returns: (abs_ipa_start, ratio, None) | 彻底失败 (None, 0.0, None)
        """
        if not ipa_asr:
            return current_ipa_cursor, 1.0, None

        asr_len = len(ipa_asr)

        if retry_level == 0:
            search_end = current_ipa_cursor
            lookback_size = max(250, asr_len * 3)
            MATCH_THRESHOLD = 0.35
        elif retry_level == 1:
            search_end = current_ipa_cursor
            lookback_size = max(800, asr_len * 10)
            MATCH_THRESHOLD = 0.20
        elif retry_level == 2:
            search_end = min(len(ipa_ref), current_ipa_cursor + 60)
            lookback_size = max(500, asr_len * 5)
            MATCH_THRESHOLD = 0.20
        else:
            teleport_res = self._global_teleport_anchor_reverse(ipa_asr, ipa_ref, current_ipa_cursor)
            if teleport_res is not None:
                new_cursor, score = teleport_res
                log_warn(f"全局自愈成功(反向) | 物理切片 {chunk_idx + 1} 定位崩溃！已抓取核心 IPA 绝对锚点，游标由 {current_ipa_cursor} 瞬移至 -> {new_cursor} (锚点匹配度: {score:.2f})", req_id="Unknown")
                return new_cursor, score, None
            return None, 0.0, None

        search_start = max(0, search_end - lookback_size)
        search_window = ipa_ref[search_start : search_end]
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
            return self._macro_route_start_cursor(ipa_asr, ipa_ref, current_ipa_cursor, chunk_idx, chunk_info, retry_level + 1)

        # 对称 end 版去尾部 insert：start 版去头部 insert
        sub_match = search_window[best_match_start : best_match_start + window_size]
        opcodes = Levenshtein.opcodes(ipa_asr, sub_match)
        exact_match_start = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                exact_match_start = j2
            else:
                break

        best_match_start_idx = best_match_start + exact_match_start
        return search_start + best_match_start_idx, highest_ratio, None

    # 首尾 FA 验证的健康度阈值下限（阈值梯度递减到此为止）
    # 0.9：尾 chunk 音频若被截断，FA 对齐会明显崩（squash < 0.9），必须报错让用户检查音频
    ASR_FA_HEALTH_THRESHOLD = 0.90
    # 最终裁定阈值（原 FA_HEALTH_THRESHOLD，用于纠偏分支是否采信）
    FINAL_FA_HEALTH_THRESHOLD = 0.5

    def _evaluate_fa_health(self, word_alignments, chunk_audio_duration=None, diag_out=None, run_lang="zh"):
        """
        FA 对齐健康度评估（跨语言/数字/口胡/语速通用）。

        信号：
          - squash_score    : 主信号。挤碎词占比（duration < 20ms，FA 硬塞进静音）
          - stretch_score   : 主信号。拉伸词占比（duration > stretch_ms，FA 硬拉过非语音段）
          - variance_score  : 软辅助。词时长方差过低（线性插值痕迹）
          (coverage 砍掉：FA 时间戳预测可能偏长，last_end 超出 chunk_dur 是 FA 本身特性不是文本不足)
          (start 砍掉：VAD padding 让首词 start_time 正常就有 0.2-0.3s，误杀正常对齐)

        合成：hard = min(squash_score, stretch_score)
              soft_penalty = 0.7 + 0.3 * variance（variance 极低才温和惩罚）
              return hard * soft_penalty

        阈值按书写系统分两类（与 asyncProxyServer 统一，无需逐语言配置）：
          - 音节文字（中/日/韩）：FA 按字符对齐，每字 ≈ 1 音节。
            挤碎 < 20ms/字（物理发音下限），拉伸 > 350ms/字（正常 ~222ms/字 @ 4.5字/秒，
            350ms = 1.6x 已属异常；500ms 太松会漏掉 FA 拉伸静音段）。
          - 字母/辅音音素文字（英/俄/阿拉伯/孟加拉等）：FA 按词对齐，词含多音节。
            挤碎 < 20ms/词，拉伸 > 500ms/词（多音节词如 "international" 正常可达 ~1.2s，
            但 FA 对齐粒度是词级，500ms 是极慢朗读 ~2 syl/s 的保守下限）。

        物理依据（跨语言通用）：
          - 挤碎下限 20ms：人类最小发音动作（articulatory gesture：唇闭合、舌抬起）
            不可能快于 20ms，低于此值的对齐必然是 FA 把字硬塞进静音/间隙。
          - 拉伸上限：Pellegrino et al. 2011（跨 7 语言）音节速率 4.5-9.1 syl/s → 110-222ms/音节；
            Cruttenden 2014（英语）~4 syl/s → 250ms/音节；极慢朗读 ~2 syl/s = 500ms/音节。

        diag_out: 可选 list，传入则把诊断行 append 进去（供调用方聚合打印）
        """
        if not word_alignments: return 0.0

        is_syllabic = any(kw in (run_lang or "").lower() for kw in ["zh", "chinese", "cmn", "hans", "hant", "ja", "japanese", "ko", "korean"])
        stretch_threshold = 0.35 if is_syllabic else 0.5

        n = len(word_alignments)
        durations = [max(0.0, w.end_time - w.start_time) for w in word_alignments]

        # 1. squash_score：挤碎词占比（阈值门控，非渐进计分）
        # 挤碎判据：duration < 0.02s（20ms 物理发不完一个字/词，是 FA 硬塞的标志）
        # 门控逻辑：挤碎词占比 ≤ 60% 视为正常（英文 TTS 音频中 FA 天然产出大量 <20ms 词：
        # function words a/the/in/of/to 正常 10-30ms，FA 对齐粒度细时易压到 20ms 以下，
        # 实测正常英文 TTS 对齐 squash 占比 40-55%），通过即 1.0；
        # 超过 60% 才判定为文本远超音频容量，直接归零。
        squashed = sum(1 for d in durations if d < 0.02)
        squash_ratio = squashed / n
        squash_score = 1.0 if squash_ratio <= 0.60 else 0.0

        # 2. stretch_score：拉伸词占比（阈值门控，非渐进计分）
        # 拉伸判据：音节文字 > 350ms，字母文字 > 500ms
        # 门控逻辑：拉伸词占比 ≤ 40% 视为正常（FA 对 TTS 音频的固有拉伸），通过即 1.0；
        # 超过 40% 才判定为文本-音频严重不匹配，直接归零。
        stretched = sum(1 for d in durations if d > stretch_threshold)
        stretch_ratio = stretched / n
        stretch_score = 1.0 if stretch_ratio <= 0.40 else 0.0

        # 3. variance_score：词时长变异系数（CV = std/mean）
        # 只在 CV 极低（<0.2，线性插值痕迹）才惩罚；正常匀速朗读 CV 0.3-0.8 不扣
        mean_dur = sum(durations) / n
        if mean_dur > 1e-6:
            var = sum((d - mean_dur) ** 2 for d in durations) / n
            std = var ** 0.5
            cv = std / mean_dur
            if cv < 0.2:
                variance_score = 0.5 + cv * 2.5   # cv=0 -> 0.5, cv=0.2 -> 1.0
            else:
                variance_score = 1.0
        else:
            variance_score = 0.0

        hard = min(squash_score, stretch_score)
        soft_penalty = 0.7 + 0.3 * variance_score
        final = hard * soft_penalty

        # 诊断行收集（供调用方聚合打印，避免并发日志交织）
        if diag_out is not None and chunk_audio_duration and chunk_audio_duration > 0:
            last_end = word_alignments[-1].end_time
            diag_out.append(
                f"  [FA健康度] 对齐词数={n} | 音频时长={chunk_audio_duration:.2f}s | "
                f"末词结束={last_end:.2f}s | 挤碎词={squashed}个(占{squashed/n*100:.1f}%, <20ms被FA硬塞) | "
                f"拉伸词={stretched}个(占{stretched/n*100:.1f}%, >{int(stretch_threshold*1000)}ms被FA硬拉) | "
                f"挤碎分={squash_score:.3f} | 拉伸分={stretch_score:.3f} | 时长方差分={variance_score:.3f} | "
                f"综合得分={final:.3f}"
            )
        return final

    def _assert_fa_timestamps_in_chunk(self, word_alignments, chunk_start, chunk_end, req_id="Unknown"):
        """Reject invalid FA timestamps before they can be converted to absolute time."""
        try:
            overflow = validate_fa_timestamps_in_chunk(word_alignments, chunk_start, chunk_end)
        except FaTimestampError as err:
            log_error(f"❌ {err}", req_id=req_id)
            raise
        if overflow:
            log_warn(
                f"FA 时间戳小幅越界已放行 | 窗口={chunk_start:.3f}..{chunk_end:.3f}s | "
                f"越界={overflow:+.3f}s | 跨模型边界抖动，非伪对齐",
                req_id=req_id,
            )

    def _fa_search_bounds(self, is_head, chunk_audio, clean_reference_text, run_lang, chunk_dur, req_id="Unknown", diag_out=None, asr_char_count=None):
        """
        FA 搜边界纠偏（对称模型）：
          - 首 (is_head=True):  锚点 start=0 固定，扫 end 候选（0 -> text_len 方向）
          - 尾 (is_head=False): 锚点 end=text_len 固定，扫 start 候选（text_len -> 0 方向）
        用增强健康度选最优候选。

        候选长度锚点：优先按音频时长估字数（最稳，不受 ASR 幻觉影响），
        ASR 字数仅作辅助参考（ASR 可能幻觉/重复识别导致字数虚高）。
          - 中文 ~4.5 字/秒（270 字/分钟）
          - 英文 ~2.5 词/秒（150 词/分钟）× 5 字符/词 = 12.5 字符/秒
        diag_out: 可选 list，传入则把每个候选的 FA-health 诊断行 append 进去
        Returns: (orig_start, orig_end, fa_res, score) | 失败返回 (None, None, None, 0.0)
        """
        text_len = len(clean_reference_text)
        uses_character_tokens = self._uses_character_tokens(run_lang)

        # 按音频时长估锚点（主，稳）
        rate = 4.5 if uses_character_tokens else 12.5   # 字符/秒
        dur_anchor = int(chunk_dur * rate)
        # ASR 字数作辅助参考（若与时长估差异过大，取两者较小值，防 ASR 幻觉虚高）
        if asr_char_count and asr_char_count > 0:
            asr_anchor = int(asr_char_count * (1.0 if uses_character_tokens else 5.0))
            anchor = min(dur_anchor, asr_anchor) if asr_anchor > dur_anchor * 1.5 else dur_anchor
        else:
            anchor = dur_anchor
        diag_out.append(f"  [FA搜边界锚点] dur_anchor={dur_anchor} asr_anchor={asr_char_count} -> anchor={anchor}")

        # 围绕锚点 ± 范围搜长度
        candidate_lengths = sorted(set([
            max(30, int(anchor * 0.5)),
            int(anchor * 0.8),
            anchor,
            int(anchor * 1.2),
            int(anchor * 1.5),
        ]))

        diag_out.append(f"    [候选边界] 共{len(candidate_lengths)}个候选（锚点={anchor}字）:")

        best = (None, None, None, -1.0)   # 初始 -1，让 score=0 也能被选中
        best_idx = -1

        for ci, length in enumerate(candidate_lengths):
            if length <= 0: continue
            if is_head:
                # 锚点 start=0 固定，扫 end
                c_start, c_end = 0, min(length, text_len)
            else:
                # 锚点 end=text_len 固定，扫 start
                c_start, c_end = max(0, text_len - length), text_len
            cand_text = clean_reference_text[c_start:c_end]
            if not cand_text.strip():
                continue
            try:
                with self.fa_lock:
                    fa_res = self.fa_model.align(audio=[chunk_audio], text=[cand_text], language=run_lang)[0]
                self._assert_fa_timestamps_in_chunk(
                    fa_res, chunk_info['start'], chunk_info['end'], req_id,
                )
                score = self._evaluate_fa_health(fa_res, chunk_dur, diag_out=diag_out, run_lang=run_lang)
                if diag_out:
                    diag_out[-1] = f"    [候选{ci+1}/{len(candidate_lengths)}] 原文[{c_start}:{c_end}]={c_end-c_start}字 | " + diag_out[-1].lstrip()
                if score > best[3]:
                    best = (c_start, c_end, fa_res, score)
                    best_idx = ci + 1
            except FaTimestampError:
                raise
            except Exception:
                pass

        if best[3] > 0 and diag_out:
            diag_out.append(f"    [✅ 最优候选{best_idx}] 原文[{best[0]}:{best[1]}]={best[1]-best[0]}字 | score={best[3]:.3f}")
        return best

    def _align_head_tail_chunk(self, is_head, idx, chunk_info, asr_result,
                               batch_audios, ipa_ref, ipa_to_norm_map, ref_idx_map,
                               clean_reference_text, run_lang, req_id,
                               current_ipa_cursor, prev_orig_end):
        """
        首尾 chunk 对称路径：锚点固定、扫另一端、ASR寻路→FA验证→阈值梯度重试。
          - 首：锚点 start=0 固定，扫 end（沿向 text_len 方向）
          - 尾：锚点 end=text_len 固定，扫 start（沿向 0 方向）
        首尾的 FA 结果直接返回，不进末尾批量（性能：省末尾 batch 2 个 chunk）。

        尾 chunk 起点独立由反向 IPA 寻路定位，不再继承 prev_orig_end（中间 chunk 累积漂移
        会让尾 chunk 起点偏移到错误位置；尾 chunk 应像首 chunk 一样独立定位自己锚点
        对侧的边界）。

        阈值梯度：attempt 1 (ASR寻路) >= 0.95 / attempt 2 (FA搜边界) >= 0.92 / attempt 3 (FA搜边界) >= 0.90
        日志：诊断行收集到 diag_lines 返回，由调用方聚合打印（不在此方法内打印）
        报错：失败时附最佳候选 score + 音频检查提醒

        Returns: (orig_start, orig_end, ground_truth, fa_res, new_ipa_cursor, diag_lines)
        Raises:  AlignmentThresholdError (异常时 diag_lines 已填充，调用方可捕获后打印)
        """
        label = "首" if is_head else "尾"
        diag_lines = []
        diag_lines.append(f"━━━ {label}chunk[{idx}] 处理 | 音频{chunk_info['end']-chunk_info['start']:.2f}s ━━━")

        epi = self._get_epitran(run_lang)
        norm_asr = self._normalize_text(asr_result.text, run_lang)
        ipa_asr = epi.transliterate(norm_asr) if (epi is not None and norm_asr) else norm_asr

        text_len = len(clean_reference_text)
        chunk_dur = chunk_info['end'] - chunk_info['start']
        chunk_audio = batch_audios[idx]
        cursor = current_ipa_cursor

        # 追踪所有 attempt 的最佳候选（用于失败报错）
        best_attempt_score = -1.0
        best_attempt_gt = None
        best_attempt_fa = None

        # attempt 1: ASR/IPA 寻路（只跑一次，寻路结果固定）
        # 对称：首用 _macro_route_end_cursor 扫 end，尾用 _macro_route_start_cursor 扫 start
        if is_head:
            diag_lines.append(f"[attempt 1] ASR/IPA 寻路 (正向，扫 end)")
            abs_ipa_end, _, _ = self._macro_route_end_cursor(
                ipa_asr, ipa_ref, cursor, idx, chunk_info, retry_level=0
            )
            asr_orig_start = 0
            asr_orig_end = None
            if abs_ipa_end is not None:
                if abs_ipa_end < len(ipa_ref):
                    norm_pos = ipa_to_norm_map[abs_ipa_end]
                    asr_orig_end = ref_idx_map[norm_pos] if norm_pos < len(ref_idx_map) else text_len
                else:
                    asr_orig_end = text_len
            new_ipa_cursor_from_asr = abs_ipa_end
        else:
            # 尾 chunk：锚点 end=text_len 固定，反向寻路扫 start
            # search_cursor = len(ipa_ref)（文本末尾），从 IPA 末尾往前找 ASR 文本开头
            diag_lines.append(f"[attempt 1] ASR/IPA 寻路 (反向，扫 start)")
            abs_ipa_start, _, _ = self._macro_route_start_cursor(
                ipa_asr, ipa_ref, len(ipa_ref), idx, chunk_info, retry_level=0
            )
            asr_orig_end = text_len
            asr_orig_start = None
            if abs_ipa_start is not None:
                if abs_ipa_start < len(ipa_ref):
                    norm_pos = ipa_to_norm_map[abs_ipa_start]
                    asr_orig_start = ref_idx_map[norm_pos] if norm_pos < len(ref_idx_map) else 0
                else:
                    asr_orig_start = 0
            new_ipa_cursor_from_asr = abs_ipa_start

        if asr_orig_start is not None and asr_orig_end is not None and asr_orig_start < asr_orig_end:
            ground_truth = clean_reference_text[asr_orig_start:asr_orig_end]
            if ground_truth.strip():
                try:
                    with self.fa_lock:
                        fa_res = self.fa_model.align(audio=[chunk_audio], text=[ground_truth], language=run_lang)[0]
                    self._assert_fa_timestamps_in_chunk(
                        fa_res, chunk_info['start'], chunk_info['end'], req_id,
                    )
                    score = self._evaluate_fa_health(fa_res, chunk_dur, diag_out=diag_lines, run_lang=run_lang)
                    if score > best_attempt_score:
                        best_attempt_score, best_attempt_gt, best_attempt_fa = score, ground_truth, fa_res
                    if score >= 0.95:
                        diag_lines.append(f"✅ attempt 1 通过 | score={score:.3f} >= 0.95 -> 采信 ASR 寻路边界")
                        return asr_orig_start, asr_orig_end, ground_truth, fa_res, new_ipa_cursor_from_asr, diag_lines
                    diag_lines.append(f"⚠️ attempt 1 失败 | score={score:.3f} < 0.95 -> 转 FA 搜边界")
                except FaTimestampError:
                    raise
                except Exception as e:
                    diag_lines.append(f"⚠️ attempt 1 FA 异常 | {e} -> 转 FA 搜边界")
        else:
            diag_lines.append(f"⚠️ attempt 1 ASR 寻路无有效边界 -> 转 FA 搜边界")

        # attempt 2: FA 搜边界，阈值 0.92
        # attempt 3: FA 搜边界，阈值下限 0.90
        attempt_thresholds = {2: 0.92, 3: self.ASR_FA_HEALTH_THRESHOLD}
        asr_char_count = len(norm_asr) if norm_asr else None
        for attempt in (2, 3):
            diag_lines.append(f"[attempt {attempt}] FA 搜边界 (按音频时长估锚)")
            orig_start, orig_end, fa_res, score = self._fa_search_bounds(
                is_head, chunk_audio, clean_reference_text, run_lang, chunk_dur, req_id,
                diag_out=diag_lines, asr_char_count=asr_char_count
            )
            if orig_start is None or orig_end is None:
                diag_lines.append(f"⚠️ attempt {attempt} 无有效候选")
                continue

            ground_truth = clean_reference_text[orig_start:orig_end]
            if not ground_truth.strip():
                continue

            if score > best_attempt_score:
                best_attempt_score, best_attempt_gt, best_attempt_fa = score, ground_truth, fa_res

            effective_threshold = attempt_thresholds[attempt]
            if score >= effective_threshold:
                # 推算 ipa cursor（搜边界没用 IPA，从 orig_start/orig_end 反推）
                if is_head:
                    new_cursor = cursor
                    for n_idx in range(cursor, len(ipa_ref)):
                        norm_pos = ipa_to_norm_map[n_idx]
                        if norm_pos < len(ref_idx_map) and ref_idx_map[norm_pos] >= orig_end:
                            new_cursor = n_idx
                            break
                    else:
                        new_cursor = len(ipa_ref)
                else:
                    # 尾 chunk：cursor 应指 orig_start 对应的 IPA 位置（往前的位置）
                    new_cursor = 0
                    for n_idx in range(len(ipa_ref) - 1, -1, -1):
                        norm_pos = ipa_to_norm_map[n_idx]
                        if norm_pos < len(ref_idx_map) and ref_idx_map[norm_pos] <= orig_start:
                            new_cursor = n_idx
                            break
                diag_lines.append(f"✅ attempt {attempt} 通过 | score={score:.3f} >= {effective_threshold:.2f} -> 采信 FA 搜边界")
                return orig_start, orig_end, ground_truth, fa_res, new_cursor, diag_lines
            diag_lines.append(f"⚠️ attempt {attempt} 失败 | 最佳 score={score:.3f} < {effective_threshold:.2f}")

        # 全部失败：构建报错（音频检查提醒）
        # 注意：FA 是强制对齐器，fa_text 永远回声输入文本，"原文 vs FA识别"字数恒等，
        # 匹配/未匹配对比无意义，不打印。
        diag_lines.append(f"🔴 {label}chunk 彻底失败 | 3 次尝试均未通过阈值（下限 {self.ASR_FA_HEALTH_THRESHOLD}）")

        error_msg = (
            f"[🔴 {label}chunk 对齐失败] idx={idx} | 音频{chunk_dur:.1f}s\n"
            f"  ASR 寻路 + FA 搜边界均未通过健康度阈值（下限 {self.ASR_FA_HEALTH_THRESHOLD}）\n"
            f"  最佳候选 score={best_attempt_score:.3f}"
            f"  ⚠️ 请检查音频是否完整：{label}部音频可能被截断或生成出错，导致 FA 无法对齐"
        )
        raise AlignmentThresholdError(error_msg, diag_lines)

    def _rebuild_words_from_ground_truth(self, ground_truth, fa_words, lang="zh"):
        """
        用 ground_truth（原文切片，含标点/换行）重建 word 列表，
        使返回的 word 文本与原文同源，时间戳取自 FA 对齐结果。

        FA 返回的 fa_words 只含有效内容（标点无对齐时间戳），
        按顺序与 ground_truth 的有效 token 一一对应：
          - 中文：有效 token = 单个有效字符（汉字/字母/数字）
          - 英文：有效 token = 连续有效字母数字组成的词
        非有效 token（标点/换行/空白）继承前一个有效 token 的 end_time
        （开头无前驱时 start=end=0）。
        这样 word 文本拼接严格等于 ground_truth，与 originalText 同源。
        """
        uses_character_tokens = self._uses_character_tokens(lang)
        if uses_character_tokens:
            # CJK languages without reliable whitespace tokenization: one
            # Unicode letter/number per aligned token.
            tokens = list(ground_truth)
            def is_valid_token(tok):
                return bool(tok) and self._is_text_character(tok)
        else:
            # Whitespace-delimited languages, including Cyrillic and Hangul.
            tokens = re.findall(r'\w+|[^\w]', ground_truth, re.UNICODE)
            def is_valid_token(tok):
                return bool(tok) and self._is_text_character(tok[0])

        rebuilt = []
        fa_idx = 0
        prev_end = 0.0
        for tok in tokens:
            if is_valid_token(tok) and fa_idx < len(fa_words):
                w = fa_words[fa_idx]
                fa_idx += 1
                start = w.start_time
                end = w.end_time
                prev_end = end
            else:
                start = prev_end
                end = prev_end
            rebuilt.append({"text": tok, "start": start, "end": end})

        return rebuilt

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

    def _validate_subtitle_timestamps(self, words, req_id="Unknown"):
        """校验字幕时间戳无时间倒流。

        每个词条必须满足 start <= end，且 word 流的 start/end 都不能比
        前一个词回退。后者不能只靠单词自身校验发现：例如前词
        [180, 185]、后词 [170, 178] 都是合法区间，但字幕 cue 取首词
        start 和末词 end 后会变成负时长。
        """
        try:
            validate_absolute_timestamps(words)
        except FaTimestampError as err:
            log_error(f"❌ {err}", req_id=req_id)
            raise
        log_info(f"✅ 字幕时间戳校验通过 | {len(words)} 个词条时间单调且满足 start <= end", req_id=req_id)

    def process_audio(self, audio_path, reference_text, language_code=None, req_id="Unknown"):
        """Align reference text using the ASR boundary route followed by FA."""
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        
        try:
            # 记录最原始未经任何清理的 Prompt 字符数
            raw_prompt_length = len(reference_text)
            
            # 保留换行符（下游 convertToLineTimestamps 按 \n 切行），
            # 仅折叠除 \n 外的其他空白为单空格、去除首尾空白。
            # 归一化（剥标点/空白）只在 _build_index_map 内部用于 IPA 对齐，
            # 不污染 ground_truth 与最终返回的 wordTimestamps 文本。
            clean_reference_text = re.sub(r'[ \t\r\f\v]+', ' ', reference_text).strip()
            
            # 1. 前置一键全量编译为 IPA，并产出对应的映射数组 (若不支持语言，此处将直接被 Fail-Fast 阻断)
            ipa_ref, norm_ref, ipa_to_norm_map, ref_idx_map = self._build_index_map(clean_reference_text, run_lang)

            try: audio_data, sample_rate = sf.read(audio_path)
            except Exception as err:
                log_error(f"阶段: 音频读取失败 | 详情: {err}", req_id=req_id)
                raise
                
            wav = torch.from_numpy(audio_data).float()
            if wav.dim() > 1: wav = wav.mean(dim=1)
            if sample_rate != 16000:
                import torchaudio.functional as F
                wav = F.resample(wav, sample_rate, 16000)
                sample_rate = 16000

            total_audio_duration = len(wav) / sample_rate

            # ── 全局音节速率预检（fail-fast，不浪费下游 VAD/ASR/FA 计算）──
            # 物理依据：Pellegrino et al. (2011) 跨 7 语言（中/英/日/西/法/德/意）研究表明，
            # 所有语言的音节速率落在 4.5-9.1 syll/s 区间——这是人类发声器官的生物力学常数，
            # 与具体语言无关。取该区间最慢端（4.5）的一半 ≈ 2.0 syll/s 作极端保守下限：
            # 任何正常语音（含慢读、停顿、思考）都不可能整体低于 2 syll/s。
            # 低于此值说明音频时长与文本严重不匹配——音频含大量非语音内容（音乐/音效/长静音），
            # 或文本残缺。此时强行对齐只会产出拉伸词（FA 把少量字硬拉过非语音段），结果不可信。
            #
            # 音节数估算：中文 1 字 = 1 音节（最精确）；其他语言用 "词数 × 1.5" 粗估
            # （英语平均 1.5 音节/词，其他印欧语言类似）。阈值 2.0 极保守，估算误差不会误杀。
            uses_character_tokens = self._uses_character_tokens(run_lang)
            if uses_character_tokens:
                est_syllables = len(clean_reference_text)
            else:
                est_syllables = len(clean_reference_text.split()) * 1.5
            global_syll_rate = est_syllables / total_audio_duration if total_audio_duration > 0 else 0
            if global_syll_rate < 2.0:
                raise ValueError(
                    f"音频时长与文本严重不匹配：估算音节速率 {global_syll_rate:.2f} syll/s "
                    f"（文本≈{est_syllables:.0f}音节 / 音频{total_audio_duration:.1f}s），"
                    f"远低于正常语音下限 2.0 syll/s。音频可能含大量非语音内容（音乐/音效/长静音）或文本残缺，"
                    f"强制对齐结果不可信。"
                )

            try:
                with self.vad_lock:
                    # min_silence_duration_ms=300：静音 ≥300ms 才切分（默认 100ms 会把句中短停顿也切碎，
                    # 导致细分片过度碎片化，下游合并边界噪声大；200ms 只在真停顿切，边界更可靠）
                    raw_speech_timestamps = self.get_speech_timestamps(
                        wav, self.vad_model, return_seconds=True,
                        min_silence_duration_ms=300,
                    )
                if not raw_speech_timestamps:
                    raise ValueError("VAD 未检测到任何有效人声。")
            except Exception as err:
                log_error(f"阶段: VAD 物理切片失败 | 详情: {err}", req_id=req_id)
                raise

            # ASR 硬匹配合并：从 VAD 细分片出发，沿向量方向贪婪合并，
            # 每合并一块跑 ASR，ASR 末尾/开头词在原文里唯一匹配则定边界（硬锚点）。
            # 首/中间 chunk 前向（0→text_len），尾 chunk 反向（text_len→0）。
            # 失败 fallback 到时长切片 + 下游 IPA 兜底。
            physical_chunks, asr_results, orig_boundaries, hardmatch_fallback = self._group_vad_by_asr_hardmatch(
                raw_speech_timestamps, total_audio_duration, wav, sample_rate,
                clean_reference_text, norm_ref, ref_idx_map, run_lang, req_id=req_id
            )

            batch_audios = []
            for chunk_info in physical_chunks:
                start_sample = max(0, int(chunk_info['start'] * sample_rate))
                end_sample = min(len(wav), int(chunk_info['end'] * sample_rate))
                chunk_wav = wav[start_sample:end_sample]
                batch_audios.append((chunk_wav.numpy(), sample_rate))

            # fallback 路径：硬匹配预推理失败，asr_results 为空，需按 chunk 重跑 ASR
            if hardmatch_fallback and not asr_results:
                log_warn("⚠️ ASR 硬匹配预推理失败，fallback 到时长切片 + 按 chunk 重跑 ASR", req_id=req_id)
                try:
                    asr_results = self.asr_scheduler.submit(audios=batch_audios, language=run_lang, req_id=req_id)
                except Exception as err:
                    log_error(f"阶段: ASR 批量模型推理崩溃（fallback） | 详情: {err}", req_id=req_id)
                    raise

            # ── 请求级聚合日志：音频概况 + ASR 识别概况 ──
            task_info = get_task_info(req_id)
            audio_name = task_info.get("file", "") if task_info else ""
            if not audio_name:
                audio_name = os.path.basename(audio_path) if audio_path else "(未知)"
            req_diag = [
                f"━━━━━ 请求处理开始 | {audio_name} ━━━━━",
                f"音频总时长 : {total_audio_duration:.2f}s",
                f"原文长度   : raw={raw_prompt_length}字 clean={len(clean_reference_text)}字 norm={len(norm_ref)}字",
                f"VAD 细分片 : {len(raw_speech_timestamps)} 段（合并前）→ 合并后 {len(physical_chunks)} 段 chunk",
                f"语言       : {run_lang}",
                f"ASR 识别概况（每 chunk: 时长s / ASR字数 / ASR文本前30字 / 对应原文片段）:",
            ]
            for i, (ci, res) in enumerate(zip(physical_chunks, asr_results)):
                ci_dur = ci['end'] - ci['start']
                asr_txt = self._normalize_text(res.text, run_lang) if res.text else ""
                tag = ""
                if i == 0: tag = " [首]"
                elif i == len(physical_chunks) - 1: tag = " [尾]"
                ob = orig_boundaries[i] if i < len(orig_boundaries) else (None, None)
                ref_part = ""
                ob_start, ob_end = ob
                if ob_start is not None and ob_end is not None:
                    ref_snippet = clean_reference_text[ob_start:ob_end][:30].replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    ref_part = f" | 原文[{ob_start}:{ob_end}]{ob_end-ob_start}字: {ref_snippet}..."
                elif ob_start is not None:
                    est_end = min(ob_start + 200, len(clean_reference_text))
                    ref_snippet = clean_reference_text[ob_start:est_end][:30].replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    ref_part = f" | 原文[{ob_start}:?]≥{est_end-ob_start}字: {ref_snippet}..."
                elif ob_end is not None:
                    est_start = max(0, ob_end - 200)
                    ref_snippet = clean_reference_text[est_start:ob_end][:30].replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    ref_part = f" | 原文[?:{ob_end}]≤{ob_end-est_start}字: {ref_snippet}..."
                else:
                    ref_part = " | 原文[未定位]"
                req_diag.append(
                    f"  chunk{i}{tag} | {ci['start']:.1f}-{ci['end']:.1f}s ({ci_dur:.1f}s) | "
                    f"asr={len(asr_txt)}字 | {asr_txt[:30]}...{ref_part}"
                )
            log_block(req_diag, req_id=req_id, level="INFO")

            fa_batch_audios = []
            fa_batch_texts = []
            # Keep the integer chunk index with each FA request. Recovering it
            # later from a floating-point start timestamp is unsafe.
            chunk_mapping_meta = []
            # 首尾特例路径产出的 FA 结果：idx -> (chunk_info, ground_truth, fa_res)
            # 末尾批量 FA 时跳过这些 idx，直接复用
            head_tail_fa_results = {}
            # 首尾 chunk 诊断行收集：idx -> diag_lines，末尾统一打印
            head_tail_diag = {}

            current_ipa_cursor = 0
            prev_orig_end = 0
            total_chunks = len(physical_chunks)

            for chunk_idx in range(1, total_chunks):
                previous_chunk = physical_chunks[chunk_idx - 1]
                current_chunk = physical_chunks[chunk_idx]
                if current_chunk['start'] < previous_chunk['start']:
                    raise ValueError(
                        f"chunk 起点回退: chunk[{chunk_idx - 1}].start={previous_chunk['start']}, "
                        f"chunk[{chunk_idx}].start={current_chunk['start']}"
                    )

            epi = self._get_epitran(run_lang)

            try:
                # 首尾优先：先处理首尾 chunk，失败则立即中止（中间 chunk 无需浪费计算）
                processing_order = [0] + ([total_chunks - 1] if total_chunks > 1 else []) + list(range(1, total_chunks - 1))
                head_cursor_state = None

                for idx in processing_order:
                    chunk_info = physical_chunks[idx]
                    res = asr_results[idx]
                    is_head = (idx == 0)
                    is_tail = (idx == total_chunks - 1)
                    hard_start, hard_end = orig_boundaries[idx] if idx < len(orig_boundaries) else (None, None)

                    if idx == 1 and head_cursor_state is not None:
                        current_ipa_cursor, prev_orig_end = head_cursor_state

                    # ── 硬匹配命中：FA 验证健康度，通过则采信，不通过则 fallback 寻路 ──
                    if hard_start is not None and hard_end is not None:
                        ground_truth = clean_reference_text[hard_start:hard_end]
                        if ground_truth.strip():
                            chunk_dur = chunk_info['end'] - chunk_info['start']
                            with self.fa_lock:
                                fa_res = self.fa_model.align(audio=[batch_audios[idx]], text=[ground_truth], language=run_lang)[0]
                            self._assert_fa_timestamps_in_chunk(
                                fa_res, chunk_info['start'], chunk_info['end'], req_id,
                            )
                            fa_score = self._evaluate_fa_health(fa_res, chunk_dur, diag_out=None, run_lang=run_lang)
                            label = "首" if is_head else ("尾" if is_tail else f"中间[{idx}]")

                            if fa_score >= self.ASR_FA_HEALTH_THRESHOLD:
                                hard_diag = [
                                    f"━━━ {label}chunk[{idx}] 处理 | 音频{chunk_dur:.2f}s ━━━ | "
                                    f"✅ 硬匹配锚点 | 原文[{hard_start}:{hard_end}] | 原文{hard_end-hard_start}字 | 跳过寻路直接 FA | "
                                    f"[FA-health] score={fa_score:.3f} >= {self.ASR_FA_HEALTH_THRESHOLD} (硬匹配采信)",
                                ]
                                if is_head or is_tail:
                                    head_tail_fa_results[idx] = (chunk_info, ground_truth, fa_res)
                                    head_tail_diag[idx] = hard_diag
                                else:
                                    fa_batch_audios.append(batch_audios[idx])
                                    fa_batch_texts.append(ground_truth)
                                    chunk_mapping_meta.append((idx, chunk_info))
                                    log_block(hard_diag, req_id=req_id, level="INFO")
                                if is_head:
                                    new_cursor = 0
                                    for n_idx in range(len(ipa_ref) - 1, -1, -1):
                                        norm_pos = ipa_to_norm_map[n_idx]
                                        if norm_pos < len(ref_idx_map) and ref_idx_map[norm_pos] <= hard_end:
                                            new_cursor = n_idx
                                            break
                                    current_ipa_cursor = new_cursor
                                    prev_orig_end = hard_end
                                    head_cursor_state = (current_ipa_cursor, prev_orig_end)
                                elif is_tail:
                                    prev_orig_end = hard_end
                                else:
                                    new_cursor = current_ipa_cursor
                                    for n_idx in range(current_ipa_cursor, len(ipa_ref)):
                                        norm_pos = ipa_to_norm_map[n_idx]
                                        if norm_pos < len(ref_idx_map) and ref_idx_map[norm_pos] >= hard_end:
                                            new_cursor = n_idx
                                            break
                                    else:
                                        new_cursor = len(ipa_ref)
                                    current_ipa_cursor = new_cursor
                                    prev_orig_end = hard_end
                                continue
                            else:
                                log_warn(
                                    f"⚠️ {label}chunk[{idx}] 硬匹配 FA 健康度不达标 | "
                                    f"score={fa_score:.3f} < {self.ASR_FA_HEALTH_THRESHOLD} | "
                                    f"原文[{hard_start}:{hard_end}] | 放弃硬匹配，fallback 寻路",
                                    req_id=req_id,
                                )

                    # ── 硬匹配未命中：走原寻路逻辑 ──
                    if is_head or is_tail:
                        orig_start, orig_end, ground_truth, fa_res, new_cursor, diag_lines = self._align_head_tail_chunk(
                            is_head=is_head, idx=idx, chunk_info=chunk_info, asr_result=res,
                            batch_audios=batch_audios, ipa_ref=ipa_ref, ipa_to_norm_map=ipa_to_norm_map,
                            ref_idx_map=ref_idx_map, clean_reference_text=clean_reference_text,
                            run_lang=run_lang, req_id=req_id,
                            current_ipa_cursor=current_ipa_cursor, prev_orig_end=prev_orig_end
                        )
                        head_tail_fa_results[idx] = (chunk_info, ground_truth, fa_res)
                        head_tail_diag[idx] = diag_lines
                        if is_head:
                            current_ipa_cursor = new_cursor
                            prev_orig_end = orig_end
                            head_cursor_state = (current_ipa_cursor, prev_orig_end)
                        else:
                            prev_orig_end = orig_end
                        continue

                    # 中间 chunk 走原路径
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
                                with self.fa_lock:
                                    fa_res = self.fa_model.align(audio=[current_chunk_audio], text=[candidate_text], language=run_lang)[0]
                                self._assert_fa_timestamps_in_chunk(
                                    fa_res, chunk_info['start'], chunk_info['end'], req_id,
                                )
                                score = self._evaluate_fa_health(fa_res, run_lang=run_lang)
                                if score > best_fa_score:
                                    best_fa_score = score
                                    best_candidate_cursor = orig_start + len(candidate_text)
                            except FaTimestampError:
                                raise
                            except Exception:
                                pass

                        if best_fa_score > self.FINAL_FA_HEALTH_THRESHOLD and best_candidate_cursor is not None:
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
                    chunk_mapping_meta.append((idx, chunk_info))

            except AlignmentThresholdError as threshold_err:
                # 首尾对齐失败：把诊断行攒进 head_tail_diag，末尾统一打印
                if threshold_err.diag_lines:
                    # 从异常消息反推失败的 idx（首尾特例抛的才带 diag_lines）
                    # 直接附加到一个特殊 key，末尾统一打印
                    head_tail_diag['FAILED'] = threshold_err.diag_lines
                # 先打印所有攒着的首尾诊断（成功 + 失败），再重新抛出
                if head_tail_diag:
                    all_diag = ["━━━━━ 首尾 chunk 处理诊断汇总 ━━━━━"]
                    for k in sorted(head_tail_diag.keys(), key=lambda x: (x != 'FAILED', x)):
                        all_diag.extend(head_tail_diag[k])
                        all_diag.append("")
                    log_block(all_diag, req_id=req_id, level="WARN")
                raise threshold_err
            except Exception as err:
                log_error(f"阶段: 文本对齐路由匹配失败 | 详情: {err}", req_id=req_id)
                raise

            verbose_json = {"task": "transcribe", "language": run_lang, "duration": round(total_audio_duration, 2), "report": {}, "text": "", "words": []}

            # 合并首尾特例的 FA 结果 + 中间 chunk 的批量 FA 结果，按 idx 顺序输出（时间序）
            uses_character_tokens = self._uses_character_tokens(run_lang)
            fa_text_parts = []

            def _emit_fa(chunk_info, ground_truth, single_fa_res):
                chunk_start = chunk_info['start']
                self._assert_fa_timestamps_in_chunk(
                    single_fa_res, chunk_start, chunk_info['end'], req_id,
                )
                rebuilt_words = self._rebuild_words_from_ground_truth(ground_truth, single_fa_res, run_lang)
                for w in rebuilt_words:
                    verbose_json["words"].append({
                        "word": w["text"],
                        "start": round(chunk_start + w["start"], 3),
                        "end": round(chunk_start + w["end"], 3)
                    })
                for word_align in single_fa_res:
                    fa_text_parts.append(word_align.text)

            # 1. 收集每个 idx 的 (chunk_info, ground_truth, fa_res)，按 idx 排序后统一输出
            final_chunks = {}  # idx -> (chunk_info, ground_truth, fa_res)
            for idx, payload in head_tail_fa_results.items():
                final_chunks[idx] = payload

            # 2. 中间 chunk 跑批量 FA（首尾 idx 不在 batch 里）。
            # qwen_asr 按输入顺序返回结果，直接使用提交时绑定的 idx 回填。
            if fa_batch_audios:
                try:
                    with self.fa_lock:
                        fa_results = self.fa_model.align(audio=fa_batch_audios, text=fa_batch_texts, language=run_lang)
                except Exception as err:
                    log_error(f"阶段: FA ForcedAligner 批量推理对齐失败 | 详情: {err}", req_id=req_id)
                    raise

                if len(fa_results) != len(chunk_mapping_meta):
                    raise ValueError(
                        f"FA batch 结果数量不匹配: requested={len(chunk_mapping_meta)}, "
                        f"returned={len(fa_results)}"
                    )

                for (idx, chunk_info), ground_truth, single_fa_res in zip(chunk_mapping_meta, fa_batch_texts, fa_results):
                    self._assert_fa_timestamps_in_chunk(
                        single_fa_res, chunk_info['start'], chunk_info['end'], req_id,
                    )
                    final_chunks[idx] = (chunk_info, ground_truth, single_fa_res)

            # 3. 按 idx 时间序统一输出 + 收集全部 word_alignments 算全文 FA 健康度
            all_word_alignments = []
            fa_emit_diag = []
            missing_chunks = [idx for idx in range(total_chunks) if idx not in final_chunks]
            if missing_chunks:
                raise ValueError(f"FA 结果缺少 chunk: {missing_chunks}")
            for idx in sorted(final_chunks.keys()):
                chunk_info, ground_truth, single_fa_res = final_chunks[idx]
                _emit_fa(chunk_info, ground_truth, single_fa_res)
                all_word_alignments.extend(single_fa_res)
                if single_fa_res:
                    chunk_duration = chunk_info['end'] - chunk_info['start']
                    relative_start = single_fa_res[0].start_time
                    relative_end = single_fa_res[-1].end_time
                    start_overflow = max(0.0, -relative_start)
                    end_overflow = max(0.0, relative_end - chunk_duration)
                    mapping_status = "正常"
                    if start_overflow > 0.05 or end_overflow > 0.05:
                        mapping_status = (
                            f"⚠️ FA相对时间越界"
                            f"(start超界{start_overflow:.3f}s,end超界{end_overflow:.3f}s)"
                        )
                        log_error(
                            f"❌ FA chunk 时间映射越界 | chunk[{idx}] "
                            f"音频窗口={chunk_duration:.3f}s | "
                            f"FA相对={relative_start:.3f}..{relative_end:.3f}s | "
                            f"FA绝对={chunk_info['start'] + relative_start:.3f}.."
                            f"{chunk_info['start'] + relative_end:.3f}s | "
                            f"这会与相邻 chunk 重叠并导致后续词时间戳非单调",
                            req_id=req_id,
                        )
                    fa_emit_diag.append(
                        f"chunk[{idx}] start={chunk_info['start']:.3f}s end={chunk_info['end']:.3f}s "
                        f"窗口={chunk_duration:.3f}s "
                        f"FA相对={relative_start:.3f}..{relative_end:.3f}s "
                        f"FA绝对={chunk_info['start'] + single_fa_res[0].start_time:.3f}.."
                        f"{chunk_info['start'] + relative_end:.3f}s "
                        f"{mapping_status} "
                        f"words={len(single_fa_res)}"
                    )

            if fa_emit_diag:
                log_block(["━━━━━ FA chunk 时间映射诊断 ━━━━━"] + fa_emit_diag, req_id=req_id)

            verbose_json["text"] = "".join(fa_text_parts) if uses_character_tokens else " ".join(fa_text_parts)

            self._validate_subtitle_timestamps(verbose_json["words"], req_id=req_id)

            verbose_json["report"] = self._generate_diff_report(clean_reference_text, verbose_json["text"], run_lang)
            total_pipeline_time = time.time() - pipeline_start_time
            
            accuracy_str = verbose_json["report"].get("accuracy_ratio", "0.00%")
            accuracy_val = float(accuracy_str.strip('%'))
            align_status = "✅ 成功" if accuracy_val > 50 else "⚠️ 相似度低"
            ref_len = verbose_json["report"].get("ref_length", 0)
            res_len = verbose_json["report"].get("result_length", 0)

            global_fa_diag = []
            global_fa_score = self._evaluate_fa_health(all_word_alignments, total_audio_duration, diag_out=global_fa_diag, run_lang=run_lang)

            task_info = get_task_info(req_id)
            file_name = task_info.get("file", "(未知)") if task_info else "(未知)"

            # ── 首尾 chunk 诊断汇总（成功路径：所有 diag_lines 统一打印）──
            if head_tail_diag:
                all_diag = ["━━━━━ 首尾 chunk 处理诊断汇总 ━━━━━"]
                for idx in sorted(head_tail_diag.keys()):
                    if idx == 'FAILED': continue
                    all_diag.extend(head_tail_diag[idx])
                    all_diag.append("")
                # ── 首尾特例结果汇总 ──
                if head_tail_fa_results:
                    all_diag.append("━━━ 首尾对齐结果汇总 ━━━")
                    is_syllabic_lang = any(kw in (run_lang or "").lower() for kw in ["zh", "chinese", "cmn", "hans", "hant", "ja", "japanese", "ko", "korean"])
                    for idx in sorted(head_tail_fa_results.keys()):
                        chunk_info, ground_truth, fa_res = head_tail_fa_results[idx]
                        tag = "首" if idx == 0 else "尾"
                        ci_dur = chunk_info['end'] - chunk_info['start']
                        n_words = len(fa_res) if fa_res else 0
                        if is_syllabic_lang:
                            text_desc = f"原文{len(ground_truth)}字"
                        else:
                            text_desc = f"原文{len(ground_truth.split())}词({len(ground_truth)}字符)"
                        all_diag.append(
                            f"  {tag}chunk[{idx}] | 音频{ci_dur:.1f}s | "
                            f"{text_desc} | FA对齐{n_words}词"
                        )
                log_block(all_diag, req_id=req_id, level="INFO")

            log_block([
                f"📊 最终报告 (对齐模式)",
                f"文件名       : {file_name}",
                f"执行状态     : {align_status} (相似度: {accuracy_str} | FA-health: {global_fa_score:.3f})",
                f"物理切片     : {len(physical_chunks)} 段 -> 有效对齐 {len(fa_batch_audios) + len(head_tail_fa_results)} 段",
                f"字符匹配     : 原文={raw_prompt_length}字 | 归一化={ref_len}字 | 对齐={res_len}字",
                f"音频时长     : {total_audio_duration:.2f}s",
                f"流水线耗时   : {total_pipeline_time:.2f}s",
                f"RTF          : {total_pipeline_time / total_audio_duration:.3f}",
            ] + global_fa_diag, req_id=req_id)
            
            return verbose_json

        except Exception as outer_err:
            raise outer_err

    def process_audio_fa_only(self, audio_path, reference_text, language_code=None, req_id="Unknown"):
        """Align reference text with FA only; this path never invokes the ASR scheduler."""
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]
        clean_reference_text = re.sub(r'[ \t\r\f\v]+', ' ', reference_text).strip()
        if not clean_reference_text:
            raise ValueError("纯 ForcedAligner 模式要求 prompt 不能为空。")

        try:
            audio_data, sample_rate = sf.read(audio_path)
        except Exception as err:
            log_error(f"阶段: 音频读取失败 | 详情: {err}", req_id=req_id)
            raise

        wav = torch.from_numpy(audio_data).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sample_rate != 16000:
            import torchaudio.functional as F
            wav = F.resample(wav, sample_rate, 16000)
            sample_rate = 16000

        total_audio_duration = len(wav) / sample_rate
        if total_audio_duration <= 0:
            raise ValueError("音频为空，无法进行 ForcedAligner 对齐。")

        try:
            with self.vad_lock:
                raw_speech_timestamps = self.get_speech_timestamps(
                    wav,
                    self.vad_model,
                    return_seconds=True,
                    min_silence_duration_ms=300,
                )
            if not raw_speech_timestamps:
                raise ValueError("VAD 未检测到任何有效人声。")
        except Exception as err:
            log_error(f"阶段: VAD 物理切片失败 | 详情: {err}", req_id=req_id)
            raise

        physical_chunks = self._group_vad_timestamps(
            raw_speech_timestamps,
            total_audio_duration,
            max_sec=self.config.get("chunk_max_sec", 60),
        )
        if not physical_chunks:
            raise ValueError("VAD 未生成可用的 ForcedAligner 音频分块。")

        uses_character_tokens = self._uses_character_tokens(run_lang)
        if uses_character_tokens:
            token_end_indices = [
                index + 1
                for index, char in enumerate(clean_reference_text)
                if self._is_text_character(char)
            ]
        else:
            token_end_indices = [match.end() for match in re.finditer(r'\w+', clean_reference_text, re.UNICODE)]

        if not token_end_indices:
            raise ValueError("prompt 不包含可供 ForcedAligner 对齐的文本。")
        if len(physical_chunks) > len(token_end_indices):
            raise ValueError(
                f"VAD 分块数 ({len(physical_chunks)}) 超过可对齐文本单元数 ({len(token_end_indices)})，"
                "无法在不使用 ASR 的情况下安全推断文本边界。"
            )

        def speech_duration(chunk):
            return sum(
                max(0.0, min(chunk["end"], ts["end"]) - max(chunk["start"], ts["start"]))
                for ts in raw_speech_timestamps
            )

        chunk_speech_durations = [speech_duration(chunk) for chunk in physical_chunks]
        total_speech_duration = sum(chunk_speech_durations)
        if total_speech_duration <= 0:
            raise ValueError("VAD 语音时长为零，无法推断 ForcedAligner 文本边界。")

        aligned_chunks = []
        cursor_token = 0
        cumulative_speech_duration = 0.0
        total_tokens = len(token_end_indices)

        for chunk_idx, chunk_info in enumerate(physical_chunks):
            cumulative_speech_duration += chunk_speech_durations[chunk_idx]
            remaining_chunks = len(physical_chunks) - chunk_idx - 1
            min_token_end = cursor_token + 1
            max_token_end = total_tokens - remaining_chunks

            if remaining_chunks == 0:
                candidate_token_counts = [total_tokens]
            else:
                target_token_count = round(
                    total_tokens * cumulative_speech_duration / total_speech_duration
                )
                search_radius = max(3, int(total_tokens * 0.12))
                candidate_token_counts = sorted({
                    max(min_token_end, min(max_token_end, target_token_count + offset))
                    for offset in (-search_radius, -search_radius // 2, 0, search_radius // 2, search_radius)
                })

            start_sample = max(0, int(chunk_info["start"] * sample_rate))
            end_sample = min(len(wav), int(chunk_info["end"] * sample_rate))
            chunk_audio = (wav[start_sample:end_sample].numpy(), sample_rate)
            chunk_duration = chunk_info["end"] - chunk_info["start"]
            best = None
            diagnostics = []

            for candidate_token_count in candidate_token_counts:
                text_start = 0 if cursor_token == 0 else token_end_indices[cursor_token - 1]
                text_end = token_end_indices[candidate_token_count - 1]
                ground_truth = clean_reference_text[text_start:text_end]
                if not ground_truth.strip():
                    continue
                try:
                    with self.fa_lock:
                        fa_res = self.fa_model.align(
                            audio=[chunk_audio],
                            text=[ground_truth],
                            language=run_lang,
                        )[0]
                    self._assert_fa_timestamps_in_chunk(
                        fa_res,
                        chunk_info["start"],
                        chunk_info["end"],
                        req_id,
                    )
                    score = self._evaluate_fa_health(
                        fa_res,
                        chunk_duration,
                        run_lang=run_lang,
                    )
                    diagnostics.append(
                        f"候选文本单元={candidate_token_count} | FA-health={score:.3f}"
                    )
                    if best is None or score > best[0]:
                        best = (score, candidate_token_count, ground_truth, fa_res)
                except FaTimestampError:
                    raise
                except Exception as err:
                    diagnostics.append(
                        f"候选文本单元={candidate_token_count} | FA异常={err}"
                    )

            if best is None or best[0] < self.FINAL_FA_HEALTH_THRESHOLD:
                details = "；".join(diagnostics) if diagnostics else "无有效候选"
                raise AlignmentThresholdError(
                    f"[纯 FA 对齐失败] chunk[{chunk_idx}] | 音频 {chunk_duration:.1f}s | {details}",
                    diagnostics,
                )

            score, selected_token_count, ground_truth, fa_res = best
            aligned_chunks.append((chunk_info, ground_truth, fa_res))
            cursor_token = selected_token_count
            log_info(
                f"纯 FA 边界 | chunk[{chunk_idx}] | 文本单元={selected_token_count}/{total_tokens} | "
                f"FA-health={score:.3f}",
                req_id=req_id,
            )

        verbose_json = {
            "task": "transcribe",
            "language": run_lang,
            "duration": round(total_audio_duration, 2),
            "report": {},
            "text": clean_reference_text,
            "words": [],
        }
        all_word_alignments = []
        for chunk_info, ground_truth, fa_res in aligned_chunks:
            self._assert_fa_timestamps_in_chunk(
                fa_res,
                chunk_info["start"],
                chunk_info["end"],
                req_id,
            )
            rebuilt_words = self._rebuild_words_from_ground_truth(
                ground_truth,
                fa_res,
                run_lang,
            )
            for word in rebuilt_words:
                verbose_json["words"].append({
                    "word": word["text"],
                    "start": round(chunk_info["start"] + word["start"], 3),
                    "end": round(chunk_info["start"] + word["end"], 3),
                })
            all_word_alignments.extend(fa_res)

        self._validate_subtitle_timestamps(verbose_json["words"], req_id=req_id)
        verbose_json["report"] = self._generate_diff_report(
            clean_reference_text,
            verbose_json["text"],
            run_lang,
        )

        total_pipeline_time = time.time() - pipeline_start_time
        global_fa_diag = []
        global_fa_score = self._evaluate_fa_health(
            all_word_alignments,
            total_audio_duration,
            diag_out=global_fa_diag,
            run_lang=run_lang,
        )
        task_info = get_task_info(req_id)
        file_name = task_info.get("file", "(未知)") if task_info else "(未知)"
        log_block([
            "📊 最终报告 (纯 ForcedAligner 模式)",
            f"文件名       : {file_name}",
            f"执行状态     : ✅ 成功 (FA-health: {global_fa_score:.3f})",
            f"物理切片     : {len(physical_chunks)} 段 -> 有效对齐 {len(aligned_chunks)} 段",
            f"字符匹配     : 原文={len(clean_reference_text)}字 | 对齐={len(verbose_json['words'])}词条",
            f"音频时长     : {total_audio_duration:.2f}s",
            f"流水线耗时   : {total_pipeline_time:.2f}s",
            f"RTF          : {total_pipeline_time / total_audio_duration:.3f}",
        ] + global_fa_diag, req_id=req_id)
        return verbose_json

    def process_audio_asr_only(self, audio_path, language_code=None, req_id="Unknown", sampling_params=None):
        pipeline_start_time = time.time()
        run_lang = language_code if language_code else self.config["language"]

        try:
            try: audio_data, sample_rate = sf.read(audio_path)
            except Exception as err:
                log_error(f"阶段: 音频读取失败 | 详情: {err}", req_id=req_id)
                raise
                
            wav = torch.from_numpy(audio_data).float()
            if wav.dim() > 1: wav = wav.mean(dim=1)
            if sample_rate != 16000:
                import torchaudio.functional as F
                wav = F.resample(wav, sample_rate, 16000)
                sample_rate = 16000

            total_audio_duration = len(wav) / sample_rate
            
            try:
                with self.vad_lock:
                    raw_speech_timestamps = self.get_speech_timestamps(
                        wav, self.vad_model, return_seconds=True,
                        min_silence_duration_ms=300,
                    )
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

            try:
                asr_results = self.asr_scheduler.submit(
                    audios=batch_audios,
                    language=run_lang,
                    req_id=req_id,
                    sampling_params=sampling_params,
                )
            except Exception as err:
                log_error(f"阶段: 纯ASR模型批量推理崩溃 | 详情: {err}", req_id=req_id)
                raise

            asr_stage_duration = time.time() - asr_stage_start

            text_parts = []
            for idx, (chunk_info, res) in enumerate(zip(physical_chunks, asr_results)):
                asr_text = res.text
                if asr_text:
                    text_parts.append(asr_text)
                    verbose_json["segments"].append({
                        "id": idx, "start": round(chunk_info['start'], 3), "end": round(chunk_info['end'], 3), "text": asr_text
                    })

            verbose_json["text"] = "".join(text_parts)

            total_pipeline_time = time.time() - pipeline_start_time

            seg_count = len(verbose_json["segments"])
            char_count = len(verbose_json["text"])

            task_info = get_task_info(req_id)
            file_name = task_info.get("file", "(未知)") if task_info else "(未知)"

            log_block([
                f"📊 最终报告 (纯ASR模式)",
                f"文件名       : {file_name}",
                f"执行状态     : ✅ 成功完成",
                f"切片与分段   : VAD {len(physical_chunks)} 段 -> {seg_count} 个识别段落",
                f"识别字符     : {char_count} 字",
                f"音频时长     : {total_audio_duration:.2f}s",
                f"ASR 推理     : {asr_stage_duration:.2f}s",
                f"流水线耗时   : {total_pipeline_time:.2f}s",
                f"RTF          : {total_pipeline_time / total_audio_duration:.3f}",
            ], req_id=req_id)

            return verbose_json
            
        except Exception as outer_err:
            raise outer_err
