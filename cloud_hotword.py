import os
import re
import time
import unicodedata
import requests

import Levenshtein
from fa_timestamp_validation import (
    FaTimestampError,
    validate_absolute_timestamps,
    validate_fa_timestamps_in_chunk,
)
from hotword_matching import (
    assess_reference_coverage,
    build_asr_stream,
    find_best_span,
    find_tail_matches,
    select_monotonic_chain,
)

try:
    import zhconv
except ImportError:
    raise ImportError("未找到 zhconv 库，请运行: pip install zhconv")

import sys as _sys
if 'main' not in _sys.modules and '__main__' in _sys.modules:
    _sys.modules['main'] = _sys.modules['__main__']
from main import log, log_info, log_warn, log_error, log_block, get_task_info

# Cloud ASR (qwen-audio-3.0-asr-flash-filetrans 等) 的语言代码契约。
# 来源：https://platform.qianwenai.com/docs/api-reference/speech-recognition/fun-asr-recording/restful-api
#
# 适用模型：qwen-audio-3.0-asr-flash-filetrans, fun-asr, fun-asr-2025-11-07,
#           fun-asr-mtl, fun-asr-mtl-2025-08-25
# 官方支持的语言代码（parameters.language）：
#   zh 中文, en 英文, ja 日语, ko 韩语, vi 越南语, th 泰语, id 印尼语,
#   ms 马来语, tl 菲律宾语, hi 印地语, ar 阿拉伯语, fr 法语, de 德语,
#   es 西班牙语, pt 葡萄牙语, ru 俄语, it 意大利语, nl 荷兰语, sv 瑞典语,
#   da 丹麦语, fi 芬兰语, no 挪威语, el 希腊语, pl 波兰语, cs 捷克语,
#   hu 匈牙利语, ro 罗马尼亚语, bg 保加利亚语, hr 克罗地亚语, sk 斯洛伐克语
#
# 本分支当前仅放行 zh / en；未来扩展其他语言时，把目标代码加入
# SUPPORTED_LANGUAGES 即可（热词提取/归一化目前只对中英做了专门适配，
# 其余语言需先验证 extract_hotwords / norm_for_match 的行为）。
SUPPORTED_LANGUAGES = {"zh", "en"}

LANGUAGE_ALIASES = {
    "chinese": "zh",
    "english": "en",
}


def resolve_language(language_code):
    """Resolve language code/alias to zh or en. Raises ValueError for unsupported."""
    lang = (language_code or "").lower().strip()
    lang_key = lang.split("-")[0].split("_")[0]
    lang_key = LANGUAGE_ALIASES.get(lang_key, lang_key)
    if lang_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Cloud hotword 分支仅支持中文(zh)和英文(en)，"
            f"当前语言 '{language_code}' 不支持。请使用本地模型分支处理其他语言。"
        )
    return lang_key

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/api/v1"
CLOUD_ASR_MODEL = "qwen-audio-3.0-asr-flash-filetrans"
CLOUD_ASR_ENDPOINT = f"{DASHSCOPE_BASE}/services/audio/asr/transcription"

HOTWORD_WEIGHT = 4
MAX_HOTWORDS = 2000
CHUNK_SEC = 60
HOTWORDS_PER_CHUNK = 3
PARTIAL_MATCH_RATIO = 0.8
PARTIAL_MIN_LEN = 5
FA_MAX_SEC = 270
TEXT_RATE_BY_LANG = {"zh": 4.5, "en": 12.5}
TAIL_TEXT_CAPACITY_FACTOR = 1.8

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

EN_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}

def _is_punct(ch):
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _cn_number_to_digits(s):
    def _replace(m):
        match = m.group(0)
        if len(match) == 1 and match in CN_DIGIT and match not in CN_UNIT:
            return match
        result = 0
        current = 0
        wan = 0
        for ch in match:
            if ch in CN_DIGIT:
                current = CN_DIGIT[ch]
            elif ch in CN_UNIT:
                u = CN_UNIT[ch]
                if u == 100000000:
                    result = (result + (current or 1)) * u
                    wan = 0
                    current = 0
                elif u == 10000:
                    wan = (result + (current or 1)) * u
                    result = 0
                    current = 0
                else:
                    result += (current or 1) * u
                    current = 0
        return str(wan + result + current)

    return re.sub(r'[零一二三四五六七八九十百千万亿两]+', _replace, s)


def _parse_en_number_words(words):
    result = 0
    current = 0
    for w in words:
        if w in EN_ONES:
            current += EN_ONES[w]
        elif w in EN_TENS:
            current += EN_TENS[w]
        elif w in EN_SCALES:
            scale = EN_SCALES[w]
            if scale == 100:
                current = (current or 1) * scale
            else:
                result = (result + (current or 1)) * scale
                current = 0
    return result + current


def _en_number_words_to_digits(s):
    tokens = s.split()
    out = []
    num_buf = []

    def flush():
        if num_buf:
            out.append(str(_parse_en_number_words(num_buf)))
            num_buf.clear()

    for tok in tokens:
        clean = re.sub(r'[^\w]', '', tok).lower()
        ordinal = re.sub(r'(st|nd|rd|th)$', '', clean)
        if clean in EN_ONES or clean in EN_TENS or clean in EN_SCALES:
            num_buf.append(clean)
        elif ordinal != clean and (ordinal in EN_ONES or ordinal in EN_TENS or ordinal in EN_SCALES):
            num_buf.append(ordinal)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def norm_for_match(s, lang):
    if lang == "zh":
        s = _cn_number_to_digits(s)
        s = re.sub(r'\s+', '', s)
        s = "".join(ch for ch in s if not _is_punct(ch))
        return s.lower()
    else:
        s = _en_number_words_to_digits(s)
        s = "".join(ch for ch in s if not _is_punct(ch))
        s = re.sub(r'\s+', '', s)
        return s.lower()


def _is_contraction_apostrophe(line, i):
    ch = line[i]
    if ch not in ("'", "\u2019"):
        return False
    return i > 0 and i < len(line) - 1 and line[i - 1].isalpha() and line[i + 1].isalpha()


def _split_clauses(line):
    clauses = []
    current = ""
    for i, ch in enumerate(line):
        if _is_punct(ch) and not _is_contraction_apostrophe(line, i):
            if current.strip():
                clauses.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        clauses.append(current.strip())
    return clauses


def _is_valid_hotword_zh(clause):
    stripped = re.sub(r'\s+', '', clause)
    return 5 <= len(stripped) <= 15


def _is_valid_hotword_en(clause):
    words = clause.strip().split()
    return 2 <= len(words) <= 7


def extract_hotwords(txt_content, lang):
    lines = txt_content.split("\n")
    full_norm = norm_for_match("\n".join(lines), lang)
    is_zh = lang == "zh"

    candidates = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for clause in _split_clauses(stripped):
            valid = _is_valid_hotword_zh(clause) if is_zh else _is_valid_hotword_en(clause)
            if not valid:
                continue
            norm_text = norm_for_match(clause, lang)
            if not norm_text:
                continue
            count = full_norm.count(norm_text)
            if count != 1:
                continue
            offset = full_norm.index(norm_text)
            candidates.append({"text": clause, "normText": norm_text, "offset": offset})

    candidates.sort(key=lambda c: c["offset"])

    rate = TEXT_RATE_BY_LANG["zh" if is_zh else "en"]
    total_chars = len(full_norm)
    est_duration_sec = total_chars / rate
    num_segments = max(1, int(est_duration_sec / CHUNK_SEC) + (1 if est_duration_sec % CHUNK_SEC else 0))
    chars_per_segment = total_chars / num_segments

    selected = []
    seen_norms = set()
    for seg in range(num_segments):
        seg_start = seg * chars_per_segment
        seg_end = (seg + 1) * chars_per_segment
        in_seg = [c for c in candidates if seg_start <= c["offset"] < seg_end]
        if not in_seg:
            continue
        step = len(in_seg) / HOTWORDS_PER_CHUNK
        for i in range(HOTWORDS_PER_CHUNK):
            idx = min(int(i * step), len(in_seg) - 1)
            c = in_seg[idx]
            if c["normText"] not in seen_norms:
                seen_norms.add(c["normText"])
                selected.append({
                    "text": c["text"],
                    "normText": c["normText"],
                    "offsetInFullNorm": c["offset"],
                    "segmentIndex": seg,
                })

    return selected[:MAX_HOTWORDS]


def _get_api_key():
    key = os.environ.get("dashscope_apikey") or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("环境变量 dashscope_apikey 未设置，cloud ASR 分支需要百炼 API Key")
    return key


def _upload_oss(local_path, req_id="Unknown"):
    api_key = _get_api_key()
    policy_url = f"{DASHSCOPE_BASE}/uploads?action=getPolicy&model={CLOUD_ASR_MODEL}"
    resp = requests.get(policy_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    resp.raise_for_status()
    policy = resp.json().get("data", {})

    upload_dir = policy["upload_dir"]
    filename = os.path.basename(local_path)
    key = f"{upload_dir}/{filename}"

    field_map = {
        "oss_access_key_id": "OSSAccessKeyId",
        "signature": "Signature",
        "policy": "policy",
        "x_oss_object_acl": "x-oss-object-acl",
        "x_oss_forbid_overwrite": "x-oss-forbid-overwrite",
    }
    data = {}
    for src, dst in field_map.items():
        data[dst] = str(policy[src])
    data["key"] = key
    data["success_action_status"] = "200"

    f = open(local_path, "rb")
    try:
        upload_resp = requests.post(policy["upload_host"], data=data, files={"file": (filename, f)})
    finally:
        f.close()
    if upload_resp.status_code != 200:
        raise RuntimeError(f"OSS 上传失败: {upload_resp.status_code} {upload_resp.text}")

    log_info(f"OSS 上传完成 | key={key}", req_id=req_id)
    return f"oss://{key}"


def _submit_cloud_asr(file_url, lang, vocabulary, req_id="Unknown"):
    api_key = _get_api_key()
    body = {
        "model": CLOUD_ASR_MODEL,
        "input": {"file_urls": [file_url]},
        "parameters": {
            "channel_id": [0],
            "enable_itn": False,
            "language": lang,
            "enable_words": True,
            "vocabulary": vocabulary,
        },
    }
    resp = requests.post(
        CLOUD_ASR_ENDPOINT,
        json=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        },
    )
    resp.raise_for_status()
    result = resp.json()
    task_id = result.get("output", {}).get("task_id", "")
    if not task_id:
        raise RuntimeError(f"Cloud ASR 提交失败，无 task_id: {result}")
    log_info(f"Cloud ASR 已提交 | task_id={task_id}", req_id=req_id)
    return task_id


def _poll_cloud_asr(task_id, req_id="Unknown", timeout_sec=600):
    api_key = _get_api_key()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = requests.get(
            f"{DASHSCOPE_BASE}/tasks/{task_id}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        query = resp.json()
        output = query.get("output", {})
        status = output.get("task_status", "")

        if status == "SUCCEEDED":
            results = output.get("results", [])
            url = ""
            if results:
                url = results[0].get("transcription_url", "")
            if not url:
                url = output.get("result", {}).get("transcription_url", "")
            if not url:
                url = output.get("transcription_url", "")
            if not url:
                raise RuntimeError(f"Cloud ASR SUCCEEDED 但无 transcription_url: {output}")
            tr_resp = requests.get(url)
            tr_resp.raise_for_status()
            log_info(f"Cloud ASR 转写完成 | task_id={task_id}", req_id=req_id)
            return tr_resp.json()

        if status in ("FAILED", "UNKNOWN"):
            raise RuntimeError(f"Cloud ASR 任务 {status}: {output}")

        time.sleep(3)

    raise RuntimeError(f"Cloud ASR 轮询超时 ({timeout_sec}s) | task_id={task_id}")


def _extract_words(transcription):
    transcripts = transcription.get("transcripts", [])
    if not transcripts:
        return []
    sentences = transcripts[0].get("sentences", [])
    words = []
    for s in sentences:
        words.extend(s.get("words", []))
    return words


def _transcript_text(transcription):
    transcripts = transcription.get("transcripts", [])
    if transcripts:
        return transcripts[0].get("text", "")
    return ""


def _match_hotwords(hotwords, words, lang):
    stream, char_to_word = build_asr_stream(words, lang, norm_for_match)

    matched = []
    partial_matched = []
    unmatched = []
    duplicate_in_asr = []
    search_cursor = 0

    for hw in hotwords:
        norm = hw["normText"]
        first_idx = stream.find(norm, search_cursor)
        if first_idx >= 0:
            second_idx = stream.find(norm, first_idx + 1)
            if second_idx >= 0:
                duplicate_in_asr.append(hw)
                continue
            start_word_idx = char_to_word[first_idx]
            end_word_idx = char_to_word[first_idx + len(norm) - 1]
            begin_time = words[start_word_idx].get("begin_time", 0)
            end_time = words[end_word_idx].get("end_time", 0)
            matched.append({
                **hw,
                "asrOffset": first_idx,
                "asrEndOffset": first_idx + len(norm),
                "beginTime": begin_time,
                "endTime": end_time,
                "matchScore": 1.0,
                "matchType": "exact",
            })
            search_cursor = first_idx + len(norm)
            continue

        span = find_best_span(
            norm,
            stream,
            start=search_cursor,
            min_ratio=PARTIAL_MATCH_RATIO,
            min_length=PARTIAL_MIN_LEN,
        )
        if span is not None:
            start = span["start"]
            end = span["end"]
            start_word_idx = char_to_word[start]
            end_word_idx = char_to_word[min(end - 1, len(char_to_word) - 1)]
            begin_time = words[start_word_idx].get("begin_time", 0)
            end_time = words[end_word_idx].get("end_time", 0)
            partial_matched.append({
                **hw,
                "asrOffset": start,
                "asrEndOffset": end,
                "beginTime": begin_time,
                "endTime": end_time,
                "matchScore": span["score"],
                "matchType": "fuzzy",
            })
            search_cursor = end
            continue

        unmatched.append(hw)

    return matched, partial_matched, unmatched, duplicate_in_asr


def _match_tail_hotwords(hotwords, words, lang, existing_matches):
    candidates = find_tail_matches(
        hotwords,
        words,
        lang,
        norm_for_match,
        existing_matches,
        min_ratio=PARTIAL_MATCH_RATIO,
        min_length=PARTIAL_MIN_LEN,
    )
    matched = []
    for candidate in candidates:
        start_word_idx = candidate["startWordIndex"]
        end_word_idx = candidate["endWordIndex"]
        matched.append({
            **candidate,
            "beginTime": words[start_word_idx].get("begin_time", 0),
            "endTime": words[end_word_idx].get("end_time", 0),
        })
    return matched


def _build_chunks(valid_hotwords, full_norm_len, total_duration_ms):
    if not valid_hotwords:
        return []

    chunks = []
    idx = 0

    first = valid_hotwords[0]
    if first["offsetInFullNorm"] > 0:
        chunks.append({
            "index": idx,
            "textStart": 0,
            "textEnd": first["offsetInFullNorm"] + len(first["normText"]),
            "audioStartMs": 0,
            "audioEndMs": first["endTime"],
        })
        idx += 1

    for i in range(len(valid_hotwords) - 1):
        cur = valid_hotwords[i]
        nxt = valid_hotwords[i + 1]
        chunks.append({
            "index": idx,
            "textStart": cur["offsetInFullNorm"],
            "textEnd": nxt["offsetInFullNorm"] + len(nxt["normText"]),
            "audioStartMs": cur["beginTime"],
            "audioEndMs": nxt["endTime"],
        })
        idx += 1

    last = valid_hotwords[-1]
    if last["offsetInFullNorm"] + len(last["normText"]) < full_norm_len:
        chunks.append({
            "index": idx,
            "textStart": last["offsetInFullNorm"],
            "textEnd": full_norm_len,
            "audioStartMs": last["beginTime"],
            "audioEndMs": total_duration_ms,
        })

    return chunks


def _norm_text_to_orig_map(original_text, lang):
    """Build mapping: norm_text index -> original_text index.

    Applies the same normalization as norm_for_match (number conversion,
    punctuation/whitespace removal, lowercase) while maintaining a
    per-character back-map to the original text positions.
    """
    is_zh = lang == "zh"

    if is_zh:
        work_text = zhconv.convert(original_text, 'zh-cn')
    else:
        work_text = original_text.lower()

    if is_zh:
        work_text = _cn_number_to_digits(work_text)
    else:
        work_text = _en_number_words_to_digits(work_text)

    norm_chars = []
    idx_map = []
    for i, ch in enumerate(work_text):
        if _is_punct(ch) or ch.isspace():
            continue
        norm_chars.append(ch)
        idx_map.append(min(i, len(original_text) - 1))

    norm_text = "".join(norm_chars).lower()
    return norm_text, idx_map


class CloudHotwordAlignPipeline:
    """Cloud ASR (hotword-anchored pathfinding) + local FA alignment pipeline.

    Flow: extract hotwords from reference text -> upload audio to OSS ->
    cloud ASR with vocabulary -> match hotwords in ASR word stream ->
    monotonicity enforcement -> build chunks -> FA alignment per chunk.
    """

    def __init__(self, fa_model, fa_lock, config):
        self.fa_model = fa_model
        self.fa_lock = fa_lock
        self.config = config

    def process_audio(self, audio_path, reference_text, language_code=None, req_id="Unknown"):
        pipeline_start_time = time.time()
        run_lang = resolve_language(language_code or self.config.get("language", "zh"))

        clean_reference_text = re.sub(r'[ \t\r\f\v]+', ' ', reference_text).strip()

        log_info(f"━━━ Cloud Hotword 分支启动 | 语言={run_lang} ━━━", req_id=req_id)

        hotwords = extract_hotwords(clean_reference_text, run_lang)
        log_info(f"热词提取: {len(hotwords)} 个", req_id=req_id)

        if len(hotwords) < 2:
            raise ValueError(
                f"热词不足 2 个（当前 {len(hotwords)}），无法进行锚点切块。"
                f"原文可能过短或缺乏唯一子句。"
            )

        vocabulary = {hw["text"]: HOTWORD_WEIGHT for hw in hotwords}

        file_url = _upload_oss(audio_path, req_id)
        task_id = _submit_cloud_asr(file_url, run_lang, vocabulary, req_id)
        transcription = _poll_cloud_asr(task_id, req_id)

        words = _extract_words(transcription)
        asr_text = _transcript_text(transcription)
        asr_end_time_ms = words[-1].get("end_time", 0) if words else 0

        coverage = assess_reference_coverage(
            clean_reference_text,
            words,
            run_lang,
            norm_for_match,
        )
        log_info(
            f"ASR 文本覆盖 | ratio={coverage.get('coverageRatio', 0):.4f} | "
            f"leading={coverage.get('leadingGap', 0)} | "
            f"internal={coverage.get('internalGap', 0)} | "
            f"trailing={coverage.get('trailingGap', 0)} | "
            f"allowed={coverage.get('allowedGap', 0)}",
            req_id=req_id,
        )
        if not coverage["ok"]:
            reasons = ", ".join(coverage["reasons"])
            raise ValueError(
                f"音频与参考文本覆盖不一致: {reasons} | "
                f"leading={coverage.get('leadingGap', 0)} | "
                f"internal={coverage.get('internalGap', 0)} | "
                f"trailing={coverage.get('trailingGap', 0)} | "
                f"allowed={coverage.get('allowedGap', 0)}"
            )

        import soundfile as sf
        import torch
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

        physical_audio_duration = len(wav) / sample_rate
        physical_duration_ms = physical_audio_duration * 1000
        asr_duration_sec = asr_end_time_ms / 1000.0
        if abs(asr_duration_sec - physical_audio_duration) > 0.05:
            log_warn(
                f"ASR/物理音频时长差异 | ASR末词={asr_duration_sec:.3f}s | "
                f"物理音频={physical_audio_duration:.3f}s | "
                f"差异={physical_audio_duration - asr_duration_sec:+.3f}s",
                req_id=req_id,
            )

        log_info(
            f"ASR 结果 | words={len(words)} | ASR末词={asr_duration_sec:.1f}s | "
            f"音频={physical_audio_duration:.1f}s | 文本={len(asr_text)}字",
            req_id=req_id,
        )

        full_norm = norm_for_match(clean_reference_text, run_lang)
        matched, partial_matched, unmatched, duplicate_in_asr = _match_hotwords(
            hotwords, words, run_lang
        )
        all_matched = matched + partial_matched
        tail_matched = _match_tail_hotwords(unmatched, words, run_lang, all_matched)
        if tail_matched:
            partial_matched.extend(tail_matched)
            all_matched.extend(tail_matched)
            matched_norms = {item["normText"] for item in tail_matched}
            unmatched = [
                item for item in unmatched if item["normText"] not in matched_norms
            ]

        log_block([
            "━━━ 热词匹配诊断 ━━━",
            f"  精确匹配: {len(matched)} | 退化匹配(≥80%): {len(partial_matched)} | "
            f"未匹配: {len(unmatched)} | ASR内重复: {len(duplicate_in_asr)}",
        ] + ([
            "  未匹配热词: " + " | ".join(
                f"{h['text']}@norm[{h['offsetInFullNorm']}]" for h in unmatched
            )
        ] if unmatched else []),
            req_id=req_id, level="INFO")

        valid, discarded = select_monotonic_chain(all_matched, "beginTime")
        if discarded:
            log_warn(f"单调性丢弃: {len(discarded)} 个热词", req_id=req_id)

        # The ASR last-word timestamp can omit trailing audio. The decoded
        # file duration is the authoritative end boundary for the tail chunk.
        # If the last matched anchor is too late, the remaining reference text
        # can be much longer than the tail audio. Qwen FA extrapolates timestamps
        # for this case instead of clamping them to the supplied audio window.
        effective_valid = list(valid)
        removed_tail_anchors = []
        first_tail_capacity_issue = None
        text_rate = TEXT_RATE_BY_LANG[run_lang]
        while len(effective_valid) > 1:
            candidate_chunks = _build_chunks(
                effective_valid, len(full_norm), physical_duration_ms
            )
            if not candidate_chunks:
                break

            tail = candidate_chunks[-1]
            tail_audio_sec = (tail["audioEndMs"] - tail["audioStartMs"]) / 1000.0
            tail_text_len = tail["textEnd"] - tail["textStart"]
            if tail["textEnd"] != len(full_norm) or tail_audio_sec <= 0:
                break

            max_tail_text_len = tail_audio_sec * text_rate * TAIL_TEXT_CAPACITY_FACTOR
            if tail_text_len <= max_tail_text_len:
                break

            if first_tail_capacity_issue is None:
                first_tail_capacity_issue = (
                    tail_text_len, tail_audio_sec, max_tail_text_len
                )
            removed_tail_anchors.append(effective_valid.pop())

        if removed_tail_anchors:
            tail_text_len, tail_audio_sec, max_tail_text_len = first_tail_capacity_issue
            log_warn(
                f"尾块文本容量不足 | norm字数={tail_text_len} | "
                f"音频={tail_audio_sec:.3f}s | 估算容量={max_tail_text_len:.1f}字 | "
                f"合并前移 {len(removed_tail_anchors)} 个尾锚点 | "
                f"移除锚点: "
                + " | ".join(
                    f"{h['text']}@{h['beginTime'] / 1000:.3f}s"
                    for h in removed_tail_anchors
                ),
                req_id=req_id,
            )

        chunks = _build_chunks(effective_valid, len(full_norm), physical_duration_ms)
        log_info(f"切块: {len(chunks)} 个", req_id=req_id)

        if not chunks:
            raise ValueError("热词切块结果为空，无法进行 FA 对齐。")

        norm_ref, ref_idx_map = _norm_text_to_orig_map(clean_reference_text, run_lang)

        verbose_json = {
            "task": "transcribe",
            "language": run_lang,
            "duration": round(physical_audio_duration, 2),
            "report": {},
            "text": "",
            "words": [],
        }

        fa_text_parts = []
        all_word_alignments = []
        fa_diag = []
        emitted_orig_end = 0

        for chunk in chunks:
            text_start_norm = chunk["textStart"]
            text_end_norm = chunk["textEnd"]
            audio_start_sec = chunk["audioStartMs"] / 1000.0
            audio_end_sec = chunk["audioEndMs"] / 1000.0

            orig_start = ref_idx_map[text_start_norm] if text_start_norm < len(ref_idx_map) else 0
            if text_end_norm - 1 < len(ref_idx_map):
                orig_end = ref_idx_map[text_end_norm - 1] + 1
            else:
                orig_end = len(clean_reference_text)

            ground_truth = clean_reference_text[orig_start:orig_end]
            if not ground_truth.strip():
                continue

            chunk_dur = audio_end_sec - audio_start_sec
            if chunk["index"] == chunks[-1]["index"]:
                log_info(
                    f"FA 尾部输入 | chunk[{chunk['index']}] | "
                    f"音频={audio_start_sec:.3f}..{audio_end_sec:.3f}s ({chunk_dur:.3f}s) | "
                    f"原文norm={text_start_norm}:{text_end_norm} ({text_end_norm - text_start_norm}字) | "
                    f"ground_truth={orig_start}:{orig_end} ({len(ground_truth)}字)",
                    req_id=req_id,
                )
            if chunk_dur > FA_MAX_SEC:
                log_warn(
                    f"⚠️ chunk[{chunk['index']}] 时长 {chunk_dur:.1f}s 超过 FA 上限 {FA_MAX_SEC}s，截断",
                    req_id=req_id,
                )
                audio_end_sec = audio_start_sec + FA_MAX_SEC
                chunk_dur = FA_MAX_SEC

            start_sample = max(0, int(audio_start_sec * sample_rate))
            end_sample = min(len(wav), int(audio_end_sec * sample_rate))
            chunk_audio = (wav[start_sample:end_sample].numpy(), sample_rate)
            actual_chunk_dur = (end_sample - start_sample) / sample_rate
            if abs(actual_chunk_dur - chunk_dur) > 0.05:
                raise ValueError(
                    f"chunk 实际音频长度异常 | chunk[{chunk['index']}] "
                    f"requested={chunk_dur:.3f}s | actual={actual_chunk_dur:.3f}s | "
                    f"samples={end_sample - start_sample} | "
                    f"window={audio_start_sec:.3f}..{audio_end_sec:.3f}s"
                )

            if chunk["index"] == chunks[-1]["index"]:
                log_info(
                    f"FA 尾部实际输入 | chunk[{chunk['index']}] | "
                    f"requested={chunk_dur:.3f}s | actual={actual_chunk_dur:.3f}s | "
                    f"samples={end_sample - start_sample}",
                    req_id=req_id,
                )

            try:
                with self.fa_lock:
                    fa_res = self.fa_model.align(audio=[chunk_audio], text=[ground_truth], language=run_lang)[0]
            except Exception as err:
                log_error(f"FA 对齐失败 | chunk[{chunk['index']}] | {err}", req_id=req_id)
                raise

            self._assert_fa_timestamps(
                fa_res, audio_start_sec, audio_end_sec, chunk["index"], req_id
            )

            for w in fa_res:
                all_word_alignments.append(w)

            rebuilt_words = self._rebuild_words(ground_truth, fa_res, run_lang)
            overlap_end = max(emitted_orig_end, orig_start)
            keep_from = max(0, overlap_end - orig_start)
            kept_words, fa_start = self._trim_rebuilt_words(
                ground_truth, rebuilt_words, keep_from, run_lang
            )
            for w in kept_words:
                verbose_json["words"].append({
                    "word": w["text"],
                    "start": round(audio_start_sec + w["start"], 3),
                    "end": round(audio_start_sec + w["end"], 3),
                })
            fa_text_parts.extend(w.text for w in fa_res[fa_start:])
            emitted_orig_end = max(emitted_orig_end, orig_end)

            fa_diag.append(
                f"chunk[{chunk['index']}] {audio_start_sec:.1f}-{audio_end_sec:.1f}s ({chunk_dur:.1f}s) "
                f"原文[{orig_start}:{orig_end}]={orig_end - orig_start}字 FA词数={len(fa_res)}"
            )

        uses_character_tokens = run_lang == "zh"
        verbose_json["text"] = "".join(fa_text_parts) if uses_character_tokens else " ".join(fa_text_parts)

        self._validate_subtitle_timestamps(verbose_json["words"], req_id=req_id)

        norm_result = norm_for_match(verbose_json["text"], run_lang)
        ratio = Levenshtein.ratio(full_norm, norm_result)
        verbose_json["report"] = {
            "accuracy_ratio": f"{ratio * 100:.2f}%",
            "ref_length": len(full_norm),
            "result_length": len(norm_result),
        }

        total_pipeline_time = time.time() - pipeline_start_time
        task_info = get_task_info(req_id)
        file_name = task_info.get("file", "(未知)") if task_info else "(未知)"

        log_block([
            "📊 最终报告 (Cloud Hotword 对齐模式)",
            f"文件名       : {file_name}",
            f"执行状态     : ✅ 成功 (相似度: {verbose_json['report']['accuracy_ratio']})",
            f"热词         : {len(hotwords)} 个 | 有效锚点: {len(effective_valid)} | 切块: {len(chunks)}",
            f"音频时长     : {physical_audio_duration:.2f}s",
            f"流水线耗时   : {total_pipeline_time:.2f}s",
            f"RTF          : {total_pipeline_time / physical_audio_duration:.3f}" if physical_audio_duration > 0 else "RTF: N/A",
        ] + fa_diag, req_id=req_id)

        return verbose_json

    def _assert_fa_timestamps(self, fa_res, chunk_start, chunk_end, chunk_idx, req_id):
        try:
            validate_fa_timestamps_in_chunk(fa_res, chunk_start, chunk_end)
        except FaTimestampError as err:
            message = f"{err} | chunk[{chunk_idx}]"
            log_error(message, req_id=req_id)
            raise

    def _validate_subtitle_timestamps(self, words, req_id="Unknown"):
        try:
            validate_absolute_timestamps(words)
        except FaTimestampError as err:
            log_error(f"❌ {err}", req_id=req_id)
            raise
        log_info(f"✅ 字幕时间戳校验通过 | {len(words)} 个词条时间单调且满足 start <= end", req_id=req_id)

    def _trim_rebuilt_words(self, ground_truth, rebuilt_words, keep_from, lang):
        """Drop text already emitted by an overlapping preceding chunk."""
        uses_character_tokens = lang == "zh"
        if uses_character_tokens:
            tokens = list(ground_truth)

            def is_valid_token(tok):
                return bool(tok) and (tok.isalnum() or unicodedata.category(tok).startswith("L"))
        else:
            tokens = re.findall(r'\w+|[^\w]', ground_truth, re.UNICODE)

            def is_valid_token(tok):
                return bool(tok) and (tok[0].isalnum() or unicodedata.category(tok[0]).startswith("L"))

        kept = []
        fa_start = 0
        cursor = 0
        for token, word in zip(tokens, rebuilt_words):
            token_end = cursor + len(token)
            if token_end <= keep_from:
                if is_valid_token(token):
                    fa_start += 1
            else:
                kept.append(word)
            cursor = token_end
        return kept, fa_start

    def _rebuild_words(self, ground_truth, fa_words, lang):
        uses_character_tokens = lang == "zh"
        if uses_character_tokens:
            tokens = list(ground_truth)

            def is_valid_token(tok):
                return bool(tok) and (tok.isalnum() or unicodedata.category(tok).startswith("L"))
        else:
            tokens = re.findall(r'\w+|[^\w]', ground_truth, re.UNICODE)

            def is_valid_token(tok):
                return bool(tok) and (tok[0].isalnum() or unicodedata.category(tok[0]).startswith("L"))

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
