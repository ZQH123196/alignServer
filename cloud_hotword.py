import os
import re
import time
import unicodedata
import requests

import Levenshtein
from env_config import server_env
from fa_timestamp_validation import (
    FA_BOUNDARY_JITTER_SEC,
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

# Cloud ASR (qwen-audio-3.1-asr-flash-filetrans 等) 的语言代码契约。
# 来源：https://platform.qianwenai.com/docs/api-reference/speech-recognition/fun-asr-recording/restful-api
#
# 适用模型：qwen-audio-3.1-asr-flash-filetrans, fun-asr, fun-asr-2025-11-07,
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
CLOUD_ASR_MODEL = "qwen-audio-3.1-asr-flash-filetrans"
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
# 相邻 chunk 以热词为锚点、文本互相重叠，重叠区由前一个 chunk 输出；但紧随重叠区
# 之后的词（典型为标点）继承的是当前 chunk 自己的 FA 时间戳。同一段文本在两个不同
# 音频窗口里各对齐一次，FA 边界存在毫秒级抖动，块接缝处可能出现小幅时间回退。
# 回退在该阈值内视为正常抖动，输出时钳制为单调并 WARN；超过阈值判定为锚点错位或
# FA 对齐漂移，Fail-Fast 抛错。
SEAM_JITTER_MAX_SEC = FA_BOUNDARY_JITTER_SEC

# chunk 音频窗口两端各留余量，让 FA 的文本有落点，消除「末端差 0.2s」这类纯
# 边界抖动（ASR 锚点 vs FA 边界）。真伪对齐（文本远超音频）余量盖不住，仍会
# 被窗口容差/覆盖闸门抓住。与接缝容差同源，避免同一种物理量出现两把标尺。
FA_CHUNK_PAD_SEC = FA_BOUNDARY_JITTER_SEC

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


def _cn_number_repl(m):
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


CN_NUMBER_RE = re.compile(r'[零一二三四五六七八九十百千万亿两]+')


def _cn_number_to_digits(s):
    return CN_NUMBER_RE.sub(_cn_number_repl, s)


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


def _en_number_word_spans(s):
    """按 token 产出 (文本, 原文起始下标, 是否锚定段首)。

    与 _en_number_words_to_digits 同源：连续数字词归并成一个阿拉伯数字（锚定到该词组首字），
    其余 token 原样保留，故下标可逐字对齐回原文。
    """
    pieces = []
    num_buf = []
    num_start = 0

    def flush():
        if num_buf:
            pieces.append((str(_parse_en_number_words(num_buf)), num_start, True))
            num_buf.clear()

    for m in re.finditer(r'\S+', s):
        tok = m.group(0)
        clean = re.sub(r'[^\w]', '', tok).lower()
        ordinal = re.sub(r'(st|nd|rd|th)$', '', clean)
        if clean in EN_ONES or clean in EN_TENS or clean in EN_SCALES:
            is_number_word = True
            word = clean
        elif ordinal != clean and (ordinal in EN_ONES or ordinal in EN_TENS or ordinal in EN_SCALES):
            is_number_word = True
            word = ordinal
        else:
            is_number_word = False

        if is_number_word:
            if not num_buf:
                num_start = m.start()
            num_buf.append(word)
        else:
            flush()
            pieces.append((tok, m.start(), False))
    flush()
    return pieces


def _en_number_words_to_digits(s):
    return " ".join(text for text, _origin, _anchor in _en_number_word_spans(s))


def _normalized_segments(s, lang):
    """把原文切成 (文本段, 原文起始下标, 是否锚定段首)，覆盖 norm_for_match 全部改写规则。

    未改写的段落逐字对齐原文；数字改写段（中文数字/英文数词 → 阿拉伯数字）整段锚定到
    其在原文中的起点，因此 norm 下标回映射原文不会出现累计漂移。
    """
    if lang != "zh":
        yield from _en_number_word_spans(s)
        return

    cursor = 0
    for m in CN_NUMBER_RE.finditer(s):
        if m.start() > cursor:
            yield s[cursor:m.start()], cursor, False
        digits = _cn_number_repl(m)
        yield digits, m.start(), digits != m.group(0)
        cursor = m.end()
    if cursor < len(s):
        yield s[cursor:], cursor, False


def norm_for_match_with_map(s, lang):
    """norm_for_match 的同源内核，额外产出 norm 下标 -> 原文下标映射。

    禁止另起一套归一化规则：映射与匹配必须逐字同源，否则诊断回显会指错原文。
    """
    last = max(len(s) - 1, 0)
    norm_chars = []
    idx_map = []
    for text, origin, anchor_first in _normalized_segments(s, lang):
        offset = 0
        for ch in text:
            if not _is_punct(ch) and not ch.isspace():
                norm_chars.append(ch.lower())
                idx_map.append(origin if anchor_first else min(origin + offset, last))
            offset += 1
    return "".join(norm_chars), idx_map


def norm_for_match(s, lang):
    return norm_for_match_with_map(s, lang)[0]


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


def _upload_oss(local_path, req_id="Unknown"):
    api_key = server_env().dashscope_apikey
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
    api_key = server_env().dashscope_apikey
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
    api_key = server_env().dashscope_apikey
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

    pad_ms = int(FA_CHUNK_PAD_SEC * 1000)

    def _window(audio_start_ms, audio_end_ms):
        return (
            max(0, audio_start_ms - pad_ms),
            min(total_duration_ms, audio_end_ms + pad_ms),
        )

    chunks = []
    idx = 0

    first = valid_hotwords[0]
    if first["offsetInFullNorm"] > 0:
        audio_start_ms, audio_end_ms = _window(0, first["endTime"])
        chunks.append({
            "index": idx,
            "textStart": 0,
            "textEnd": first["offsetInFullNorm"] + len(first["normText"]),
            "audioStartMs": audio_start_ms,
            "audioEndMs": audio_end_ms,
        })
        idx += 1

    for i in range(len(valid_hotwords) - 1):
        cur = valid_hotwords[i]
        nxt = valid_hotwords[i + 1]
        audio_start_ms, audio_end_ms = _window(cur["beginTime"], nxt["endTime"])
        chunks.append({
            "index": idx,
            "textStart": cur["offsetInFullNorm"],
            "textEnd": nxt["offsetInFullNorm"] + len(nxt["normText"]),
            "audioStartMs": audio_start_ms,
            "audioEndMs": audio_end_ms,
        })
        idx += 1

    last = valid_hotwords[-1]
    if last["offsetInFullNorm"] + len(last["normText"]) < full_norm_len:
        audio_start_ms, audio_end_ms = _window(last["beginTime"], total_duration_ms)
        chunks.append({
            "index": idx,
            "textStart": last["offsetInFullNorm"],
            "textEnd": full_norm_len,
            "audioStartMs": audio_start_ms,
            "audioEndMs": audio_end_ms,
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


DIAG_SNIPPET_LIMIT = 160
# 音频尾部未被转写的时长在此以内，视为 ASR 已走到音频结尾（正常收尾静音）
COVERAGE_TAIL_SILENCE_MS = 1000


def _clip_text(text, limit, from_tail=False):
    collapsed = re.sub(r'\s+', ' ', text).strip()
    if len(collapsed) <= limit:
        return collapsed
    if from_tail:
        return "…" + collapsed[-limit:]
    head = limit // 2
    return collapsed[:head] + "…" + collapsed[-(limit - head):]


def describe_reference_region(reference_text, norm_to_orig, start, end, limit=DIAG_SNIPPET_LIMIT):
    """归一化下标区间 -> 原文可读片段（须与 norm_for_match_with_map 同源，故下标严格对应）。"""
    if not norm_to_orig:
        return ""
    left = norm_to_orig[min(max(start, 0), len(norm_to_orig) - 1)]
    right = norm_to_orig[min(max(end - 1, 0), len(norm_to_orig) - 1)] + 1
    return _clip_text(reference_text[left:right], limit)


def coverage_gap_verdict(missing_chars, physical_duration_ms, asr_end_time_ms, lang):
    """由「缺失字数」与「音频里没被转写的尾时长」推出成因分流判据。

    两者按 TEXT_RATE_BY_LANG 换算后对得上，说明音频时长确实容得下这段文本而 ASR 没转出来
    （ASR 侧问题）；对不上说明音频本身就没这段内容（上游产物缺料）。
    """
    if missing_chars <= 0:
        return "缺失面为零"
    uncovered_tail_sec = (physical_duration_ms - asr_end_time_ms) / 1000.0
    expected_tail_sec = missing_chars / TEXT_RATE_BY_LANG[lang]
    if uncovered_tail_sec < 0:
        return (
            f"ASR 末词时间戳 {-uncovered_tail_sec:.2f}s 超出音频结尾——时间戳与音频本身矛盾，"
            f"先查 Cloud ASR 返回的 words 时间轴，再谈缺失的 {missing_chars} 字"
        )
    if uncovered_tail_sec <= COVERAGE_TAIL_SILENCE_MS / 1000.0:
        return (
            f"ASR 末词已走到音频结尾（未转写尾 {uncovered_tail_sec:.2f}s）→ "
            f"音频本身不含这段文本：查上游 TTS 产物是否缺尾段，不要查 ASR"
        )
    if uncovered_tail_sec >= expected_tail_sec * 0.5:
        return (
            f"音频尾部有 {uncovered_tail_sec:.2f}s 未被转写，缺失文本按 {TEXT_RATE_BY_LANG[lang]} 字/s "
            f"约需 {expected_tail_sec:.2f}s → 时长对得上：疑 Cloud ASR 截断/漏转该段，"
            f"音频可能是完整的"
        )
    return (
        f"音频尾部有 {uncovered_tail_sec:.2f}s 未被转写，但缺失文本只需约 {expected_tail_sec:.2f}s → "
        f"多出的静音无法解释：参考文本比音频长（多余尾注/未合成内容），查送进来的 prompt"
    )


def format_coverage_diagnosis(
    coverage, asr_norm_text,
    physical_duration_ms, asr_end_time_ms, lang,
):
    """覆盖闸门的人读诊断，日志与异常共用这一份文本，禁止两处措辞分叉。

    缺失区段的 `text` 已在 coverage 内由调用方的 describe 回调渲染成原文片段。
    """
    offenders = coverage.get("offenders", [])
    missing_chars = sum(item["length"] for item in offenders)
    lines = [
        f"refLen={coverage.get('referenceLength', 0)} | asrLen={coverage.get('asrLength', 0)} | "
        f"matched={coverage.get('matchedLength', 0)} | ratio={coverage.get('coverageRatio', 0):.4f} | "
        f"leading={coverage.get('leadingGap', 0)} | internal={coverage.get('internalGap', 0)} | "
        f"trailing={coverage.get('trailingGap', 0)} | allowed={coverage.get('allowedGap', 0)}",
        f"音频={physical_duration_ms / 1000.0:.2f}s | ASR末词={asr_end_time_ms / 1000.0:.2f}s | "
        f"缺失合计={missing_chars}字",
        f"判据: {coverage_gap_verdict(missing_chars, physical_duration_ms, asr_end_time_ms, lang)}",
    ]
    for item in offenders:
        lines.append(
            f"缺失[{item['kind']}] refNorm[{item['start']}..{item['end']}]（{item['length']}字）: "
            f"「{item['text']}」"
        )
    if asr_norm_text:
        lines.append(
            f"ASR 流尾: 「{_clip_text(asr_norm_text, DIAG_SNIPPET_LIMIT, from_tail=True)}」"
        )
    return "\n".join(lines)


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
                f"语言={run_lang} | 参考文本={len(clean_reference_text)}字 | "
                f"文本头: 「{_clip_text(clean_reference_text, DIAG_SNIPPET_LIMIT)}」"
            )

        vocabulary = {hw["text"]: HOTWORD_WEIGHT for hw in hotwords}

        file_url = _upload_oss(audio_path, req_id)
        task_id = _submit_cloud_asr(file_url, run_lang, vocabulary, req_id)
        transcription = _poll_cloud_asr(task_id, req_id)

        words = _extract_words(transcription)
        asr_text = _transcript_text(transcription)
        asr_end_time_ms = words[-1].get("end_time", 0) if words else 0

        # 音频时长必须在覆盖闸门之前算：它是区分「音频缺料」与「ASR 漏转」的唯一判据，
        # 写在 raise 之后等于失败时永远拿不到。
        import soundfile as sf
        import torch
        try:
            audio_data, sample_rate = sf.read(audio_path)
        except Exception as err:
            log_error(
                f"阶段: 音频读取失败 | 文件={audio_path} | "
                f"大小={os.path.getsize(audio_path)}B | 详情: {err}",
                req_id=req_id,
            )
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

        # 变量名与 FA 路径的 ref_idx_map（含繁简转换，另一套归一化）刻意分开：两者下标空间不同
        reference_norm, diag_idx_map = norm_for_match_with_map(clean_reference_text, run_lang)
        asr_norm, _ = build_asr_stream(words, run_lang, norm_for_match)
        coverage = assess_reference_coverage(
            clean_reference_text,
            words,
            run_lang,
            norm_for_match,
            describe=lambda start, end: describe_reference_region(
                clean_reference_text, diag_idx_map, start, end
            ),
        )
        coverage_diagnosis = format_coverage_diagnosis(
            coverage, asr_norm, physical_duration_ms, asr_end_time_ms, run_lang,
        )
        if not coverage["ok"]:
            log_block(
                ["━━━ ASR 文本覆盖不足 ━━━"] + coverage_diagnosis.split("\n"),
                req_id=req_id, level="ERROR",
            )
            raise ValueError(
                f"音频与参考文本覆盖不一致: {', '.join(coverage['reasons'])}\n"
                f"{coverage_diagnosis}\n"
                f"参考文本尾: 「{_clip_text(clean_reference_text, DIAG_SNIPPET_LIMIT, from_tail=True)}」"
            )

        log_info(
            f"ASR 文本覆盖 OK | ratio={coverage['coverageRatio']:.4f} | "
            f"internal={coverage['internalGap']} | allowed={coverage['allowedGap']}",
            req_id=req_id,
        )

        full_norm = reference_norm
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
        word_origins = []
        last_emitted = None
        seam_clamp_count = 0

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
            chunk_origin = f"chunk[{chunk['index']}] {audio_start_sec:.3f}..{audio_end_sec:.3f}s"

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
                start = round(audio_start_sec + w["start"], 3)
                end = round(audio_start_sec + w["end"], 3)
                start, end, clamped = self._enforce_seam_monotonicity(
                    w["text"], start, end, chunk_origin, last_emitted, req_id
                )
                if clamped:
                    seam_clamp_count += 1
                verbose_json["words"].append({
                    "word": w["text"],
                    "start": start,
                    "end": end,
                })
                word_origins.append(chunk_origin)
                last_emitted = (start, end, w["text"], chunk_origin)
            fa_text_parts.extend(w.text for w in fa_res[fa_start:])
            emitted_orig_end = max(emitted_orig_end, orig_end)

            fa_diag.append(
                f"chunk[{chunk['index']}] {audio_start_sec:.1f}-{audio_end_sec:.1f}s ({chunk_dur:.1f}s) "
                f"原文[{orig_start}:{orig_end}]={orig_end - orig_start}字 FA词数={len(fa_res)}"
            )

        uses_character_tokens = run_lang == "zh"
        verbose_json["text"] = "".join(fa_text_parts) if uses_character_tokens else " ".join(fa_text_parts)

        if seam_clamp_count:
            log_warn(
                f"块接缝钳制汇总 | 共 {seam_clamp_count} 个词条时间戳被钳制为单调（详见上方逐条 WARN）",
                req_id=req_id,
            )
        self._validate_subtitle_timestamps(
            verbose_json["words"], origins=word_origins, req_id=req_id
        )

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
            overflow = validate_fa_timestamps_in_chunk(fa_res, chunk_start, chunk_end)
        except FaTimestampError as err:
            message = f"{err} | chunk[{chunk_idx}]"
            log_error(message, req_id=req_id)
            raise
        if overflow:
            log_warn(
                f"FA 时间戳小幅越界已放行 | chunk[{chunk_idx}] | 越界={overflow:+.3f}s | "
                f"窗口={chunk_start:.3f}..{chunk_end:.3f}s | 跨模型边界抖动，非伪对齐",
                req_id=req_id,
            )

    def _enforce_seam_monotonicity(self, word_text, start, end, origin, last_emitted, req_id):
        """强制跨 chunk 输出单调，吸收块接缝处的 FA 时间戳抖动。

        重叠文本在相邻两个 chunk 里各对齐一次：重叠区本身由前一个 chunk 输出，
        但紧随重叠区之后的词（典型为标点，继承当前块 FA 的前词 end）来自当前 chunk
        的对齐，两者存在毫秒级差异，接缝处可能出现小幅时间回退。
        回退 <= SEAM_JITTER_MAX_SEC 视为正常抖动，钳制到前项边界并 WARN；
        超过阈值判定为锚点错位/FA 漂移，Fail-Fast 抛错。
        返回 (start, end, clamped)。
        """
        if last_emitted is None:
            return start, end, False
        p_start, p_end, p_word, p_origin = last_emitted
        regression = max(p_start - start, p_end - end)
        if regression <= 0:
            return start, end, False
        if regression > SEAM_JITTER_MAX_SEC:
            message = (
                f"❌ 跨 chunk 接缝时间戳回退超容差 | 回退={regression:.3f}s "
                f"(容差={SEAM_JITTER_MAX_SEC}s) | "
                f"前项'{p_word}' {p_start:.3f}..{p_end:.3f}s 来源={p_origin} | "
                f"当前'{word_text}' {start:.3f}..{end:.3f}s 来源={origin} | "
                f"可能原因: 热词锚点(beginTime/endTime)错位或该段 FA 对齐漂移"
            )
            log_error(message, req_id=req_id)
            raise FaTimestampError(message)
        new_start = max(start, p_start)
        new_end = max(end, p_end)
        log_warn(
            f"块接缝时间戳抖动钳制 | 回退={regression:.3f}s | "
            f"'{word_text}' {start:.3f}..{end:.3f}s -> {new_start:.3f}..{new_end:.3f}s | "
            f"前项'{p_word}' 来源={p_origin} | 当前来源={origin}",
            req_id=req_id,
        )
        return new_start, new_end, True

    def _validate_subtitle_timestamps(self, words, origins=None, req_id="Unknown"):
        try:
            validate_absolute_timestamps(words, origins=origins)
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
