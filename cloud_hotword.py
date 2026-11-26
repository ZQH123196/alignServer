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
    clamp_seam_interval,
    normalize_fa_interval_overlaps,
    validate_absolute_timestamps,
    validate_fa_timestamps_in_chunk,
)
from fallback_chunks import build_fallback_chunks
from hotword_matching import assess_reference_coverage, build_asr_stream
from occurrence_matching import (
    build_reference_offset_lookup,
    match_hotword_anchors,
    sample_anchors,
)
from term_selection import select_hotword_terms

# 繁简转换只在本文件之外的 pipeline 分支使用；本分支的 norm 下标必须与切块同源，
# 故这里不再做 zhconv 改写（改写会改变文本长度，正是 2026-10-18 第 44 章漂移的成因）。

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
# 其余语言需先验证 term_selection / norm_for_match 的行为）。
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
ANCHOR_GAP_WARN_MS = 60_000
FA_MAX_SEC = 270
TEXT_RATE_BY_LANG = {"zh": 4.5, "en": 12.5}
TAIL_TEXT_CAPACITY_FACTOR = 1.8
# 接缝抖动容差（秒）。**只属于热词锚点分块这一条链路**，不是全仓通用规则。
#
# 为什么只有它有：相邻 chunk 的文本按热词锚点互相重叠，同一段文本会在两个音频窗口
# 里各对齐一次；重叠区由前一个 chunk 输出，紧随其后的词（典型为标点）继承当前 chunk
# 自己的 FA 时间戳。同一个边界的两次估边存在毫秒级差异，块接缝处就会出现时间回退。
#
# 判据是**区间**：后一个词的 start 不得早于此前所有已输出词的 end。只比前一项的
# start/end 会让「标点 end 被抬高、后一个实词起点仍在它之前」漏过去（见
# `_enforce_seam_monotonicity` 的说明）。回退在该阈值内视为正常抖动，钳制并 WARN；
# 超过阈值判定为锚点错位或 FA 对齐漂移，Fail-Fast 抛错。
#
# 另外两条链路（pipeline 的本地 ASR+FA、纯 FA）按 IPA 游标切文本，每段只对齐一次，
# 不存在同一段文本的两次估边，因此不套本阈值，只在末尾做点态单调校验。
# **两条链路的接缝行为不同是有意为之**：压平的前提是重叠来自「同一段文本被对齐两次」，
# 不是把任何接缝偏差都吸收掉。本仓投产的只有热词这一条；将来若要启用那两条，
# 先决定用哪条不变式，再谈共用。
SEAM_JITTER_MAX_SEC = FA_BOUNDARY_JITTER_SEC

# chunk 音频窗口两端各留余量，让 FA 的文本有落点，消除「末端差 0.2s」这类纯
# 边界抖动（ASR 锚点 vs FA 边界）。真伪对齐（文本远超音频）余量盖不住，仍会
# 被窗口容差/覆盖率校验抓住。与接缝容差同源，避免同一种物理量出现两把标尺。
FA_CHUNK_PAD_SEC = FA_BOUNDARY_JITTER_SEC

# ── 块级修复判据（一次请求 = 一个章节的一种语言）──────────────────────
#
# 为什么需要：FA 对块内一部分有效 token 给不出时间戳时，`_rebuild_words` 只能把
# 尾部 token 写成零时长（start == end）。零时长 token 一路流到字幕，变成零时长 cue，
# 最后撞上成片的字幕几何契约（begin == end 非法）。补位率门禁看不见它 —— 第 6 章
# 补位率 5.32%（阈值 10%），却有一段 15.48s 的空档和 71 个连续零时长 token。
#
# 修复单位是**块**，不是 token：块内时间轴缺口只能按原文 token 长度摊平，逐 token
# 猜时间没有依据。因此均分判据、容差带、每请求可容忍的坏块上限全部留在本文件，
# 客户端只读响应里的 `report.align.defect` 是否存在，不认识块结构。
#
# 上限写在这里而不是客户端配置：分块是服务端内部概念，客户端无从知道本次请求被切成
# 几块，也就无法判定「坏块是否过多」。一次请求就是一次章节级判定。
MAX_REPAIRED_CHUNKS_PER_REQUEST = 2

# 语速容差带，**比值**不是绝对值。参考语速 R 取本次请求内未判坏块的有效 token 数 /
# 音频时长，坏块语速 r 与它同单位（token/秒），故 r/R 与语言、音色、语速设置无关。
# 禁止改成绝对 token/秒阈值：绝对值随语言与语速变化，等于给每种语言配一套魔数。
CHUNK_RATE_BAND_LOW = 0.6
CHUNK_RATE_BAND_HIGH = 1.5

# 词流对块窗口的覆盖率容差（秒）。首词起点距块起点、末词终点距块终点超过它即判坏：
# 说明这段音频里的发音没有被 FA 覆盖，块内必然存在零时长区段。
CHUNK_COVERAGE_TOLERANCE_SEC = 1.0

# 均分后单个 token 的最小跨度（秒）。低于它说明该块 token 密度远超音频容量，
# 摊平会把时间轴压成零时长，同样不可用。
CHUNK_TOKEN_MIN_SPAN_SEC = 0.001

# 零时长有效 token 的**连续段**下限（个）。达到它才判块坏。
#
# 为什么不是「存在一个就算」：FA 逐字估边，零星给出 start == end 是常态而非故障
# （第 6 章整章补位率 5.32%，散落在各块里）。实测 2026-10-27 单章验证：按「存在即坏」
# 判定时 16 块里 12 块被判坏，全部误报 —— 那个判据把正常块当成坏块去均分，
# 反而毁掉了本来准确的块内时间。真正的故障形态是**成段**零时长（第 6 章实测
# 71 个连续零时长 token），故判据只认连续段。
#
# 取 5 与客户端摘要的告警阈值（`LONG_ZERO_RUN_TOKENS`）同数量级，便于两侧日志对照。
# **两处统计口径不同**：服务端只数有效 token，客户端连标点一起数，不要当成同一条判据。
# 判据本身是绝对个数而不是比例：块内 token 规模在百级，5 个连续零时长在任何语言里
# 都已经是「FA 放弃这一段」而不是零星估边。
CHUNK_ZERO_RUN_MIN = 5

# 词锚定的前跳上限（字符）。FA 偶尔在一处吞掉几个字，后一个词仍要锚回原位，故允许
# 越过缺口去匹配；但跳太远说明这条词和原文对不上，宁可不锚、交给缺口夹 ——
# 错锚会让整块时间跳变，比留一个缺口危险得多。
# 取 40：中文一个词最长不过十余字，40 已覆盖「整句被吞」的缺口，又不足以跨过
# 相邻的同名短语（同名词重复出现时按顺序锚定，不会误配后面那处）。
CHUNK_ANCHOR_SEARCH_CHARS = 40

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

def _is_match_char(ch):
    """参与匹配的字符：字母、数字、组合标记。

    与 server 端 TTS 白名单同源（`apps/server/src/shared/utils/text-normalize.ts` 的
    `isAllowedTtsChar`）：TTS 那份额外保留若干标点，而标点在这里必须删（ASR 不转写停顿）。
    除此之外两侧保留集逐类别相同 —— 因此 TTS 删掉的字符，这里一定也删。

    不能写成「非标点即保留」：那样会漏掉 `C` 组（零宽空格、BOM、控制符），
    它们是不可见字符，留在归一化结果里就是凭空的参考字。

    版本偏差（已知、可接受）：`unicodedata` 的 Unicode 版本可能落后于 Node 的 ICU
    （本机 Python 15.0 / Node 17.0）。新版本才分配的生僻码位在这里是 `Cn`，会被删；
    TTS 侧按新版本认成字母、会保留。影响面是每出现一次差一个字。
    CJK 扩展 I（U+2EBF0 起）正是这一类：它是汉字，但 Python 15 判 `Cn`。
    """
    return unicodedata.category(ch)[0] in ("L", "N", "M")


_FULLWIDTH_ALNUM_OFFSET = 0xFEE0


def _halfwidth_alnum(ch):
    """全角字母数字 -> 半角；全角空格 -> 半角空格。标点一律不动。

    必须与 server 端 `normalizeFullwidthAlnum` 逐字相同：TTS 把 `Ａ` 转成 `A` 送合成，
    参考文本若仍是 `Ａ`，`Ａ`.lower() 得到 `ａ` 而 `A`.lower() 得到 `a`，逐字不等。
    映射是 1:1，不改变长度，`norm_for_match_with_map` 的下标映射不受影响。
    """
    cp = ord(ch)
    if 0xFF10 <= cp <= 0xFF19 or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return chr(cp - _FULLWIDTH_ALNUM_OFFSET)
    if cp == 0x3000:
        return " "
    return ch


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
            half = _halfwidth_alnum(ch)
            if _is_match_char(half):
                lowered = half.lower()
                norm_chars.append(lowered)
                # lower() 可能产出 2 个码位（U+0130「İ」→「i」+ 组合点），
                # idx_map 必须按**输出码位**补，否则 norm 比 map 长、其后下标全错位。
                idx_map.extend([origin if anchor_first else min(origin + offset, last)] * len(lowered))
            offset += 1
    return "".join(norm_chars), idx_map


def norm_for_match(s, lang):
    return norm_for_match_with_map(s, lang)[0]


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
    parameters = {
        "channel_id": [0],
        "enable_itn": False,
        "language": lang,
        "enable_words": True,
    }
    # 空词表必须整个键都不传：传空对象会被上游判成参数错误。
    if vocabulary:
        parameters["vocabulary"] = vocabulary
    body = {
        "model": CLOUD_ASR_MODEL,
        "input": {"file_urls": [file_url]},
        "parameters": parameters,
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


def _fa_time(item, key):
    """读 FA 词条的时间字段。FA 输出可能是对象也可能是 dict，两种都接受。"""
    if isinstance(item, dict):
        return float(item[key])
    return float(getattr(item, key))


def _fa_text(item):
    """读 FA 词条的文本。与 `_fa_time` 同一条兼容约束：对象与 dict 都接受。"""
    if isinstance(item, dict):
        return item.get("text") or ""
    return getattr(item, "text", "") or ""


def _same_word_class(left, right):
    """两个相邻字符是否属于同一个「词单位」。

    只判数字与 ASCII 字母：这两类在 FA 的词表里会被并成一个词，从中间切开就是错锚。
    汉字、假名、西里尔等逐字成词，相邻两字本来就是两个单位，一律返回 False。
    """
    if left.isdigit() and right.isdigit():
        return True
    return left.isascii() and right.isascii() and left.isalpha() and right.isalpha()


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


def _build_chunks(anchors, full_norm_len, total_duration_ms):
    """按锚点切块。anchors 是已抽样的锚点，按参考位置升序。"""

    if not anchors:
        return []

    pad_ms = int(FA_CHUNK_PAD_SEC * 1000)

    def _window(audio_start_ms, audio_end_ms):
        return (
            max(0, audio_start_ms - pad_ms),
            min(total_duration_ms, audio_end_ms + pad_ms),
        )

    chunks = []
    idx = 0

    first = anchors[0]
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

    for i in range(len(anchors) - 1):
        cur = anchors[i]
        nxt = anchors[i + 1]
        audio_start_ms, audio_end_ms = _window(cur["beginTime"], nxt["endTime"])
        chunks.append({
            "index": idx,
            "textStart": cur["offsetInFullNorm"],
            "textEnd": nxt["offsetInFullNorm"] + len(nxt["normText"]),
            "audioStartMs": audio_start_ms,
            "audioEndMs": audio_end_ms,
        })
        idx += 1

    last = anchors[-1]
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


def norm_span_to_original_span(norm_to_orig, norm_start, norm_end, original_len):
    """把 norm 下标区间映射成原文下标区间（左闭右开）。

    映射必须来自 `norm_for_match_with_map`：它的数字改写段整段锚定到原文起点，
    故下标不随改写长度漂移。禁止用「改写后文本的下标 = 原文下标」的映射 ——
    `_cn_number_to_digits` 会把「十」改写成「10」等，每个改写点都会让后续位置
    整体右移，FA 参考文本因此越过 chunk 音频窗口（2026-10-18 第 44 章：漂移 12 字，
    FA 末词 439.39s 超出 436.68s 窗口）。
    """
    if not norm_to_orig:
        return 0, original_len
    start = norm_to_orig[norm_start] if 0 <= norm_start < len(norm_to_orig) else 0
    # 区间右端取**下一个 norm 字**的原文下标，不是最后一个字的 +1：
    # 数字改写段（「十二」→「12」）每个 norm 字都锚到段首，+1 会把区间截在段内，
    # 「十二个士兵」的 norm[0,2) 只能拿回「十」，丢掉「二」，FA 参考文本因此变短。
    if 0 <= norm_end < len(norm_to_orig):
        end = norm_to_orig[norm_end]
    elif norm_to_orig:
        # 区间走到 norm 末尾：停在最后一个参与匹配的字之后，末尾标点不计入
        end = norm_to_orig[-1] + 1
    else:
        end = original_len
    return start, end


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


GAP_KIND_LABEL = {"leading": "开头", "internal": "中间", "trailing": "结尾"}


def coverage_gap_verdict(missing_chars, physical_duration_ms, asr_end_time_ms, lang, gap_kinds=()):
    """用白话给出「缺这段字」的成因判断，不出现内部指标名。

    尾部未转写时长只能解释**结尾**缺段。开头/中间缺段不能拿它推断成因：
    第 8 章是中间缺 240 字，尾部只差 0.21s，旧文案却把原因写成「缺尾段」，
    把排查方向引到音频结尾（2026-10-18 实测：实际是 TTS 在中间复读、吞掉了正文）。

    `gap_kinds` 是 offenders 的缺失类别（leading / internal / trailing）。
    """
    if missing_chars <= 0:
        return "没有缺失。"
    uncovered_tail_sec = (physical_duration_ms - asr_end_time_ms) / 1000.0
    if uncovered_tail_sec < 0:
        return (
            f"ASR 最后一个字比音频结尾还晚 {-uncovered_tail_sec:.2f}s，时间轴本身矛盾。"
            "先查 ASR 返回的逐字时间。"
        )
    kinds = {kind for kind in gap_kinds if kind}
    if kinds == {"trailing"}:
        expected_tail_sec = missing_chars / TEXT_RATE_BY_LANG[lang]
        if uncovered_tail_sec <= COVERAGE_TAIL_SILENCE_MS / 1000.0:
            return (
                f"ASR 已经转写到音频结尾（最后 {uncovered_tail_sec:.2f}s 没有字），没有提前截断。\n"
                "缺的文字最可能不在音频里：先查上游 TTS 产物是否漏读。"
            )
        if uncovered_tail_sec >= expected_tail_sec * 0.5:
            return (
                f"音频结尾有 {uncovered_tail_sec:.2f}s 没有转写；缺的 {missing_chars} 字按语速约需 "
                f"{expected_tail_sec:.2f}s，时长对得上。\n"
                "最可能是 ASR 截断或漏转，音频可能完整。"
            )
        return (
            f"音频结尾有 {uncovered_tail_sec:.2f}s 没有转写，但缺的 {missing_chars} 字按语速只需 "
            f"{expected_tail_sec:.2f}s，多出的静音无法解释。\n"
            "参考文本可能比音频长（多了尾注或未合成内容）。"
        )

    # 开头/中间缺段：结尾还剩多少秒说明不了成因
    if uncovered_tail_sec <= COVERAGE_TAIL_SILENCE_MS / 1000.0:
        return (
            f"ASR 转写一直覆盖到音频结尾（最后 {uncovered_tail_sec:.2f}s 没有字），没有提前截断。\n"
            f"缺的 {missing_chars} 字没有出现在转写里。先听音频这一段，确认是否漏读、复读或改读。\n"
            "音频确认无误后，再查 ASR。"
        )
    return (
        f"音频结尾还有 {uncovered_tail_sec:.2f}s 没有转写。缺的是开头或中间的文字，"
        "结尾还剩多少秒不能解释它。\n"
        "先对照音频核对这一段。"
    )


def format_coverage_diagnosis(
    coverage, asr_norm_text,
    physical_duration_ms, asr_end_time_ms, lang,
):
    """覆盖率校验的人读诊断，日志与异常共用这一份文本，禁止两处措辞分叉。

    缺失区段的 `text` 已在 coverage 内由调用方的 describe 回调渲染成原文片段。
    文案面向调用方（不懂服务内部指标）：不出现 refLen / matched / refNorm / 流尾 等词。
    """
    offenders = coverage.get("offenders", [])
    missing_chars = sum(item["length"] for item in offenders)
    gap_kinds = [item.get("kind") for item in offenders if item.get("kind")]
    ref_len = coverage.get("referenceLength", 0)
    asr_len = coverage.get("asrLength", 0)
    # 两个提前返回分支的 offenders 为空，不能走「缺了 N 字」的模板
    if ref_len == 0:
        return "参考文本为空，没有可对齐的文字。"
    if asr_len == 0:
        return (
            f"ASR 没有转写出任何文字（参考文本 {ref_len} 个有效字）。\n"
            "音频可能没有语音，或 ASR 服务异常；先查 ASR 返回结果。"
        )
    matched = coverage.get("matchedLength", 0)
    ratio = coverage.get("coverageRatio", 0.0)
    allowed = coverage.get("allowedGap", 0)
    kinds_text = "、".join(dict.fromkeys(GAP_KIND_LABEL.get(kind, kind) for kind in gap_kinds))
    lines = [
        f"音频与参考文本对不上：{kinds_text}缺了 {missing_chars} 字。",
        f"参考文本 {ref_len} 个有效字；ASR 转写 {asr_len} 个有效字，能对上 {matched} 个（{ratio:.1%}）。",
        "「有效字」= 去掉标点、空格，并把中文数字转成阿拉伯数字后的字。",
        f"单段缺失超过 {allowed} 字才判失败。",
    ]
    for item in offenders:
        label = GAP_KIND_LABEL.get(item["kind"], item["kind"])
        start_number = item["start"] + 1  # 内部下标从 0 起，展示改成从 1 起
        clip_note = "（该段较长，下面只显示首尾各 80 字）" if item["length"] > DIAG_SNIPPET_LIMIT else ""
        lines.append(
            f"缺失位置：{label}，从第 {start_number} 个有效字到第 {item['end']} 个，共 {item['length']} 字。"
        )
        lines.append(f"这一段原文{clip_note}：「{item['text']}」")
    lines.append(
        f"音频时长 {physical_duration_ms / 1000.0:.2f}s；"
        f"ASR 最后一个字在 {asr_end_time_ms / 1000.0:.2f}s。"
    )
    lines.append(
        "判断："
        + coverage_gap_verdict(missing_chars, physical_duration_ms, asr_end_time_ms, lang, gap_kinds)
    )
    if asr_norm_text:
        lines.append(
            f"ASR 转写结尾 {DIAG_SNIPPET_LIMIT} 个有效字（对照用）："
            f"「{_clip_text(asr_norm_text, DIAG_SNIPPET_LIMIT, from_tail=True)}」"
        )
    return "\n".join(lines)


class CloudHotwordAlignPipeline:
    """Cloud ASR (hotword-anchored pathfinding) + local FA alignment pipeline.

    Flow: select hotword terms from reference text -> upload audio to OSS ->
    cloud ASR with vocabulary -> match term occurrences by LCS -> sample anchors ->
    build chunks (or fixed-window fallback) -> FA alignment per chunk.
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

        # 选词、锚点、切块、FA 参考文本必须共用这一份归一化文本与下标映射。
        # 数字改写段（十→10）整段锚定到原文起点，混用两套映射会让后续位置整体漂移。
        reference_norm, ref_idx_map = norm_for_match_with_map(clean_reference_text, run_lang)
        terms = select_hotword_terms(
            clean_reference_text,
            run_lang,
            normalize=norm_for_match,
            norm_text=reference_norm,
            norm_map=ref_idx_map,
        )
        vocabulary = {term.text: HOTWORD_WEIGHT for term in terms}
        log_info(
            f"热词提取: {len(terms)} 个"
            + (
                " | " + " | ".join(f"{term.text}({term.count})" for term in terms)
                if terms
                else "（空词表，不传 vocabulary）"
            ),
            req_id=req_id,
        )

        file_url = _upload_oss(audio_path, req_id)
        task_id = _submit_cloud_asr(file_url, run_lang, vocabulary, req_id)
        transcription = _poll_cloud_asr(task_id, req_id)

        words = _extract_words(transcription)
        asr_text = _transcript_text(transcription)
        asr_end_time_ms = words[-1].get("end_time", 0) if words else 0

        # 音频时长必须在覆盖率校验之前算：它是区分「音频缺料」与「ASR 漏转」的唯一判据，
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

        # 覆盖率诊断与 FA 参考文本共用上面那一份映射。数字改写段（十→10、两百→200）
        # 整段锚定到原文起点，故 norm 下标回映射原文不漂移；若另用「改写后文本下标 =
        # 原文下标」的映射，每次改写都会让后续位置整体右移（2026-10-18 第 44 章：
        # 漂移 12 字 → FA 参考文本越过音频窗口 3.1s，被判成伪对齐）。
        asr_norm, char_to_word = build_asr_stream(words, run_lang, norm_for_match)
        coverage = assess_reference_coverage(
            clean_reference_text,
            words,
            run_lang,
            norm_for_match,
            describe=lambda start, end: describe_reference_region(
                clean_reference_text, ref_idx_map, start, end
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
            # 异常文案直接用人读诊断：不拼内部指标名与 reason code，
            # 调用方（WebUI / 节点错误面板）读到的就是完整解释。
            raise ValueError(coverage_diagnosis)

        log_info(
            f"ASR 文本覆盖 OK | 覆盖率={coverage['coverageRatio']:.1%} | "
            f"中间缺失={coverage['internalGap']} 字 | 单段容差={coverage['allowedGap']} 字",
            req_id=req_id,
        )

        full_norm = reference_norm
        anchors, term_stats, dropped_anchors = match_hotword_anchors(
            [term.text for term in terms],
            reference_norm,
            asr_norm,
            char_to_word,
            words,
        )
        log_block(
            ["━━━ 热词匹配诊断 ━━━"]
            + [
                f"  {stat.text}: 参考 {stat.reference_count} | ASR {stat.asr_count} | "
                f"命中 {stat.matched}"
                for stat in term_stats
            ]
            + [f"  锚点合计: {len(anchors)}"]
            + (
                [f"  时间回退丢弃: {len(dropped_anchors)} 个"]
                if dropped_anchors
                else []
            ),
            req_id=req_id,
            level="INFO",
        )

        sampled = []
        if len(anchors) < 2:
            fallback_anchor = anchors[0] if anchors else None
            reason = "没有热词锚点" if fallback_anchor is None else "只有 1 个热词锚点"
            offset_lookup = build_reference_offset_lookup(
                reference_norm, asr_norm, char_to_word, words
            )
            chunks = build_fallback_chunks(
                clean_reference_text,
                int(physical_duration_ms),
                text_offset_for_time=lambda ms: ref_idx_map[
                    min(offset_lookup(ms), len(ref_idx_map) - 1)
                ],
                anchor_ms=fallback_anchor["beginTime"] if fallback_anchor else None,
                anchor_text_offset=(
                    ref_idx_map[fallback_anchor["offsetInFullNorm"]]
                    if fallback_anchor
                    else None
                ),
            )
            log_warn(
                f"{reason}，改用固定窗口回退 | 文本={len(clean_reference_text)}字 | "
                f"音频={physical_audio_duration:.3f}s | 切块={len(chunks)} | "
                f"文本边界按 ASR 词时间映射",
                req_id=req_id,
            )
        else:
            sampled = sample_anchors(anchors)
            anchor_gaps = [
                current["beginTime"] - previous["endTime"]
                for previous, current in zip(sampled, sampled[1:])
            ]
            max_gap = max(anchor_gaps) if anchor_gaps else 0
            head_gap = max(0, sampled[0]["beginTime"])
            tail_gap = max(0, int(physical_duration_ms) - sampled[-1]["endTime"])
            log_info(
                f"锚点抽样: {len(anchors)} -> {len(sampled)} | "
                f"距开头={head_gap / 1000:.2f}s | 最大间隔={max_gap / 1000:.2f}s | "
                f"距尾部={tail_gap / 1000:.2f}s",
                req_id=req_id,
            )
            if (
                head_gap > ANCHOR_GAP_WARN_MS
                or max_gap > ANCHOR_GAP_WARN_MS
                or tail_gap > ANCHOR_GAP_WARN_MS
            ):
                log_warn(
                    f"锚点间隔过大 | 距开头={head_gap / 1000:.2f}s | "
                    f"最大间隔={max_gap / 1000:.2f}s | "
                    f"距尾部={tail_gap / 1000:.2f}s | "
                    f"上限={ANCHOR_GAP_WARN_MS / 1000:.0f}s | "
                    f"超过 {FA_MAX_SEC}s 的块会被截断",
                    req_id=req_id,
                )

            # The ASR last-word timestamp can omit trailing audio. The decoded
            # file duration is the authoritative end boundary for the tail chunk.
            # If the last matched anchor is too late, the remaining reference text
            # can be much longer than the tail audio. Qwen FA extrapolates timestamps
            # for this case instead of clamping them to the supplied audio window.
            effective_valid = list(sampled)
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
            raise ValueError("切块结果为空，无法进行 FA 对齐。")

        verbose_json = {
            "task": "transcribe",
            "language": run_lang,
            "duration": round(physical_audio_duration, 2),
            "report": {},
            "text": "",
            "words": [],
        }

        text_parts = []
        all_word_alignments = []
        fa_diag = []
        emitted_orig_end = 0
        chunk_records = []
        word_origins = []
        last_emitted = None
        seam_clamp_count = 0

        # ── 第一趟：逐块 FA，只算不判 ─────────────────────────────
        # 修复判据要用「本次请求内未判坏的块」实时标定参考语速，而参考语速只有在
        # 所有块都对齐完才齐备。故 FA 与输出拆成两趟：第一趟只产词流与结构诊断，
        # 第二趟才决定均分并落流。
        for chunk in chunks:
            audio_start_sec = chunk["audioStartMs"] / 1000.0
            audio_end_sec = chunk["audioEndMs"] / 1000.0

            if "origStart" in chunk:
                orig_start = chunk["origStart"]
                orig_end = chunk["origEnd"]
            else:
                orig_start, orig_end = norm_span_to_original_span(
                    ref_idx_map,
                    chunk["textStart"],
                    chunk["textEnd"],
                    len(clean_reference_text),
                )

            ground_truth = clean_reference_text[orig_start:orig_end]
            if not ground_truth.strip():
                continue

            chunk_dur = audio_end_sec - audio_start_sec
            if chunk["index"] == chunks[-1]["index"]:
                log_info(
                    f"FA 尾部输入 | chunk[{chunk['index']}] | "
                    f"音频={audio_start_sec:.3f}..{audio_end_sec:.3f}s ({chunk_dur:.3f}s) | "
                    f"原文={orig_start}:{orig_end} ({len(ground_truth)}字)",
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
            # rebuild 用压平后的一份；健康度评分仍用 FA 原始输出，评的是模型本身。
            disjoint_fa_res, fa_clamped, fa_max_overlap = normalize_fa_interval_overlaps(
                fa_res, audio_start_sec, audio_end_sec
            )
            if fa_clamped:
                log_warn(
                    f"FA 区间重叠压平 | chunk[{chunk['index']}] | 压平 {fa_clamped} 个词条 | "
                    f"最大压平 {fa_max_overlap * 1000:.0f}ms | "
                    f"逐词估边共享时间是正常现象，压平后标点补位才落在后词起点之前",
                    req_id=req_id,
                )

            for w in fa_res:
                all_word_alignments.append(w)

            rebuilt_words, rebuild_stats = self._rebuild_words(
                ground_truth, disjoint_fa_res, run_lang, chunk_dur
            )
            overlap_end = max(emitted_orig_end, orig_start)
            keep_from = max(0, overlap_end - orig_start)
            kept_words = self._trim_rebuilt_words(
                ground_truth, rebuilt_words, keep_from, run_lang
            )
            # 响应文本改由「真正输出的 token」拼（见下方 `text_parts`）。
            # 不再从 `fa_res` 反推：删掉重叠区要按**词**跳、而保留是按**字**判，
            # 两种粒度混用会在接缝处重复或漏字（2026-10-27 审查 M1）。
            emitted_orig_end = max(emitted_orig_end, orig_end)

            verdict, valid_token_count = self._diagnose_chunk(
                ground_truth, rebuilt_words, disjoint_fa_res, chunk_dur, run_lang,
                rebuild_stats["unanchored"],
            )
            chunk_records.append({
                "index": chunk["index"],
                # 绝对秒（块起点 + 块内相对）：接缝上限要跨块比较，存相对值会混用两套坐标。
                "first_kept_start_abs": (audio_start_sec + kept_words[0]["start"]) if kept_words else None,
                "origin": chunk_origin,
                "audio_start": audio_start_sec,
                "duration": chunk_dur,
                "ground_truth": ground_truth,
                "rebuilt": rebuilt_words,
                "kept": kept_words,
                "keep_from": keep_from,
                "verdict": verdict,
                "valid_tokens": valid_token_count,
            })

            fa_diag.append(
                f"chunk[{chunk['index']}] {audio_start_sec:.1f}-{audio_end_sec:.1f}s ({chunk_dur:.1f}s) "
                f"原文[{orig_start}:{orig_end}]={orig_end - orig_start}字 FA词数={len(fa_res)}"
                f" 已锚={rebuild_stats['located_words']} 未锚字={rebuild_stats['unanchored']}"
                + (f" ⚠️{verdict}" if verdict else "")
            )

        # ── 第二趟：判缺陷、均分坏块、按序落流 ────────────────────
        align_defect, repaired_indexes = self._decide_chunk_repairs(chunk_records, req_id)
        for record_index, record in enumerate(chunk_records):
            kept_words = record["kept"]
            if record["index"] in repaired_indexes:
                # 均分起点取前一块已输出的最大 end：重叠区由前块输出，均分必须从它之后开始，
                # 否则接缝处直接时间回退（见 `_repair_chunk_words` 的说明）。
                prev_max_end = last_emitted[2] if last_emitted is not None else None
                start_sec = 0.0 if prev_max_end is None else max(0.0, prev_max_end - record["audio_start"])
                # 均分终点取下一个块首个保留词的起点：不越过下一个块的估计，接缝才不会倒流。
                next_record = chunk_records[record_index + 1] if record_index + 1 < len(chunk_records) else None
                next_first_abs = next_record["first_kept_start_abs"] if next_record is not None else None
                end_sec = None if next_first_abs is None else max(0.0, next_first_abs - record["audio_start"])
                repaired_words = self._repair_chunk_words(
                    record["ground_truth"], record["duration"], run_lang,
                    keep_from=record["keep_from"], start_sec=start_sec, end_sec=end_sec,
                )
                if repaired_words is None:
                    # 判定与摊平前提不同（判定只看块窗口，摊平还要扣掉前块已输出与下一块保留词
                    # 之间的可用窗口），窗口被挤没时摊不平。按 `zero-valid-token` 判缺陷并把
                    # 原流留下 —— 崩掉会把「一个小块摊不平」升级成整章无产物。
                    message = (
                        f"chunk[{record['index']}] 可用窗口不足，无法摊平"
                        f"（有效={record['valid_tokens']} | 时长={record['duration']:.3f}s | "
                        f"起点={start_sec:.3f}s | 已跳过={record['keep_from']}字）"
                    )
                    log_warn(f"[块不修] {message}", req_id=req_id)
                    if align_defect is None:
                        align_defect = {"reason": "zero-valid-token", "message": message}
                else:
                    kept_words = repaired_words

            for w in kept_words:
                start = round(record["audio_start"] + w["start"], 3)
                end = round(record["audio_start"] + w["end"], 3)
                start, end, clamped = self._enforce_seam_monotonicity(
                    w["text"], start, end, record["origin"], last_emitted, req_id
                )
                if clamped:
                    seam_clamp_count += 1
                verbose_json["words"].append({
                    "word": w["text"],
                    "start": start,
                    "end": end,
                })
                text_parts.append(w["text"])
                word_origins.append(record["origin"])
                # 已输出最大 end 必须随流累积，不能只看上一项：标点被抬高的 end
                # 要压住紧随其后的实词起点。
                max_emitted_end = end if last_emitted is None else max(last_emitted[2], end)
                last_emitted = (start, end, max_emitted_end, w["text"], record["origin"])

        # 响应文本 = 真正输出的 token 顺序拼接。空字符输出零个 token，所以 "" 拼接
        # 对中英都成立（英文的空格本身就是一个 token），不再按语言选连接符。
        verbose_json["text"] = "".join(text_parts)

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
            # 对齐自述：**只放缺陷标记**，块的编号、数量、语速、容差全部留在本文件的
            # 日志里。客户端读到 `defect` 就判本章音频缺陷并回退重产音频；没有该键
            # 即表示本次对齐可用。客户端不认识块结构，也不参与容差判定。
            "align": {"defect": align_defect} if align_defect else {},
        }

        total_pipeline_time = time.time() - pipeline_start_time
        task_info = get_task_info(req_id)
        file_name = task_info.get("file", "(未知)") if task_info else "(未知)"

        log_block([
            "📊 最终报告 (Cloud Hotword 对齐模式)",
            f"文件名       : {file_name}",
            f"执行状态     : ✅ 成功 (相似度: {verbose_json['report']['accuracy_ratio']})",
            f"热词         : {len(terms)} 个 | 锚点: {len(anchors)} | 抽样: {len(sampled)} | 切块: {len(chunks)}",
            f"音频时长     : {physical_audio_duration:.2f}s",
            f"流水线耗时   : {total_pipeline_time:.2f}s",
            f"RTF          : {total_pipeline_time / physical_audio_duration:.3f}" if physical_audio_duration > 0 else "RTF: N/A",
        ] + (
            [f"对齐缺陷     : {align_defect['reason']} | {align_defect['message']}"]
            if align_defect else []
        ) + fa_diag, req_id=req_id)

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
        """强制跨 chunk 输出时间单调，并保证相邻词条的时间区间不重叠。

        只用于热词分块链路：只有这条链路按热词锚点切出重叠文本、把同一段对齐两次。
        pipeline 的本地 ASR+FA 与纯 FA 按游标切文本、每段只对齐一次，不调用本方法。

        重叠区由前一个 chunk 输出，但紧随其后的词（典型为标点，继承当前块 FA 的前词
        end）来自当前 chunk 的对齐，两者存在毫秒级差异，接缝处可能出现小幅时间回退。

        不变式是「每个词条的 start >= 之前所有已输出词条的 end」。只比前一项的
        start/end 不够：标点补位会把前一项的 end 抬到它自己的 start 之后，紧随其后的
        实词起点仍可能落在该 end 之前。此时点态单调成立、区间却重叠，下游按
        「前词 end 补位」重建标点时就会变成时间回退（2026-10-17 第 4 章事故：
        的 42.580..42.740 来自 chunk[1]，士 42.700..43.180 来自 chunk[2]）。

        漂移 <= SEAM_JITTER_MAX_SEC 视为正常抖动，钳制到已输出最大 end 并 WARN；
        超过阈值判定为锚点错位/FA 漂移，Fail-Fast 抛错。
        last_emitted = (start, end, 已输出最大 end, text, origin)。
        返回 (start, end, clamped)。
        """
        if last_emitted is None:
            return start, end, False
        p_start, p_end, p_max_end, p_word, p_origin = last_emitted

        new_start, new_end, overlap, regression = clamp_seam_interval(
            p_start, p_end, p_max_end, start, end
        )
        if overlap <= 0 and regression <= 0:
            return start, end, False

        drift = max(overlap, regression)
        if drift > SEAM_JITTER_MAX_SEC:
            message = (
                f"❌ 跨 chunk 接缝时间戳回退超容差 | 起点重叠={overlap:.3f}s | "
                f"点态回退={regression:.3f}s (容差={SEAM_JITTER_MAX_SEC}s) | "
                f"当前'{word_text}' {start:.3f}..{end:.3f}s 来源={origin} | "
                f"前项'{p_word}' {p_start:.3f}..{p_end:.3f}s，此前最大 end={p_max_end:.3f}s "
                f"来源={p_origin} | "
                f"可能原因: 热词锚点(beginTime/endTime)错位或该段 FA 对齐漂移"
            )
            log_error(message, req_id=req_id)
            raise FaTimestampError(message)

        log_warn(
            f"块接缝时间戳钳制 | 起点重叠={overlap:.3f}s | 点态回退={regression:.3f}s | "
            f"'{word_text}' {start:.3f}..{end:.3f}s -> {new_start:.3f}..{new_end:.3f}s | "
            f"前项'{p_word}' {p_start:.3f}..{p_end:.3f}s，此前最大 end={p_max_end:.3f}s "
            f"来源={p_origin} | 当前来源={origin}",
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

    def _tokenize_reference(self, ground_truth, lang):
        r"""按语言切参考文本 token，返回 (tokens, is_valid_token)。

        token 切分是**唯一来源**：rebuild、trim、块诊断、均分四处必须用同一套切法，
        否则「有效 token 数」在三处各有各的答案，判据互相对不上。

        切法（与客户端 `repairWordTimestamps` 同构，改一侧必须同批改另一侧）：
        - zh：逐字符（CJK 一字一 token）
        - 其他：`\w+|[^\w]`，单词 + 单个标点

        「有效」= 可发音，用 Unicode 类别判定，不靠 ASCII 白名单：任何脚本的字母/数字
        都算（`isalnum` 覆盖 CJK、假名、西里尔），其余（标点、空白、符号）不算 ——
        标点不占时间，继承前一个有效 token 的 end。
        """
        if lang == "zh":
            tokens = list(ground_truth)

            def is_valid_token(tok):
                return bool(tok) and (tok.isalnum() or unicodedata.category(tok).startswith("L"))
        else:
            tokens = re.findall(r'\w+|[^\w]', ground_truth, re.UNICODE)

            def is_valid_token(tok):
                return bool(tok) and (tok[0].isalnum() or unicodedata.category(tok[0]).startswith("L"))

        return tokens, is_valid_token

    def _diagnose_chunk(self, ground_truth, rebuilt_words, fa_words, chunk_duration, lang, unanchored):
        """结构判定：本块的词流能否直接用于字幕。返回 (原因或 None, 有效 token 数)。

        只判**结构**，不判语速（语速要等全部块对齐完才能标定，见 `_decide_chunk_repairs`）。
        三条判据都只看有效 token，纯标点段的零时长是正常形态，不参与判定：

        1. 未锚定有效字符达到 `CHUNK_ZERO_RUN_MIN` 个 —— FA 没给出这些字的词，
           它们的时间只能由缺口夹估算，块内时间轴不可信。
           **不再用「字符数 vs FA 词数」**：FA 的输出单位是词，一串阿拉伯数字是
           一个词，拿字符数比词数会把数字块全判坏（2026-10-27 事故的直接原因）。
        2. 零时长有效 token 的**连续段**达到 `CHUNK_ZERO_RUN_MIN` 个 —— 块内时间轴断了。
           零星零时长是 FA 逐字估边的常态，不判坏（见该常量的实测说明）。
        3. 词流没覆盖块窗口（首词距块起点、末词距块终点超过容差）—— 这段音频的发音
           没被 FA 消费，块内必然存在零时长区段。
        """
        tokens, is_valid_token = self._tokenize_reference(ground_truth, lang)
        valid_count = sum(1 for tok in tokens if is_valid_token(tok))
        if valid_count == 0:
            # 整块没有可发音 token：没有时间可摊，也没有零时长有效 token 可言。
            return None, 0

        # 未锚定的字只能靠前后夹，数量多说明 FA 确实没对上这一段。
        if unanchored >= CHUNK_ZERO_RUN_MIN:
            return f"unanchored({unanchored})", valid_count

        # FA 一个词都没给、这块却有声：整块时间只能靠缺口夹铺满，等于没有对齐。
        # 不能等 `unanchored >= 5` 兜住 —— 短块（块尾 2~4 个字）会静默放行。
        if not fa_words:
            return f"unanchored({valid_count})", valid_count

        # 成段的零时长有效 token：FA 对这整段没给出对齐区间，块内时间轴断了。
        # 零星零时长不算（见 CHUNK_ZERO_RUN_MIN 的说明）。
        run = 0
        longest_run = 0
        for token, word in zip(tokens, rebuilt_words):
            if not is_valid_token(token):
                continue
            if word["start"] == word["end"]:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0
        if longest_run >= CHUNK_ZERO_RUN_MIN:
            return f"zero-run({longest_run})", valid_count

        if fa_words:
            first_start = min(_fa_time(w, "start_time") for w in fa_words)
            last_end = max(_fa_time(w, "end_time") for w in fa_words)
            if first_start > CHUNK_COVERAGE_TOLERANCE_SEC:
                return f"head-gap({first_start:.3f}s)", valid_count
            if chunk_duration - last_end > CHUNK_COVERAGE_TOLERANCE_SEC:
                return f"tail-gap({chunk_duration - last_end:.3f}s)", valid_count

        return None, valid_count

    def _decide_chunk_repairs(self, records, req_id):
        """判定坏块能否均分修复，返回 (defect 或 None, 需均分的块号集合)。

        参考语速**在本次请求内实时标定**，不落盘、不预标定、不跨请求复用：

            R = Σ(未判坏块的有效 token 数) / Σ(未判坏块的音频时长)

        一次请求就是一个章节的一种语言，故 R 天然带语言与音色，跨语言共用同一组
        相对常量（`CHUNK_RATE_BAND_LOW/HIGH`）。禁止改成绝对 token/秒阈值。

        处置顺序体现优先级：越界说明音频与文本真的对不上，不能靠摊平掩盖；
        能摊的块超过上限则整章结果不可信。

        多语言：判据只比相对比例 r/R，禁止引入绝对 token/秒阈值。
        **改本函数前先读 README「多语言约束」一节。**
        """
        bad_records = [r for r in records if r["verdict"] is not None]
        if not bad_records:
            return None, set()

        good_records = [r for r in records if r["verdict"] is None]
        good_valid = sum(r["valid_tokens"] for r in good_records)
        good_duration = sum(r["duration"] for r in good_records)
        chapter_rate = (good_valid / good_duration) if good_duration > 0 else None

        candidates = []
        if chapter_rate is None:
            candidates.append((
                "no-good-chunk",
                f"本次请求 {len(records)} 块全部判坏，没有可用参考块，无法标定语速",
            ))

        repaired = set()
        for record in bad_records:
            valid = record["valid_tokens"]
            duration = record["duration"]
            index = record["index"]
            window = f"{record['audio_start']:.1f}-{record['audio_start'] + duration:.1f}s"
            # 结构原因（`record["verdict"]`）必须一路带到最后：一个块可以同时「少词」和
            # 「语速越界」，只报后者会把「FA 少给了 5 个词」这条事实从日志里抹掉，
            # 而它才是块坏的直接证据。下面每条处置都先写结构原因，再写本次处置的理由。
            structural = f"结构={record['verdict']}"

            if valid == 0:
                reason = f"chunk[{index}] {window} | {structural} | 无有效 token，无法按原文摊平"
                candidates.append(("zero-valid-token", reason))
                log_warn(f"[块不修] {reason}", req_id=req_id)
                continue
            step = duration / valid if valid else 0.0
            if step < CHUNK_TOKEN_MIN_SPAN_SEC:
                reason = (
                    f"chunk[{index}] {window} | {structural} | "
                    f"均分步长 {step * 1000:.3f}ms < {CHUNK_TOKEN_MIN_SPAN_SEC * 1000:.0f}ms"
                    f"（{valid} token / {duration:.3f}s）"
                )
                candidates.append(("zero-valid-token", reason))
                log_warn(f"[块不修] {reason}", req_id=req_id)
                continue
            if chapter_rate is None:
                log_warn(
                    f"[块不修] chunk[{index}] {window} | {structural} | "
                    f"没有可用参考块，无从判断均分是否合理",
                    req_id=req_id,
                )
                continue
            rate = valid / duration if duration > 0 else 0.0
            ratio = rate / chapter_rate
            if ratio < CHUNK_RATE_BAND_LOW or ratio > CHUNK_RATE_BAND_HIGH:
                reason = (
                    f"chunk[{index}] {window} | {structural} | "
                    f"语速比 r/R={ratio:.2f} 越界 [{CHUNK_RATE_BAND_LOW}, {CHUNK_RATE_BAND_HIGH}]"
                    f"（r={rate:.2f} token/s，R={chapter_rate:.2f} token/s）"
                )
                candidates.append(("rate-out-of-band", reason))
                log_warn(f"[块不修] {reason}", req_id=req_id)
                continue
            repaired.add(index)
            log_warn(
                f"[块修复] chunk[{index}] {window} | {structural} | 有效={valid} | "
                f"步长={step:.3f}s | r/R={ratio:.2f}",
                req_id=req_id,
            )

        if len(repaired) > MAX_REPAIRED_CHUNKS_PER_REQUEST:
            candidates.append((
                "too-many-repaired",
                f"已修复块={len(repaired)} 上限={MAX_REPAIRED_CHUNKS_PER_REQUEST} "
                f"（{', '.join(f'chunk[{i}]' for i in sorted(repaired))}）",
            ))

        # 汇总每请求都记：判缺陷时它是上下文，不判缺陷时它是「本轮修了几个块」的唯一收口。
        log_info(
            f"[块修复汇总] 块数={len(records)} | 参考块={len(good_records)} | "
            f"坏块={len(bad_records)} | 均分块={len(repaired)} | 章参考语速="
            + (f"{chapter_rate:.2f}token/s" if chapter_rate is not None else "N/A"),
            req_id=req_id,
        )

        if not candidates:
            return None, repaired

        priority = ["no-good-chunk", "rate-out-of-band", "zero-valid-token", "too-many-repaired"]
        candidates.sort(key=lambda item: priority.index(item[0]))
        reason = candidates[0][0]
        message = " | ".join(detail for _, detail in candidates)
        structural_digest = " | ".join(
            f"chunk[{record['index']}]={record['verdict']}" for record in bad_records
        )
        log_warn(f"[块缺陷] reason={reason} | {message}", req_id=req_id)
        log_warn(f"[块结构原因] 坏块 {len(bad_records)} 个 | {structural_digest}", req_id=req_id)
        return {"reason": reason, "message": message}, repaired

    def _repair_chunk_words(self, ground_truth, chunk_duration, lang, keep_from=0.0, start_sec=0.0, end_sec=None):
        r"""把块窗口按**有效 token 长度**加权均分，只输出**本次真正要输出**的那段词条。

        只在结构判定为坏块、且语速落在容差带内时调用（判定在 `_decide_chunk_repairs`）。

        两个入参缺一不可：

        - `keep_from`：块内已被前一个块输出过的字符数。热词分块的相邻窗口文本互相重叠，
          重叠区由前一个块输出，本块必须从它之后才开始（与 `_trim_rebuilt_words` 同一条规则：
          token 末尾落在 `keep_from` 之前即丢）。
        - `end_sec`：本块可用时间窗的终点（块内相对秒）。传下一个块首个保留词的起点 ——
          均分若一直铺到块窗口终点，本块尾部就会推到下一个块的估计之后，接缝处直接变成
          时间回退（2026-10-27 实测：ch8 chunk[2] 尾部铺到 46.86s，而 chunk[3] 的首个
          保留词在 46.22s，回退 0.64s 越过 0.5s 容差，整章 Fail-Fast）。
        - `start_sec`：本块可用时间窗的起点（块内相对秒）。传前一个块已输出的最大 end，
          均分从那里开始 —— 否则均分起点会**早于**前块已输出的时间，接缝处立刻变成时间回退，
          被 `_enforce_seam_monotonicity` 判成锚点错位（2026-10-27 实测：单章验证时
          chunk[1] 的首个保留词落在 18.833s，而前块已输出到 19.700s，回退 0.867s 直接 Fail-Fast）。
          不靠事后钳制吸收：钳制只允许 0.5s 抖动，而整块均分能造成的回退远大于它。

        规则（多语言共用，见 README「多语言约束」）：
        - 权重 = token 字符数。中文逐字切分，每 token 恒为 1，退化为均分；英文按词切分，
          12 字母的词拿到的时长长于 2 字母的词 —— 均分不等于公平，长度才是可比的量。
        - 每个有效 token 至少 `CHUNK_TOKEN_MIN_SPAN_SEC`。
        - 标点不占时间：继承前一个有效 token 的 end（段首标点继承窗口起点）。
        - 文本一个字不改，只改时间；输出与 `_trim_rebuilt_words` 之后同形，直接进接缝钳制。

        无有效 token 或时间不够摊时返回 None，由调用方判缺陷。

        多语言：权重取 token 字符数，标点不占时间，切分走 `_tokenize_reference`。
        **改本函数前先读 README「多语言约束」一节。**
        """
        tokens, is_valid_token = self._tokenize_reference(ground_truth, lang)
        if chunk_duration <= 0:
            return None

        window_start = max(0.0, start_sec)
        window_end = chunk_duration if end_sec is None else min(chunk_duration, end_sec)
        if window_end <= window_start:
            # 上限比起点还早：说明下一个块的估计早于本块已输出的位置，
            # 退回到整块窗口，交给接缝钳制处置（它只吸收小抖动，大偏差仍会抛）。
            window_end = chunk_duration
        window = window_end - window_start
        if window <= 0:
            return None

        # 与 `_trim_rebuilt_words` 同一条保留规则：token 末尾落在 keep_from 之前即丢。
        # 前缀和而非常量重算：块内 token 可达数百，逐 token 重算会变成平方级热点。
        kept_indexes = []
        char_cursor = 0
        for index, token in enumerate(tokens):
            char_cursor += len(token)
            if char_cursor > keep_from:
                kept_indexes.append(index)
        if not kept_indexes:
            return []

        valid_indexes = [i for i in kept_indexes if is_valid_token(tokens[i])]
        if not valid_indexes:
            # 剩余部分全是标点：它们不占时间，一起塌到窗口起点。
            return [{"text": tokens[i], "start": window_start, "end": window_start} for i in kept_indexes]

        weights = [float(max(1, len(tokens[i]))) for i in valid_indexes]
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
        if window * (min(weights) / total_weight) < CHUNK_TOKEN_MIN_SPAN_SEC:
            return None

        words = []
        cursor = window_start
        schedule = iter(zip(valid_indexes, weights))
        upcoming = next(schedule, None)
        for index in kept_indexes:
            if upcoming is not None and index == upcoming[0]:
                span = window * upcoming[1] / total_weight
                start = cursor
                end = min(window_end, cursor + span)
                cursor = end
                words.append({"text": tokens[index], "start": start, "end": end})
                upcoming = next(schedule, None)
            else:
                words.append({"text": tokens[index], "start": cursor, "end": cursor})

        return words

    def _trim_rebuilt_words(self, ground_truth, rebuilt_words, keep_from, lang):
        """丢掉前一块已经输出过的重叠部分（按字符偏移，不按词）。"""
        tokens, is_valid_token = self._tokenize_reference(ground_truth, lang)

        kept = []
        cursor = 0
        for token, word in zip(tokens, rebuilt_words):
            token_end = cursor + len(token)
            if token_end > keep_from:
                kept.append(word)
            cursor = token_end
        return kept

    def _rebuild_words(self, ground_truth, fa_words, lang, chunk_duration=None):
        """把 FA 词流锚回原文，返回 (逐 token 时间, 诊断信息)。

        ## 为什么必须按词锚，不能按序号贴

        FA 的输出单位是**词**，不是字。`1500` 在 FA 词表里是一个词，参考文本按字切成
        四个 token。按序号硬贴时，该词之后每个 token 都错位一个词，块尾若干 token 拿不到
        词、落成零时长。2026-10-27 实测：三章 54 块里 15 块因此被判 `fa-short`，进而整章
        作废、回退重跑 tts —— 而音频本身是对的。

        ## 规则（与语言无关，见 README「多语言约束」）

        1. 每个 FA 词的文本就是原文的一段连续字符（FA 只回声，不改写）。顺序定位后，
           该词的时间按**有效字符数**摊给它覆盖的字符。
        2. 没被任何词覆盖的字符（FA 真丢字）夹在前后两个已锚定字符之间按字数摊。
           头部夹在块起点与首个锚点之间，尾部夹在末个锚点与块终点之间。
        3. 标点不占时间：继承前一个 token 的 end。
        4. 文本一个字不改，只改时间。

        `chunk_duration` 只用于给头尾缺口兜边界；缺省时退化为「贴着已有锚点」。

        返回的 `word_spans` 按 FA 词顺序给出每个词覆盖的原文 `[start, end)`，定位失败为
        `None` —— 调用方用它算「块首被前一块吃掉后该跳过几个词」。
        """
        tokens, is_valid_token = self._tokenize_reference(ground_truth, lang)

        def is_valid_char(ch):
            return bool(ch) and (ch.isalnum() or unicodedata.category(ch).startswith("L"))

        # 第一趟：逐词定位，把每个词的时间摊给它覆盖的字符
        char_time = [None] * len(ground_truth)
        word_spans = []
        cursor = 0
        located = 0
        for w in fa_words:
            text = _fa_text(w)
            span = self._locate_word(text, ground_truth, cursor, is_valid_char) if text else None
            word_spans.append(span)
            if span is None:
                continue
            cursor = span[1]
            located += 1
            positions = [i for i in range(span[0], span[1]) if is_valid_char(ground_truth[i])]
            if not positions:
                continue
            start = _fa_time(w, "start_time")
            end = max(start, _fa_time(w, "end_time"))
            step = (end - start) / len(positions)
            for k, index in enumerate(positions):
                char_time[index] = (start + step * k, start + step * (k + 1))

        # 第二趟：未锚定字符按「前后夹」摊开。
        #
        # 分组只看**锚点**，不看标点：同一对锚点之间的全部未锚定有效字符共用一个窗口。
        # 若按「连续有效段」分组，标点会把缺口切碎，前一段先把窗口铺到下一个锚点，
        # 后一段就只能从合成值铺到同一点 —— 整段塌成零时长。那正是本次要消灭的形态
        # （2026-10-27 审查反例：`甲乙，丙丁戊` 只锚住「戊」时，丙丁落成 (5.0, 5.0)）。
        prev_ends = [0.0] * len(ground_truth)
        last_end = 0.0
        for i in range(len(ground_truth)):
            if char_time[i] is not None:
                last_end = char_time[i][1]
            else:
                prev_ends[i] = last_end
        next_starts = [None] * len(ground_truth)
        next_start = chunk_duration
        for i in range(len(ground_truth) - 1, -1, -1):
            if char_time[i] is not None:
                next_start = char_time[i][0]
            else:
                next_starts[i] = next_start

        unanchored = 0
        groups = []
        for i in range(len(ground_truth)):
            if char_time[i] is not None or not is_valid_char(ground_truth[i]):
                continue
            key = (prev_ends[i], next_starts[i])
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
            groups[-1][1].append(i)

        for (prev_end, next_anchor), positions in groups:
            unanchored += len(positions)
            if next_anchor is not None:
                end = next_anchor
            elif chunk_duration is not None:
                end = chunk_duration
            else:
                end = prev_end
            if end < prev_end:
                # 窗口被挤没时只留点态，交给接缝钳制与零时长判据处置，不在这里抛。
                end = prev_end
            step = (end - prev_end) / len(positions)
            for k, index in enumerate(positions):
                char_time[index] = (prev_end + step * k, prev_end + step * (k + 1))

        # 第三趟：按 token 输出。一个 token 覆盖的字符可能跨越多个词（英文按词切时）。
        rebuilt = []
        prev_end = 0.0
        offset = 0
        for tok in tokens:
            token_start = offset
            token_end = offset + len(tok)
            offset = token_end
            if is_valid_token(tok):
                times = [char_time[i] for i in range(token_start, token_end) if char_time[i] is not None]
                if times:
                    start = max(0.0, times[0][0])
                    end = max(start, times[-1][1])
                else:
                    start = end = prev_end
            else:
                start = end = prev_end
            rebuilt.append({"text": tok, "start": start, "end": end})
            prev_end = end

        stats = {
            "unanchored": unanchored,
            "located_words": located,
            "total_words": len(fa_words),
            "word_spans": word_spans,
        }
        return rebuilt, stats

    @staticmethod
    def _locate_word(text, ground_truth, cursor, is_valid_char):
        """在原文里从 `cursor` 往后定位一段词文本，返回 `[start, end)`；定位不上返回 None。

        词文本不含标点空白（FA 只把可发音字符切进词），所以先跳标点再比前缀即可。

        找不到时允许前跳有限距离：FA 偶尔在一处吞掉几个字，后一个词仍应锚回原位。
        但跳太远说明这条词对不上原文，宁可不锚（交给缺口夹），也不要锚到别处 ——
        错锚会让整块时间跳变，比留一个缺口危险得多。
        """
        index = cursor
        while index < len(ground_truth) and not is_valid_char(ground_truth[index]):
            index += 1
        if ground_truth.startswith(text, index):
            return index, index + len(text)
        # 前跳查找要守住词边界：`art` 不能锚进 `party` 中间，`500` 不能锚进 `15000`
        # 中间。那不是「词在前方」，而是「这个词被切在了另一个单位里」，锚上去会把
        # 时间推到错的位置。汉字不参与判定 —— 汉字逐字成词，相邻两字本来就是两个单位。
        limit = min(len(ground_truth), cursor + CHUNK_ANCHOR_SEARCH_CHARS)
        found = ground_truth.find(text, cursor, limit)
        while found != -1:
            right = found + len(text)
            left_split = (
                found > 0
                and _same_word_class(ground_truth[found - 1], ground_truth[found])
            )
            right_split = (
                right < len(ground_truth)
                and _same_word_class(ground_truth[right - 1], ground_truth[right])
            )
            if not left_split and not right_split:
                return found, right
            found = ground_truth.find(text, found + 1, limit)
        return None
