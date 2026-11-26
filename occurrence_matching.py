"""按热词出现序列匹配锚点，并做首尾锁定的时间抽样。

参考文本与 ASR 文本都先把每个热词的全部出现位置收集成序列（元素是热词编号），
再做最长公共子序列（LCS）。LCS 保序，天然剔除多出、缺失、错位的出现。
命中位置映射回 ASR 词的 `begin_time` / `end_time`，成为强制对齐的切块锚点。

抽样规则（2026-10-18 定案）：先锁定首、尾锚点，中间锚点与上一个保留锚点、
与尾部锚点都至少相隔 20s。锚点少时按原样返回，不补造锚点。
"""

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

ANCHOR_MIN_GAP_MS = 20_000


@dataclass(frozen=True)
class TermMatchStats:
    """单个热词的出现与命中统计，只用于日志与诊断。"""

    text: str
    reference_count: int
    asr_count: int
    matched: int


def find_occurrences(haystack: str, needle: str) -> list[int]:
    """返回 needle 在 haystack 里的全部起始位置，允许重叠。"""

    positions: list[int] = []
    if not needle:
        return positions
    cursor = 0
    while True:
        position = haystack.find(needle, cursor)
        if position < 0:
            return positions
        positions.append(position)
        cursor = position + 1


def lcs_pairs(left: Sequence[int], right: Sequence[int]) -> list[tuple[int, int]]:
    """最长公共子序列的下标对，按序列顺序返回。"""

    rows, cols = len(left), len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if left[i] == right[j]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < rows and j < cols:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _build_anchor(
    term: str,
    reference_offset: int,
    asr_position: int,
    char_to_word: Sequence[int],
    words: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    start_word = char_to_word[asr_position]
    end_word = char_to_word[asr_position + len(term) - 1]
    begin_time = words[start_word].get("begin_time", 0)
    end_time = words[end_word].get("end_time", 0)
    return {
        "text": term,
        "normText": term,
        "offsetInFullNorm": reference_offset,
        "beginTime": begin_time,
        "endTime": max(end_time, begin_time),
    }


def _dedupe_reference_positions(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一参考位置只留最长的那条词。

    「蜘蛛」与「蜘蛛丝」可能在同一位置各有一条锚点，它们给出相同起点、不同终点。
    保留最长词，切块边界才落在完整词之后。
    """

    by_offset: dict[int, dict[str, Any]] = {}
    for anchor in anchors:
        offset = anchor["offsetInFullNorm"]
        current = by_offset.get(offset)
        if current is None or len(anchor["normText"]) > len(current["normText"]):
            by_offset[offset] = anchor
    return [by_offset[offset] for offset in sorted(by_offset)]


def _drop_backward_anchors(
    anchors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    previous_begin = -1
    for anchor in anchors:
        if anchor["beginTime"] < previous_begin:
            dropped.append(anchor)
            continue
        kept.append(anchor)
        previous_begin = anchor["beginTime"]
    return kept, dropped


def match_hotword_anchors(
    terms: Sequence[str],
    reference_norm: str,
    asr_stream: str,
    char_to_word: Sequence[int],
    words: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[TermMatchStats], list[dict[str, Any]]]:
    """用出现序列 LCS 生成锚点。

    返回 (锚点, 每条热词的统计, 因时间回退被丢弃的锚点)。
    锚点按参考文本位置升序；`beginTime` 非递减。
    """

    reference_occurrences: list[tuple[int, int]] = []
    asr_occurrences: list[tuple[int, int]] = []
    reference_counts: list[int] = []
    asr_counts: list[int] = []
    for term_index, term in enumerate(terms):
        reference_positions = find_occurrences(reference_norm, term)
        asr_positions = find_occurrences(asr_stream, term)
        reference_counts.append(len(reference_positions))
        asr_counts.append(len(asr_positions))
        reference_occurrences.extend(
            (position, term_index) for position in reference_positions
        )
        asr_occurrences.extend((position, term_index) for position in asr_positions)

    stats = [
        TermMatchStats(
            text=term,
            reference_count=reference_counts[index],
            asr_count=asr_counts[index],
            matched=0,
        )
        for index, term in enumerate(terms)
    ]

    if not reference_occurrences or not asr_occurrences or not char_to_word:
        return [], stats, []

    reference_occurrences.sort()
    asr_occurrences.sort()
    pairs = lcs_pairs(
        [term_index for _, term_index in reference_occurrences],
        [term_index for _, term_index in asr_occurrences],
    )

    matched_per_term = [0] * len(terms)
    anchors: list[dict[str, Any]] = []
    for reference_index, asr_index in pairs:
        reference_position, term_index = reference_occurrences[reference_index]
        asr_position, _ = asr_occurrences[asr_index]
        matched_per_term[term_index] += 1
        anchors.append(
            _build_anchor(
                terms[term_index],
                reference_position,
                asr_position,
                char_to_word,
                words,
            )
        )

    stats = [
        TermMatchStats(
            text=stats[index].text,
            reference_count=stats[index].reference_count,
            asr_count=stats[index].asr_count,
            matched=matched_per_term[index],
        )
        for index in range(len(terms))
    ]

    anchors = _dedupe_reference_positions(anchors)
    anchors, dropped = _drop_backward_anchors(anchors)
    return anchors, stats, dropped


def build_reference_offset_lookup(
    reference_norm: str,
    asr_stream: str,
    char_to_word: Sequence[int],
    words: Sequence[dict[str, Any]],
) -> Callable[[int], int]:
    """构造「音频时刻 -> 参考文本归一化下标」的查询函数。

    没有热词锚点时，固定窗口回退需要知道每个音频窗口对应哪段参考文本。
    只按字数比例切会累计漂移（实测 53.8s 窗口偏 5.5s，FA 判越界）。这里改用
    覆盖率同款的匹配块：ASR 字符 -> 参考字符，再取目标时刻之后第一个 ASR 词的
    首字符。ASR 词时间比「平均语速」可靠。

    返回的下标落在 `[0, len(reference_norm)]`。目标时刻落在 ASR 缺口内时，
    取缺口之后那个匹配块的起点。
    """

    blocks = [
        block
        for block in SequenceMatcher(
            None, reference_norm, asr_stream, autojunk=False
        ).get_matching_blocks()
        if block.size > 0
    ]
    end_times = [word.get("end_time", 0) for word in words]

    def lookup(target_ms: int) -> int:
        if not words or not char_to_word or not blocks:
            return 0
        word_index = bisect.bisect_left(end_times, target_ms)
        word_index = min(word_index, len(words) - 1)
        asr_position = bisect.bisect_left(char_to_word, word_index)
        for block in blocks:
            if block.b <= asr_position < block.b + block.size:
                return min(block.a + (asr_position - block.b), len(reference_norm))
            if asr_position < block.b:
                return min(block.a, len(reference_norm))
        return len(reference_norm)

    return lookup


def sample_anchors(
    anchors: Sequence[dict[str, Any]], min_gap_ms: int = ANCHOR_MIN_GAP_MS
) -> list[dict[str, Any]]:
    """锁定首尾，中间锚点与两侧都至少相隔 min_gap_ms。"""

    if len(anchors) <= 2:
        return list(anchors)

    last = anchors[-1]
    kept = [anchors[0]]
    for anchor in anchors[1:-1]:
        if anchor["beginTime"] - kept[-1]["endTime"] < min_gap_ms:
            continue
        if last["beginTime"] - anchor["endTime"] < min_gap_ms:
            continue
        kept.append(anchor)
    kept.append(last)
    return kept
