"""没有足够热词锚点时的固定窗口回退。

锚点少于 2 个时无法按热词切块。此时音频按 60s 窗口切，文本边界优先由调用方
给出的「音频时刻 -> 原文下标」查询决定；映射值已经是词边界，不再吸附。没有
查询时按字数比例估算，再向最近的句末标点吸附（搜索半径 40 字）。只有一个锚点
时，把它当唯一内部边界，两侧各自按同一规则切。段与段之间不留文本重叠、
不留音频 pad。

没有查询时是纯估算，文本与音频的比例关系按平均语速假设，可能有秒级漂移。
调用方必须打 WARN，不能把它当成与热词切块同质量的结果。
"""

from collections.abc import Callable

FALLBACK_WINDOW_MS = 60_000
SNAP_RADIUS_CHARS = 40
SENTENCE_END = "。！？…\n"

TextOffsetForTime = Callable[[int], int]


def _snap_to_sentence_end(text: str, target: int, low: int, high: int) -> int:
    """把边界移到 low..high 内最近的「句末标点之后」，找不到就用原目标。"""

    for delta in range(SNAP_RADIUS_CHARS + 1):
        for candidate in (target + delta, target - delta):
            if low < candidate <= high and text[candidate - 1] in SENTENCE_END:
                return candidate
    return target


def _audio_grid(start_ms: int, end_ms: int, max_segments: int) -> list[int]:
    """从 start_ms 起每 60s 一个边界，最后一段可以短于 60s。

    段数不超过 max_segments：文本比窗口还短时不能切出空文本段。
    """

    boundaries = [start_ms]
    cursor = start_ms
    while end_ms - cursor > FALLBACK_WINDOW_MS and len(boundaries) < max_segments:
        cursor += FALLBACK_WINDOW_MS
        boundaries.append(cursor)
    if boundaries[-1] != end_ms:
        boundaries.append(end_ms)
    return boundaries


def _proportional_offset(
    boundary_ms: int, audio_start: int, audio_end: int, text_start: int, text_end: int
) -> int:
    return round(
        text_start + (text_end - text_start) * (boundary_ms - audio_start) / (audio_end - audio_start)
    )


def _resolve_text_boundaries(
    reference_text: str, raw_offsets: list[int], fixed: list[bool]
) -> list[int]:
    """把原始文本目标压成严格递增的边界，并对非固定边界做句末吸附。"""

    count = len(raw_offsets)
    boundaries = [raw_offsets[0]]
    for index in range(1, count):
        highest = len(reference_text) - (count - 1 - index)
        target = min(max(raw_offsets[index], boundaries[-1] + 1), highest)
        if fixed[index]:
            boundaries.append(target)
        else:
            boundaries.append(
                _snap_to_sentence_end(reference_text, target, boundaries[-1], highest)
            )
    return boundaries


def build_fallback_chunks(
    reference_text: str,
    total_duration_ms: int,
    *,
    text_offset_for_time: TextOffsetForTime | None = None,
    anchor_ms: int | None = None,
    anchor_text_offset: int | None = None,
) -> list[dict[str, int]]:
    """生成固定窗口切块。文本区间是原文下标，不是归一化下标。

    返回字段：index、origStart、origEnd、audioStartMs、audioEndMs。
    调用方必须按 `origStart` / `origEnd` 取 FA 参考文本。
    """

    if not reference_text or total_duration_ms <= 0:
        return []

    has_anchor = (
        anchor_ms is not None
        and anchor_text_offset is not None
        and 0 < anchor_ms < total_duration_ms
        and 0 < anchor_text_offset < len(reference_text)
    )
    if has_anchor:
        sides = [
            (0, anchor_ms, 0, anchor_text_offset),
            (anchor_ms, total_duration_ms, anchor_text_offset, len(reference_text)),
        ]
    else:
        sides = [(0, total_duration_ms, 0, len(reference_text))]

    audio_boundaries: list[int] = []
    raw_text_offsets: list[int] = []
    fixed_flags: list[bool] = []
    for audio_start, audio_end, text_start, text_end in sides:
        grid = _audio_grid(audio_start, audio_end, max(1, text_end - text_start))
        raw_offsets = [text_start]
        fixed = [True]
        for boundary in grid[1:-1]:
            if text_offset_for_time is not None:
                # 时刻已经映射到真实词边界，再向句末标点吸附只会把文本推离音频。
                raw_offsets.append(text_offset_for_time(boundary))
                fixed.append(True)
                continue
            raw_offsets.append(
                _proportional_offset(
                    boundary, audio_start, audio_end, text_start, text_end
                )
            )
            fixed.append(False)
        raw_offsets.append(text_end)
        fixed.append(True)

        offset = 1 if audio_boundaries else 0
        audio_boundaries.extend(grid[offset:])
        raw_text_offsets.extend(raw_offsets[offset:])
        fixed_flags.extend(fixed[offset:])

    text_boundaries = _resolve_text_boundaries(
        reference_text, raw_text_offsets, fixed_flags
    )
    return [
        {
            "index": index,
            "origStart": text_boundaries[index],
            "origEnd": text_boundaries[index + 1],
            "audioStartMs": audio_boundaries[index],
            "audioEndMs": audio_boundaries[index + 1],
        }
        for index in range(len(audio_boundaries) - 1)
    ]
