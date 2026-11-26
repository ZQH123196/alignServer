import math


class FaTimestampError(ValueError):
    """FA timestamps are invalid within a chunk or across the final output."""


def _read_timestamp(item, key):
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


def _read_label(item):
    if isinstance(item, dict):
        return item.get("word", item.get("text", ""))
    return getattr(item, "text", "")


def validate_fa_timestamps_in_chunk(word_alignments, chunk_start, chunk_end, tolerance=0.05):
    """Validate relative FA timestamps before applying the chunk offset."""
    if not word_alignments:
        return

    chunk_duration = chunk_end - chunk_start
    previous = None
    for index, current in enumerate(word_alignments):
        start = float(_read_timestamp(current, "start_time"))
        end = float(_read_timestamp(current, "end_time"))
        if not math.isfinite(start) or not math.isfinite(end):
            raise FaTimestampError(
                f"FA chunk 时间戳不是有限数值 | chunk={chunk_start:.3f}..{chunk_end:.3f}s | "
                f"item[{index}]={start}..{end}"
            )
        if start > end:
            raise FaTimestampError(
                f"FA chunk 内部时间戳非法 | chunk={chunk_start:.3f}..{chunk_end:.3f}s | "
                f"item[{index}]={start:.3f}..{end:.3f}s | start > end"
            )
        if previous is not None:
            previous_start, previous_end = previous
            if start < previous_start or end < previous_end:
                raise FaTimestampError(
                    f"FA chunk 内部时间戳非单调 | "
                    f"chunk={chunk_start:.3f}..{chunk_end:.3f}s | "
                    f"previous={previous_start:.3f}..{previous_end:.3f}s | "
                    f"current={start:.3f}..{end:.3f}s"
                )
        previous = (start, end)

    relative_start = min(float(_read_timestamp(word, "start_time")) for word in word_alignments)
    relative_end = max(float(_read_timestamp(word, "end_time")) for word in word_alignments)
    if relative_start < -tolerance or relative_end > chunk_duration + tolerance:
        raise FaTimestampError(
            f"FA 时间戳超出 chunk 音频窗口 | "
            f"chunk={chunk_start:.3f}..{chunk_end:.3f}s "
            f"(duration={chunk_duration:.3f}s) | "
            f"FA relative={relative_start:.3f}..{relative_end:.3f}s | "
            f"FA absolute={chunk_start + relative_start:.3f}.."
            f"{chunk_start + relative_end:.3f}s"
        )


def validate_absolute_timestamps(words):
    """Validate final output timestamps after chunk offsets have been added."""
    violations = []
    previous = None
    for index, current in enumerate(words):
        start = float(_read_timestamp(current, "start"))
        end = float(_read_timestamp(current, "end"))
        reasons = []
        if not math.isfinite(start) or not math.isfinite(end):
            reasons.append("非有限数值")
        elif start > end:
            reasons.append("start > end")
        if previous is not None:
            previous_start, previous_end = previous[0], previous[1]
            if start < previous_start:
                reasons.append("start 回退")
            if end < previous_end:
                reasons.append("end 回退")
        if reasons:
            previous_word = previous[2] if previous is not None else ""
            violations.append(
                f"当前[{index}]'{_read_label(current)}' {start:.3f}..{end:.3f}; "
                f"前项'{previous_word}' ({', '.join(reasons)})"
            )
        previous = (start, end, _read_label(current))

    if violations:
        sample = "; ".join(violations[:5])
        if len(violations) > 5:
            sample += f"; ... 共 {len(violations)} 处时间戳非单调"
        raise FaTimestampError(
            f"字幕时间戳序列校验失败：{len(violations)} 处非单调或区间非法 | {sample}"
        )
