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


def validate_absolute_timestamps(words, origins=None):
    """Validate final output timestamps after chunk offsets have been added.

    start == end 的零时长区间（标点继承前词 end）是合法的，只有 start > end
    或相对前一项的时间回退才构成违规。

    origins: 可选，与 words 等长的来源标签列表（如 "chunk[3] 120.000..180.300s"），
    用于在违规信息中标注词条来自哪个 chunk 的 FA 对齐，辅助定位跨块边界问题。
    """
    def _origin(idx):
        if origins and idx < len(origins):
            return f" 来源={origins[idx]}"
        return ""

    violations = []
    previous = None
    for index, current in enumerate(words):
        start = float(_read_timestamp(current, "start"))
        end = float(_read_timestamp(current, "end"))
        label = _read_label(current)
        reasons = []
        if not math.isfinite(start) or not math.isfinite(end):
            reasons.append("非有限数值")
        elif start > end:
            reasons.append(f"start > end（超出 {start - end:.3f}s）")
        cross_seam = False
        if previous is not None:
            p_index, p_start, p_end, _ = previous
            if start < p_start:
                reasons.append(f"start 回退 {p_start - start:.3f}s")
            if end < p_end:
                reasons.append(f"end 回退 {p_end - end:.3f}s")
            if _origin(index) and _origin(p_index) and origins[index] != origins[p_index]:
                cross_seam = True
        if reasons:
            if previous is not None:
                p_index, p_start, p_end, p_label = previous
                prev_desc = f"前项[{p_index}]'{p_label}' {p_start:.3f}..{p_end:.3f}{_origin(p_index)}"
            else:
                prev_desc = "前项=无（首词）"
            detail = ", ".join(reasons)
            if cross_seam:
                detail += " | 跨 chunk 边界：前后词来自不同 chunk 的 FA 对齐"
            violations.append(
                f"当前[{index}]'{label}' {start:.3f}..{end:.3f}{_origin(index)}; "
                f"{prev_desc} ({detail})"
            )
        previous = (index, start, end, label)

    if violations:
        sample = "; ".join(violations[:5])
        if len(violations) > 5:
            sample += f"; ... 共 {len(violations)} 处时间戳非单调"
        raise FaTimestampError(
            f"字幕时间戳序列校验失败：{len(violations)} 处非单调或区间非法 | {sample}"
        )
