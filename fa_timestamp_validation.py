import copy
import math
from dataclasses import is_dataclass, replace


# FA 区间重叠压平容差（秒）。
# 判据：重叠量级是否还在「估边噪声」范围内。Qwen3-ForcedAligner 逐词估边，
# 相邻两词共享几十毫秒是模型输出的正常产物，压平即可。
# 与 FA_BOUNDARY_JITTER_SEC 同源（同一模型对同一词边界的估计误差），但两者作用的
# 物理关系不同——那个管「输出与音频窗口」，这个管「输出与输出」——故各自独立取值，
# 当前取同值只是量级一致，不代表可以互相替代。
# 超过此值说明词序本身不可信，属数据损坏，抛错而非静默压平。
FA_INTERVAL_OVERLAP_MAX_SEC = 0.5


# 跨模型边界抖动容差（秒）：chunk 窗口由 ASR 词时间戳推出，FA 是另一个模型，
# 两者对同一个词边界的估计天然有毫秒级偏差。接缝路径已用同一量级吸收
# （cloud_hotword.SEAM_JITTER_MAX_SEC），chunk 内窗口校验不得比它更严。
FA_BOUNDARY_JITTER_SEC = 0.5

# 长 chunk 追加的相对越界容差：越界不超过 chunk 时长的此比例时放行。
# 长 chunk 的语音速率与参考文本可能有百分级偏差，纯绝对容差会把这种良性偏差
# 误判成伪对齐；但比例容差必须封顶，否则越长的 chunk 越难判死。
FA_WINDOW_OVERFLOW_RATIO = 0.05
FA_WINDOW_OVERFLOW_MAX_SEC = 2.0

# 时间戳 round 到毫秒后的浮点尾差上限（秒）。差量小于 1µs 视为相等，不钳制。
SEAM_TIME_EPSILON = 1e-6


class FaTimestampError(ValueError):
    """FA timestamps are invalid within a chunk or across the final output."""


def clamp_seam_interval(previous_start, previous_end, previous_max_end, start, end):
    """把当前词条的区间抬到「此前所有已输出词条的 end 之后」。

    不变式：当前词条的 start >= 此前所有已输出词条的 end。只比前一项的 start/end
    不够：标点补位会把前一项的 end 抬到它自己的 start 之后，紧随其后的实词起点仍可能
    落在该 end 之前。此时点态单调成立、区间却重叠，下游按「前词 end 补位」重建标点时
    就会变成时间回退（2026-10-17 第 4 章：的 42.580..42.740 来自 chunk[1]，
    士 42.700..43.180 来自 chunk[2]，标点 〖 被补在 42.740）。

    返回 (new_start, new_end, start_overlap, point_regression)。
    两个差量是原始差值，<= 0 表示没有；调用方用它们决定 WARN 还是抛错。
    小于 1µs 的浮点尾差按「没有」处理，原样返回入参。
    """
    overlap = previous_max_end - start
    regression = max(previous_start - start, previous_end - end)
    if overlap <= SEAM_TIME_EPSILON and regression <= SEAM_TIME_EPSILON:
        return start, end, 0.0, 0.0
    new_start = max(start, previous_start, previous_max_end)
    new_end = max(end, new_start)
    return new_start, new_end, max(0.0, overlap), max(0.0, regression)


def _read_timestamp(item, key):
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


def _read_label(item):
    if isinstance(item, dict):
        return item.get("word", item.get("text", ""))
    return getattr(item, "text", "")


def _with_times(item, start, end):
    """返回带新时间戳的等价条目。

    qwen_asr.ForcedAlignItem 是 frozen dataclass，不能原地写属性，故返回新对象。
    支持 dict、dataclass（含 frozen）与可 `copy.copy` 后写属性的普通对象；
    namedtuple 之类不可写的对象会抛 AttributeError，不静默产出错值。
    """
    if isinstance(item, dict):
        return {**item, "start_time": start, "end_time": end}
    if is_dataclass(item):
        return replace(item, start_time=start, end_time=end)
    clone = copy.copy(item)
    clone.start_time = start
    clone.end_time = end
    return clone


def normalize_fa_interval_overlaps(word_alignments, chunk_start=None, chunk_end=None):
    """把同一份 FA 输出内部的区间重叠压平，使相邻词条不再共享时间区间。

    ForcedAligner 逐词估边，相邻两词共享几十毫秒是模型输出的正常现象，不是伪对齐。
    下游把标点补在「前一个有效词的 end」上，该补位成立的前提是「前词 end ≤ 后词 start」。
    区间重叠让这个前提失效，补位点落到后词起点之后，区间重叠就此变成时间回退
    （2026-10-17 事故：第 4 章「的」42580..42740 与「士」42700..43180 重叠 40ms，
    补位标点「〖」落在 42740，晚于「士」起点 42700）。

    故在 FA 输出进入 rebuild 之前统一压平一次：压平后上述前提自动成立，
    标点补位不必再为这种情况兜底。

    范围只限「同一份 FA 输出内部」的区间关系。相邻 chunk 各自的 FA 结果之间的拼接
    抖动是物理必然，另行处理，不共用本函数的判据：cloud_hotword 分支由
    `_enforce_seam_monotonicity` 按 SEAM_JITTER_MAX_SEC 钳制，pipeline 分支无钳制，
    只靠末尾 `validate_absolute_timestamps` 抛错。

    自带守卫（非有限值、区间倒挂），不依赖调用方先跑
    `validate_fa_timestamps_in_chunk`。越界本身是数据损坏，按 Fail Fast 抛错。

    chunk_start / chunk_end 只用于错误消息定位，不参与判定，可省略。
    返回 (压平后的新列表, 压平条目数, 最大压平量秒)。不原地修改入参。
    """
    where = f"chunk={chunk_start}..{chunk_end}s | " if chunk_start is not None else ""
    if not word_alignments:
        return list(word_alignments), 0, 0.0

    normalized = []
    clamped = 0
    max_overlap = 0.0
    previous_end = None
    for index, item in enumerate(word_alignments):
        start = float(_read_timestamp(item, "start_time"))
        end = float(_read_timestamp(item, "end_time"))
        if not math.isfinite(start) or not math.isfinite(end):
            raise FaTimestampError(
                f"FA 时间戳不是有限数值 | {where}item[{index}]={start}..{end}"
            )
        if start > end:
            raise FaTimestampError(
                f"FA 区间倒挂 | {where}item[{index}]={start:.3f}..{end:.3f}s | start > end"
            )
        if previous_end is not None and start < previous_end:
            overlap = previous_end - start
            if overlap > FA_INTERVAL_OVERLAP_MAX_SEC:
                raise FaTimestampError(
                    f"FA 区间重叠超容差 | {where}item[{index}] "
                    f"start={start:.3f}s 早于前项 end={previous_end:.3f}s | "
                    f"重叠={overlap:.3f}s (容差={FA_INTERVAL_OVERLAP_MAX_SEC}s) | "
                    f"该量级不是估边噪声，是词序损坏"
                )
            start = previous_end
            clamped += 1
            max_overlap = max(max_overlap, overlap)
        normalized.append(_with_times(item, start, end))
        previous_end = end

    return normalized, clamped, max_overlap


def validate_fa_timestamps_in_chunk(word_alignments, chunk_start, chunk_end, tolerance=FA_BOUNDARY_JITTER_SEC):
    """Validate relative FA timestamps before applying the chunk offset.

    返回带符号的窗口越界量（秒）：正数表示对齐末端超出窗口，负数表示起点早于窗口。
    越界在容差内视为跨模型边界抖动，放行并返回该值供调用方告警；超出容差才抛错，
    因为那说明文本与音频真的不匹配（FA 伪对齐），不是抖动。

    容差 = max(tolerance, min(chunk 时长 * FA_WINDOW_OVERFLOW_RATIO, FA_WINDOW_OVERFLOW_MAX_SEC))。
    """
    if not word_alignments:
        return 0.0

    chunk_duration = chunk_end - chunk_start
    window_tolerance = max(
        tolerance,
        min(chunk_duration * FA_WINDOW_OVERFLOW_RATIO, FA_WINDOW_OVERFLOW_MAX_SEC),
    )
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
    if relative_start < -window_tolerance or relative_end > chunk_duration + window_tolerance:
        raise FaTimestampError(
            f"FA 时间戳超出 chunk 音频窗口 | "
            f"chunk={chunk_start:.3f}..{chunk_end:.3f}s "
            f"(duration={chunk_duration:.3f}s, 容差={window_tolerance:.3f}s) | "
            f"FA relative={relative_start:.3f}..{relative_end:.3f}s | "
            f"FA absolute={chunk_start + relative_start:.3f}.."
            f"{chunk_start + relative_end:.3f}s"
        )

    overflow = 0.0
    if relative_start < 0:
        overflow = relative_start
    if relative_end - chunk_duration > abs(overflow):
        overflow = relative_end - chunk_duration
    return overflow


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
