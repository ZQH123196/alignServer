import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from fa_timestamp_validation import (
    FA_INTERVAL_OVERLAP_MAX_SEC,
    FaTimestampError,
    clamp_seam_interval,
    normalize_fa_interval_overlaps,
    validate_absolute_timestamps,
    validate_fa_timestamps_in_chunk,
)


def fa_word(start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(start_time=start, end_time=end)


def abs_word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end)


@dataclass(frozen=True)
class FrozenFaWord:
    """镜像 qwen_asr.ForcedAlignItem：frozen dataclass，不可原地改。"""

    text: str
    start_time: float
    end_time: float


class FaTimestampsInChunkTests(unittest.TestCase):
    def test_alignment_inside_window_returns_zero_overflow(self) -> None:
        # Arrange
        words = [fa_word(0.0, 9.9)]

        # Act
        overflow = validate_fa_timestamps_in_chunk(words, 0.0, 10.0)

        # Assert
        self.assertEqual(overflow, 0.0)

    def test_cross_model_boundary_jitter_is_accepted(self) -> None:
        # Arrange: 生产事故复现 —— chunk 窗口 21.36s，FA 末端 21.60s，越界 0.24s
        words = [fa_word(0.640, 21.600)]

        # Act
        overflow = validate_fa_timestamps_in_chunk(words, 299.470, 320.830)

        # Assert
        self.assertAlmostEqual(overflow, 0.240, places=3)

    def test_overflow_beyond_tolerance_raises_for_short_chunk(self) -> None:
        # Arrange: 窗口 2s，容差 0.5s，FA 末端 3.0s（越界 1.0s）
        words = [fa_word(0.0, 3.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 10.0, 12.0)

    def test_long_chunk_uses_relative_tolerance(self) -> None:
        # Arrange: 窗口 100s，越界 1.5s（超过绝对容差 0.5s，但在相对容差内）
        words = [fa_word(0.0, 101.5)]

        # Act
        overflow = validate_fa_timestamps_in_chunk(words, 0.0, 100.0)

        # Assert
        self.assertAlmostEqual(overflow, 1.5, places=3)

    def test_relative_tolerance_is_capped_on_long_chunk(self) -> None:
        # Arrange: 窗口 100s，越界 4s（低于 5% 比例但超过 2s 封顶）→ 判定伪对齐
        words = [fa_word(0.0, 104.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 0.0, 100.0)

    def test_start_before_window_within_tolerance_is_accepted(self) -> None:
        # Arrange: 起点早于窗口 0.3s，在抖动容差内
        words = [fa_word(-0.3, 9.0)]

        # Act
        overflow = validate_fa_timestamps_in_chunk(words, 100.0, 110.0)

        # Assert
        self.assertAlmostEqual(overflow, -0.3, places=3)

    def test_start_before_window_beyond_tolerance_raises(self) -> None:
        # Arrange: 起点早于窗口 2s，超出抖动容差
        words = [fa_word(-2.0, 9.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 100.0, 110.0)

    def test_non_finite_timestamp_raises(self) -> None:
        # Arrange
        words = [fa_word(float("nan"), 1.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 0.0, 10.0)

    def test_start_after_end_raises(self) -> None:
        # Arrange
        words = [fa_word(2.0, 1.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 0.0, 10.0)

    def test_non_monotonic_timestamps_raise(self) -> None:
        # Arrange
        words = [fa_word(0.0, 1.0), fa_word(2.0, 3.0), fa_word(1.0, 4.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_fa_timestamps_in_chunk(words, 0.0, 10.0)

    def test_empty_alignment_returns_zero(self) -> None:
        # Arrange / Act
        overflow = validate_fa_timestamps_in_chunk([], 0.0, 10.0)

        # Assert
        self.assertEqual(overflow, 0.0)

    def test_dict_items_are_supported(self) -> None:
        # Arrange
        words = [{"start_time": 0.0, "end_time": 1.0}]

        # Act
        overflow = validate_fa_timestamps_in_chunk(words, 0.0, 10.0)

        # Assert
        self.assertEqual(overflow, 0.0)


class AbsoluteTimestampsTests(unittest.TestCase):
    def test_monotonic_sequence_passes(self) -> None:
        # Arrange
        words = [abs_word("a", 0.0, 1.0), abs_word("b", 1.0, 2.0)]

        # Act / Assert (no raise)
        validate_absolute_timestamps(words)

    def test_zero_length_punctuation_passes(self) -> None:
        # Arrange: 标点继承前词 end，start == end 合法
        words = [abs_word("a", 0.0, 1.0), abs_word("，", 1.0, 1.0)]

        # Act / Assert (no raise)
        validate_absolute_timestamps(words)

    def test_start_regression_raises(self) -> None:
        # Arrange: 第二项 start 回退到 0.5，早于前项 start 1.0
        words = [abs_word("a", 1.0, 2.0), abs_word("b", 0.5, 2.5)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_absolute_timestamps(words)

    def test_start_after_end_raises(self) -> None:
        # Arrange
        words = [abs_word("a", 2.0, 1.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError):
            validate_absolute_timestamps(words)


class NormalizeIntervalOverlapsTests(unittest.TestCase):
    def test_incident_case_overlap_is_flattened(self) -> None:
        # Arrange: 2026-10-17 事故数据。「的」42580..42740 与「士」42700..43180 重叠 40ms。
        words = [fa_word(42.580, 42.740), fa_word(42.700, 43.180)]

        # Act
        normalized, clamped, max_overlap = normalize_fa_interval_overlaps(words)

        # Assert: 「士」起点抬到前项 end，区间不再共享
        self.assertEqual(clamped, 1)
        self.assertAlmostEqual(max_overlap, 0.040, places=3)
        self.assertAlmostEqual(normalized[0].end_time, 42.740, places=3)
        self.assertAlmostEqual(normalized[1].start_time, 42.740, places=3)
        self.assertAlmostEqual(normalized[1].end_time, 43.180, places=3)

    def test_flattened_output_keeps_punctuation_padding_ordered(self) -> None:
        # Arrange: 标点补位规则把标点放在前一个实词的 end 上。
        # 该补位不制造时间回退的前提是「前词 end ≤ 后词 start」，本例即该不变量。
        words = [fa_word(42.580, 42.740), fa_word(42.700, 43.180)]

        # Act
        normalized, _, _ = normalize_fa_interval_overlaps(words)
        punct_at = normalized[0].end_time

        # Assert
        self.assertGreaterEqual(normalized[1].start_time, punct_at)

    def test_overlap_within_tolerance_is_flattened_not_raised(self) -> None:
        # Arrange: 重叠 0.04s，在容差内，属估边噪声
        words = [fa_word(0.0, 1.0), fa_word(0.96, 2.0)]

        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps(words)

        # Assert
        self.assertEqual(clamped, 1)
        self.assertAlmostEqual(normalized[1].start_time, 1.0, places=6)

    def test_overlap_exactly_at_tolerance_is_flattened(self) -> None:
        # Arrange: 重叠恰好等于容差，边界取「不超过即压平」
        words = [fa_word(0.0, 1.0), fa_word(1.0 - FA_INTERVAL_OVERLAP_MAX_SEC, 2.0)]

        # Act
        _, clamped, max_overlap = normalize_fa_interval_overlaps(words)

        # Assert
        self.assertEqual(clamped, 1)
        self.assertAlmostEqual(max_overlap, FA_INTERVAL_OVERLAP_MAX_SEC, places=6)

    def test_overlap_beyond_tolerance_raises_with_chunk_context(self) -> None:
        # Arrange: 重叠超过容差，词序不可信
        words = [fa_word(0.0, 1.0), fa_word(1.0 - FA_INTERVAL_OVERLAP_MAX_SEC - 0.1, 2.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError) as ctx:
            normalize_fa_interval_overlaps(words, chunk_start=10.0, chunk_end=12.0)

        # Assert: 错误消息能定位到 chunk
        self.assertIn("chunk=10.0..12.0s", str(ctx.exception))

    def test_inverted_interval_raises(self) -> None:
        # Arrange: start > end 是数据损坏，本函数自带守卫，不依赖调用方先跑 validate
        words = [fa_word(2.0, 1.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError) as ctx:
            normalize_fa_interval_overlaps(words)

        # Assert
        self.assertIn("区间倒挂", str(ctx.exception))

    def test_non_finite_timestamp_raises(self) -> None:
        # Arrange
        words = [fa_word(0.0, 1.0), fa_word(float("nan"), 2.0)]

        # Act / Assert
        with self.assertRaises(FaTimestampError) as ctx:
            normalize_fa_interval_overlaps(words)

        # Assert
        self.assertIn("不是有限数值", str(ctx.exception))

    def test_disjoint_input_is_returned_unchanged(self) -> None:
        # Arrange / Act
        words = [fa_word(0.0, 1.0), fa_word(1.0, 2.0)]
        normalized, clamped, max_overlap = normalize_fa_interval_overlaps(words)

        # Assert
        self.assertEqual(clamped, 0)
        self.assertEqual(max_overlap, 0.0)
        self.assertEqual(
            [(w.start_time, w.end_time) for w in normalized],
            [(0.0, 1.0), (1.0, 2.0)],
        )

    def test_result_has_no_shared_interval_and_is_pointwise_monotonic(self) -> None:
        # Arrange: 一串重叠，逐个压平
        words = [
            fa_word(0.00, 1.00),
            fa_word(0.90, 1.80),
            fa_word(1.70, 2.50),
            fa_word(2.40, 3.30),
            fa_word(3.60, 4.50),
        ]

        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps(words)

        # Assert: 相邻区间不共享，且 begin/end 各自不减
        self.assertEqual(clamped, 3)
        for previous, current in zip(normalized, normalized[1:]):
            self.assertGreaterEqual(current.start_time, previous.end_time)
            self.assertGreaterEqual(current.start_time, previous.start_time)
            self.assertGreaterEqual(current.end_time, previous.end_time)

    def test_output_keeps_length_and_order(self) -> None:
        # Arrange
        words = [fa_word(i * 0.1, i * 0.1 + 0.08) for i in range(20)]

        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps(words)

        # Assert: 长度与顺序不变，压平后仍是同一批词
        self.assertEqual(len(normalized), len(words))
        self.assertEqual(clamped, 0)
        for index, (before, after) in enumerate(zip(words, normalized)):
            self.assertAlmostEqual(after.start_time, before.start_time, places=6)
            self.assertAlmostEqual(after.end_time, before.end_time, places=6)

    def test_input_is_not_mutated(self) -> None:
        # Arrange: FA 结果不可变，归一化必须返回新列表
        words = [fa_word(0.0, 1.0), fa_word(0.5, 2.0)]

        # Act
        normalize_fa_interval_overlaps(words)

        # Assert
        self.assertAlmostEqual(words[1].start_time, 0.5, places=6)

    def test_frozen_dataclass_items_are_supported(self) -> None:
        # Arrange: 真实类型是 frozen dataclass
        words = [FrozenFaWord("的", 42.580, 42.740), FrozenFaWord("士", 42.700, 43.180)]

        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps(words)

        # Assert: 新对象带修正后的时间，原对象不变，文本保留
        self.assertEqual(clamped, 1)
        self.assertAlmostEqual(normalized[1].start_time, 42.740, places=3)
        self.assertAlmostEqual(words[1].start_time, 42.700, places=3)
        self.assertEqual(normalized[1].text, "士")

    def test_dict_items_are_supported(self) -> None:
        # Arrange
        words = [{"text": "的", "start_time": 0.0, "end_time": 1.0},
                 {"text": "士", "start_time": 0.5, "end_time": 2.0}]

        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps(words)

        # Assert
        self.assertEqual(clamped, 1)
        self.assertAlmostEqual(normalized[1]["start_time"], 1.0, places=6)
        self.assertEqual(words[1]["start_time"], 0.5)

    def test_empty_input_returns_empty(self) -> None:
        # Act
        normalized, clamped, max_overlap = normalize_fa_interval_overlaps([])

        # Assert
        self.assertEqual(normalized, [])
        self.assertEqual(clamped, 0)
        self.assertEqual(max_overlap, 0.0)

    def test_single_word_is_returned_unchanged(self) -> None:
        # Act
        normalized, clamped, _ = normalize_fa_interval_overlaps([fa_word(0.0, 1.0)])

        # Assert
        self.assertEqual(clamped, 0)
        self.assertAlmostEqual(normalized[0].start_time, 0.0, places=6)


class ClampSeamIntervalTests(unittest.TestCase):
    """块接缝不变式：每个词条的 start 必须 >= 此前所有已输出词条的 end。

    2026-10-17 第 4 章事故：只比前一任的 start/end 时，「〖」被抬到
    42700..42740 后，「士」42700..43180 仍算「没回退」，下游按前词 end
    补位标点后才炸成时间回退。
    """

    def test_incident_pair_is_pushed_after_previous_end(self) -> None:
        # Arrange: 的 42.580..42.740 来自 chunk[1]，士 42.700..43.180 来自 chunk[2]
        # Act
        new_start, new_end, overlap, regression = clamp_seam_interval(
            42.580, 42.740, 42.740, 42.700, 43.180,
        )

        # Assert: 起点抬到此前最大 end，终点不动，点态没有回退
        self.assertAlmostEqual(new_start, 42.740, places=6)
        self.assertAlmostEqual(new_end, 43.180, places=6)
        self.assertAlmostEqual(overlap, 0.040, places=6)
        self.assertAlmostEqual(regression, 0.0, places=6)

    def test_word_after_clamped_punctuation_is_pushed_too(self) -> None:
        # Arrange: 标点已被钳制成 42.740..42.740，紧随的实词仍在 42.700 起
        # Act
        new_start, new_end, overlap, _ = clamp_seam_interval(
            42.740, 42.740, 42.740, 42.700, 43.180,
        )

        # Assert: 只看前项的 start/end 时 regression = 0，靠 max_end 才抓得住
        self.assertAlmostEqual(new_start, 42.740, places=6)
        self.assertAlmostEqual(new_end, 43.180, places=6)
        self.assertAlmostEqual(overlap, 0.040, places=6)

    def test_disjoint_interval_is_returned_unchanged(self) -> None:
        # Act
        new_start, new_end, overlap, regression = clamp_seam_interval(
            0.0, 1.0, 1.0, 1.0, 2.0,
        )

        # Assert
        self.assertEqual((new_start, new_end, overlap, regression), (1.0, 2.0, 0.0, 0.0))

    def test_floating_point_tail_inside_tolerance_is_not_a_drift(self) -> None:
        # Arrange: round 到毫秒后的浮点尾差
        # Act
        new_start, new_end, overlap, regression = clamp_seam_interval(
            0.0, 1.0000000001, 1.0000000001, 1.0, 2.0,
        )

        # Assert: 不动、不报
        self.assertEqual((new_start, new_end, overlap, regression), (1.0, 2.0, 0.0, 0.0))

    def test_end_regression_is_reported(self) -> None:
        # Act
        _, new_end, overlap, regression = clamp_seam_interval(
            0.0, 1.0, 1.0, 0.5, 0.9,
        )

        # Assert
        self.assertAlmostEqual(overlap, 0.5, places=6)
        self.assertAlmostEqual(regression, 0.1, places=6)
        self.assertAlmostEqual(new_end, 1.0, places=6)

    def test_word_entirely_before_previous_end_becomes_zero_width(self) -> None:
        # Arrange: 当前词的 end 也在此前最大 end 之前
        # Act
        new_start, new_end, overlap, _ = clamp_seam_interval(0.0, 5.0, 5.0, 1.0, 2.0)

        # Assert: 抬到边界即止，不造负时长
        self.assertAlmostEqual(new_start, 5.0, places=6)
        self.assertAlmostEqual(new_end, 5.0, places=6)
        self.assertAlmostEqual(overlap, 4.0, places=6)

    def test_incident_stream_becomes_interval_disjoint(self) -> None:
        # Arrange: 真实的三个词条（的 / 〖 / 士），含块接缝抖动
        stream = [("的", 42.580, 42.740), ("〖", 42.700, 42.700), ("士", 42.700, 43.180)]

        # Act: 按 cloud_hotword 的发射顺序逐条钳制（last_emitted 存已输出最大 end）
        emitted = []
        max_end = 0.0
        previous_start, previous_end = 0.0, 0.0
        for text, start, end in stream:
            if emitted:
                start, end, _, _ = clamp_seam_interval(
                    previous_start, previous_end, max_end, start, end,
                )
            emitted.append((text, start, end))
            previous_start, previous_end = start, end
            max_end = max(max_end, end)

        # Assert: 每个词条的起点都不早于此前所有词条的 end
        running_end = 0.0
        for text, start, end in emitted:
            self.assertGreaterEqual(start, running_end, f"{text} 的起点早于此前的最大 end")
            running_end = max(running_end, end)

        # Assert: 「士」的起点已被抬到 42.740，标点补位不会再越到它后面
        self.assertAlmostEqual(emitted[1][1], 42.740, places=6)
        self.assertAlmostEqual(emitted[2][1], 42.740, places=6)


class NormalizeCallSiteTests(unittest.TestCase):
    """结构守卫：每个 FA→rebuild 接入点都必须先压平再 rebuild。

    pipeline / cloud_hotword 依赖 torch、qwen_asr 与 main 的启动检查，
    单测里无法 import，故只能对源码做结构检查。它挡不住逻辑写错，
    挡得住「有人删掉某一处接入」——那正是本次事故的失效类别。

    每项为 (文件, 接入点锚点, 该接入点内的 rebuild 调用)。锚点取接入点入口，
    断言该入口之后先出现压平调用、再出现 rebuild 调用。
    """

    SITES = (
        ("pipeline.py", "def _emit_fa(", "_rebuild_words_from_ground_truth(ground_truth, disjoint_fa_res, run_lang)"),
        ("pipeline.py", "for chunk_info, ground_truth, fa_res in aligned_chunks:", "rebuilt_words = self._rebuild_words_from_ground_truth("),
        ("cloud_hotword.py", "self._assert_fa_timestamps(", "self._rebuild_words("),
    )

    def test_every_call_site_normalizes_before_rebuild(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        for filename, anchor, rebuild_anchor in self.SITES:
            with self.subTest(file=filename, anchor=anchor):
                source = (project_root / filename).read_text(encoding="utf-8")
                self.assertIn(anchor, source, f"{filename} 找不到接入点锚点 {anchor!r}")

                body = source.split(anchor, 1)[1][:2000]
                normalize_at = body.find("normalize_fa_interval_overlaps(")
                rebuild_at = body.find(rebuild_anchor)
                self.assertNotEqual(
                    normalize_at, -1,
                    f"{filename} 在 {anchor!r} 之后没有调用 normalize_fa_interval_overlaps",
                )
                self.assertNotEqual(
                    rebuild_at, -1,
                    f"{filename} 在 {anchor!r} 之后没有找到 rebuild 调用 {rebuild_anchor!r}",
                )
                self.assertLess(
                    normalize_at, rebuild_at,
                    f"{filename} 的 {anchor!r} 处先 rebuild 后压平，顺序反了",
                )


if __name__ == "__main__":
    unittest.main()
