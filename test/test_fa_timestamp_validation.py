import unittest
from types import SimpleNamespace

from fa_timestamp_validation import (
    FaTimestampError,
    validate_absolute_timestamps,
    validate_fa_timestamps_in_chunk,
)


def fa_word(start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(start_time=start, end_time=end)


def abs_word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end)


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


if __name__ == "__main__":
    unittest.main()
