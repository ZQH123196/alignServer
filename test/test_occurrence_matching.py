import sys
import types
import unittest

if "main" not in sys.modules:
    _main_stub = types.ModuleType("main")
    for _name in ("log", "log_info", "log_warn", "log_error", "log_block"):
        setattr(_main_stub, _name, lambda *args, **kwargs: None)
    _main_stub.get_task_info = lambda: {}
    sys.modules["main"] = _main_stub

from cloud_hotword import norm_for_match  # noqa: E402
from hotword_matching import build_asr_stream  # noqa: E402
from occurrence_matching import (  # noqa: E402
    build_reference_offset_lookup,
    find_occurrences,
    lcs_pairs,
    match_hotword_anchors,
    sample_anchors,
)


def make_asr(words_with_times):
    words = [
        {"text": text, "punctuation": "", "begin_time": begin, "end_time": end}
        for text, begin, end in words_with_times
    ]
    stream, char_to_word = build_asr_stream(words, "zh", norm_for_match)
    return words, stream, char_to_word


class OccurrenceTests(unittest.TestCase):
    def test_find_occurrences_keeps_overlaps(self) -> None:
        self.assertEqual(find_occurrences("沃尔沃尔沃尔", "沃尔"), [0, 2, 4])

    def test_lcs_pairs_drop_extras_and_preserve_order(self) -> None:
        # Arrange：右侧多一个 2，左侧顺序保持不变
        pairs = lcs_pairs([0, 1, 0, 2], [1, 0, 2, 2])

        # Assert
        self.assertEqual(len(pairs), 3)
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        self.assertEqual(left, sorted(left))
        self.assertEqual(right, sorted(right))


class AnchorTests(unittest.TestCase):
    def test_anchor_spans_every_word_of_the_term(self) -> None:
        # Arrange：第二个「沃尔」是一个整词，起点 300、终点 500
        words, stream, char_to_word = make_asr(
            [
                ("沃", 0, 100),
                ("尔", 100, 200),
                ("猛", 200, 300),
                ("沃尔", 300, 500),
            ]
        )

        # Act
        anchors, stats, dropped = match_hotword_anchors(
            ["沃尔"], "沃尔猛沃尔", stream, char_to_word, words
        )

        # Assert
        self.assertEqual([anchor["beginTime"] for anchor in anchors], [0, 300])
        self.assertEqual([anchor["endTime"] for anchor in anchors], [200, 500])
        self.assertEqual(stats[0].reference_count, 2)
        self.assertEqual(stats[0].asr_count, 2)
        self.assertEqual(stats[0].matched, 2)
        self.assertEqual(dropped, [])

    def test_missing_occurrence_is_dropped_by_lcs(self) -> None:
        # Arrange：ASR 少了一次「沃尔」
        words, stream, char_to_word = make_asr(
            [("沃", 0, 100), ("尔", 100, 200), ("走", 200, 300)]
        )

        # Act
        anchors, stats, _ = match_hotword_anchors(
            ["沃尔"], "沃尔走沃尔", stream, char_to_word, words
        )

        # Assert
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["beginTime"], 0)
        self.assertEqual(stats[0].matched, 1)

    def test_out_of_order_occurrences_keep_one_anchor(self) -> None:
        # Arrange：参考是「沃尔 蜘蛛」，ASR 是「蜘蛛 沃尔」
        words, stream, char_to_word = make_asr(
            [("蜘蛛", 0, 100), ("沃尔", 100, 200)]
        )

        # Act
        anchors, _, _ = match_hotword_anchors(
            ["沃尔", "蜘蛛"], "沃尔蜘蛛", stream, char_to_word, words
        )

        # Assert
        self.assertEqual(len(anchors), 1)

    def test_same_reference_position_keeps_the_longest_term(self) -> None:
        # Arrange：两个词都从同一位置开始
        words, stream, char_to_word = make_asr([("蜘蛛丝", 0, 300)])

        # Act
        anchors, _, _ = match_hotword_anchors(
            ["蜘蛛", "蜘蛛丝"], "蜘蛛丝", stream, char_to_word, words
        )

        # Assert
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["normText"], "蜘蛛丝")

    def test_backward_time_anchor_is_dropped(self) -> None:
        # Arrange：第二次「沃尔」的时间早于第一次
        words, stream, char_to_word = make_asr(
            [
                ("沃", 300, 400),
                ("尔", 400, 500),
                ("沃", 100, 200),
                ("尔", 200, 300),
            ]
        )

        # Act
        anchors, _, dropped = match_hotword_anchors(
            ["沃尔"], "沃尔沃尔", stream, char_to_word, words
        )

        # Assert
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["beginTime"], 300)
        self.assertEqual(len(dropped), 1)

    def test_empty_terms_or_asr_returns_no_anchors(self) -> None:
        # Arrange
        words, stream, char_to_word = make_asr([("沃", 0, 100)])

        # Act
        empty_terms = match_hotword_anchors([], "沃尔", stream, char_to_word, words)
        empty_asr = match_hotword_anchors(["沃尔"], "沃尔", "", [], words)

        # Assert
        self.assertEqual(empty_terms, ([], [], []))
        self.assertEqual(empty_asr[0], [])
        self.assertEqual(empty_asr[1][0].matched, 0)


class ReferenceOffsetLookupTests(unittest.TestCase):
    def test_time_maps_to_the_word_start_in_reference(self) -> None:
        # Arrange：三个词，每词 2 个字符、1 秒
        words = [
            {"text": "ab", "begin_time": 0, "end_time": 1000},
            {"text": "cd", "begin_time": 1000, "end_time": 2000},
            {"text": "ef", "begin_time": 2000, "end_time": 3000},
        ]
        char_to_word = [0, 0, 1, 1, 2, 2]

        # Act
        lookup = build_reference_offset_lookup("abcdef", "abcdef", char_to_word, words)

        # Assert
        self.assertEqual(lookup(0), 0)
        self.assertEqual(lookup(1500), 2)
        self.assertEqual(lookup(9999), 4)

    def test_gap_in_asr_maps_to_the_next_matching_block(self) -> None:
        # Arrange：ASR 缺了 cd
        words = [
            {"text": "a", "begin_time": 0, "end_time": 1000},
            {"text": "b", "begin_time": 1000, "end_time": 2000},
            {"text": "e", "begin_time": 2000, "end_time": 3000},
            {"text": "f", "begin_time": 3000, "end_time": 4000},
            {"text": "g", "begin_time": 4000, "end_time": 5000},
            {"text": "h", "begin_time": 5000, "end_time": 6000},
        ]
        char_to_word = [0, 1, 2, 3, 4, 5]

        # Act
        lookup = build_reference_offset_lookup(
            "abcdefgh", "abefgh", char_to_word, words
        )

        # Assert
        self.assertEqual(lookup(2500), 4)

    def test_empty_asr_maps_to_zero(self) -> None:
        lookup = build_reference_offset_lookup("abc", "", [], [])

        self.assertEqual(lookup(1000), 0)


class SamplingTests(unittest.TestCase):
    def test_sampling_locks_first_and_last(self) -> None:
        # Arrange
        anchors = [
            {"beginTime": 0, "endTime": 1000},
            {"beginTime": 10_000, "endTime": 11_000},
            {"beginTime": 25_000, "endTime": 26_000},
            {"beginTime": 50_000, "endTime": 51_000},
            {"beginTime": 70_000, "endTime": 71_000},
        ]

        # Act
        sampled = sample_anchors(anchors)

        # Assert：10s 距首锚太近，50s 距尾锚太近
        self.assertEqual([anchor["beginTime"] for anchor in sampled], [0, 25_000, 70_000])

    def test_sampling_keeps_short_lists(self) -> None:
        anchors = [{"beginTime": 1000, "endTime": 2000}]

        self.assertEqual(sample_anchors(anchors), anchors)
        self.assertEqual(sample_anchors([]), [])


if __name__ == "__main__":
    unittest.main()
