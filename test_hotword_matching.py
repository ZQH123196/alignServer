import unittest

from hotword_matching import (
    assess_reference_coverage,
    build_asr_stream,
    find_best_span,
    find_tail_matches,
    select_monotonic_chain,
)


def normalize(text: str, language: str) -> str:
    del language
    return "".join(char.lower() for char in text if not char.isspace())


class HotwordMatchingTests(unittest.TestCase):
    def test_asr_stream_tracks_each_normalized_character_to_its_word(self) -> None:
        # Arrange
        words = [{"text": "Hello"}, {"text": "world"}]

        # Act
        stream, char_to_word = build_asr_stream(words, "en", normalize)

        # Assert
        self.assertEqual(stream, "helloworld")
        self.assertEqual(char_to_word, [0] * 5 + [1] * 5)

    def test_empty_asr_is_rejected_even_for_short_reference(self) -> None:
        # Arrange
        reference = "short reference"

        # Act
        coverage = assess_reference_coverage(reference, [], "en", normalize)

        # Assert
        self.assertFalse(coverage["ok"])
        self.assertIn("empty_asr", coverage["reasons"])
        self.assertEqual(coverage["coverageRatio"], 0.0)

    def test_fuzzy_span_respects_the_forward_search_cursor(self) -> None:
        # Arrange
        haystack = "prefixalfabetasuffix"

        # Act
        span = find_best_span("alphabeta", haystack, start=6)

        # Assert
        self.assertIsNotNone(span)
        self.assertEqual(span["start"], 6)
        self.assertEqual(span["matchType"], "fuzzy")

    def test_tail_matching_recovers_fuzzy_hotwords_in_reference_order(self) -> None:
        # Arrange
        hotwords = [
            {"text": "All personnel enter deep sleep", "normText": normalize("All personnel enter deep sleep", "en"), "offsetInFullNorm": 100},
            {"text": "calling out to him", "normText": normalize("calling out to him", "en"), "offsetInFullNorm": 140},
        ]
        existing = [
            {
                "normText": normalize("The door opened", "en"),
                "offsetInFullNorm": 0,
                "asrOffset": 0,
                "asrEndOffset": 13,
                "beginTime": 0,
                "endTime": 1_000,
            }
        ]
        words = [
            {"text": "The", "begin_time": 0, "end_time": 300},
            {"text": "door", "begin_time": 300, "end_time": 600},
            {"text": "opened", "begin_time": 600, "end_time": 1_000},
            {"text": "All", "begin_time": 10_000, "end_time": 10_200},
            {"text": "personnel", "begin_time": 10_200, "end_time": 10_500},
            {"text": "entered", "begin_time": 10_500, "end_time": 10_800},
            {"text": "deep", "begin_time": 10_800, "end_time": 11_000},
            {"text": "sleep", "begin_time": 11_000, "end_time": 11_300},
            {"text": "calling", "begin_time": 12_000, "end_time": 12_300},
            {"text": "out", "begin_time": 12_300, "end_time": 12_500},
            {"text": "for", "begin_time": 12_500, "end_time": 12_700},
            {"text": "him", "begin_time": 12_700, "end_time": 13_000},
        ]

        # Act
        matched = find_tail_matches(hotwords, words, "en", normalize, existing)

        # Assert
        self.assertEqual(
            [item["text"] for item in matched],
            ["All personnel enter deep sleep", "calling out to him"],
        )
        self.assertTrue(all(item["matchType"] == "fuzzy" for item in matched))

    def test_monotonic_chain_discards_backward_anchor_and_marks_interior(self) -> None:
        # Arrange
        matched = [
            {"text": "first", "offsetInFullNorm": 0, "asrOffset": 0, "beginTime": 0, "matchScore": 1.0},
            {"text": "backward", "offsetInFullNorm": 5, "asrOffset": 0, "beginTime": 500, "matchScore": 1.0},
            {"text": "middle", "offsetInFullNorm": 10, "asrOffset": 10, "beginTime": 1_000, "matchScore": 1.0},
            {"text": "last", "offsetInFullNorm": 20, "asrOffset": 20, "beginTime": 2_000, "matchScore": 1.0},
        ]

        # Act
        valid, discarded = select_monotonic_chain(matched, "beginTime")

        # Assert
        self.assertEqual([item["text"] for item in valid], ["first", "middle", "last"])
        self.assertEqual([item["text"] for item in discarded], ["backward"])
        self.assertEqual(valid[1]["anchorConfidence"], "high")

    def test_reference_coverage_rejects_internal_gap(self) -> None:
        # Arrange
        tokens = [f"segment{index:03d}" for index in range(100)]
        reference = " ".join(tokens)
        words = [{"text": tokens[index]} for index in list(range(40)) + list(range(55, 100))]

        # Act
        coverage = assess_reference_coverage(reference, words, "en", normalize)

        # Assert
        self.assertFalse(coverage["ok"])
        self.assertIn("internal_reference_gap", coverage["reasons"])

    def test_unmatched_reference_is_reported_as_boundary_gap(self) -> None:
        # Arrange
        reference = "alpha " + ("reference " * 8) + "omega"

        # Act
        coverage = assess_reference_coverage(
            reference,
            [{"text": "1234567890"}],
            "en",
            normalize,
        )

        # Assert
        self.assertFalse(coverage["ok"])
        self.assertIn("leading_reference_gap", coverage["reasons"])
        self.assertIn("trailing_reference_gap", coverage["reasons"])


if __name__ == "__main__":
    unittest.main()
