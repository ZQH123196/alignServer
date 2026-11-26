import unittest

from fallback_chunks import build_fallback_chunks


class ZeroAnchorTests(unittest.TestCase):
    def test_splits_audio_into_fixed_windows(self) -> None:
        # Arrange
        text = "甲" * 100

        # Act
        chunks = build_fallback_chunks(text, 120_000)

        # Assert
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [(chunk["audioStartMs"], chunk["audioEndMs"]) for chunk in chunks],
            [(0, 60_000), (60_000, 120_000)],
        )
        self.assertEqual(
            [(chunk["origStart"], chunk["origEnd"]) for chunk in chunks],
            [(0, 50), (50, 100)],
        )

    def test_snaps_to_nearest_sentence_end(self) -> None:
        # Arrange：句号在 50 左前方，边界应落在句号之后
        text = "甲" * 48 + "。" + "乙" * 51

        # Act
        chunks = build_fallback_chunks(text, 120_000)

        # Assert
        self.assertEqual(chunks[0]["origEnd"], 49)
        self.assertEqual(chunks[1]["origStart"], 49)

    def test_snaps_forward_too(self) -> None:
        # Arrange：句号在 50 右后方
        text = "甲" * 52 + "！" + "乙" * 47

        # Act
        chunks = build_fallback_chunks(text, 120_000)

        # Assert
        self.assertEqual(chunks[0]["origEnd"], 53)

    def test_short_audio_is_one_chunk(self) -> None:
        # Act
        chunks = build_fallback_chunks("甲" * 100, 30_000)

        # Assert
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["audioEndMs"], 30_000)

    def test_text_shorter_than_window_count_is_not_split_into_empty_chunks(self) -> None:
        # Arrange：音频 120s 要 2 段，但只有 1 个字
        chunks = build_fallback_chunks("甲", 120_000)

        # Assert
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["origStart"], 0)
        self.assertEqual(chunks[0]["origEnd"], 1)

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(build_fallback_chunks("", 120_000), [])


class SingleAnchorTests(unittest.TestCase):
    def test_anchor_is_the_only_internal_boundary(self) -> None:
        # Arrange：180s 音频，锚点在 90s、文本第 50 字
        text = "甲" * 100

        # Act
        chunks = build_fallback_chunks(
            text, 180_000, anchor_ms=90_000, anchor_text_offset=50
        )

        # Assert：两侧各 2 段，90s / 第 50 字保持不变
        self.assertEqual(len(chunks), 4)
        self.assertEqual(
            [chunk["audioStartMs"] for chunk in chunks],
            [0, 60_000, 90_000, 150_000],
        )
        self.assertEqual(
            [(chunk["origStart"], chunk["origEnd"]) for chunk in chunks],
            [(0, 33), (33, 50), (50, 83), (83, 100)],
        )

    def test_anchor_at_boundary_falls_back_to_zero_anchor(self) -> None:
        # Arrange：锚点贴在音频开头，无法当内部边界
        chunks = build_fallback_chunks(
            "甲" * 100, 120_000, anchor_ms=0, anchor_text_offset=50
        )

        # Assert
        self.assertEqual(len(chunks), 2)

    def test_short_sides_are_not_split(self) -> None:
        # Arrange：总长 60s，锚点在 30s
        chunks = build_fallback_chunks(
            "甲" * 100, 60_000, anchor_ms=30_000, anchor_text_offset=50
        )

        # Assert
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [(chunk["audioStartMs"], chunk["audioEndMs"]) for chunk in chunks],
            [(0, 30_000), (30_000, 60_000)],
        )


class TextOffsetLookupTests(unittest.TestCase):
    def test_lookup_overrides_proportional_split(self) -> None:
        # Arrange：60s 边界映射到第 80 字，而不是比例估算的第 50 字
        text = "甲" * 100

        # Act
        chunks = build_fallback_chunks(
            text, 120_000, text_offset_for_time=lambda ms: 80 if ms == 60_000 else 0
        )

        # Assert
        self.assertEqual(
            [(chunk["origStart"], chunk["origEnd"]) for chunk in chunks],
            [(0, 80), (80, 100)],
        )

    def test_lookup_boundaries_are_not_snapped_to_sentence_end(self) -> None:
        # Arrange：第 49 字是句号，lookup 给 50；不能为了句末标点把边界推到 49
        text = "甲" * 48 + "。" + "乙" * 51

        # Act
        chunks = build_fallback_chunks(
            text, 120_000, text_offset_for_time=lambda ms: 50
        )

        # Assert
        self.assertEqual(chunks[0]["origEnd"], 50)

    def test_non_monotonic_lookup_is_clamped(self) -> None:
        # Arrange：第二个查询回退到 10，必须被抬到上一段之后
        text = "甲" * 100
        mapping = {60_000: 90, 120_000: 10}

        # Act
        chunks = build_fallback_chunks(
            text, 180_000, text_offset_for_time=lambda ms: mapping.get(ms, 0)
        )

        # Assert
        boundaries = [chunks[0]["origStart"]] + [chunk["origEnd"] for chunk in chunks]
        self.assertEqual(boundaries, sorted(boundaries))
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], 100)
        self.assertEqual(boundaries[2], 91)


if __name__ == "__main__":
    unittest.main()
