import sys
import types
import unittest

if "main" not in sys.modules:
    _main_stub = types.ModuleType("main")
    for _name in ("log", "log_info", "log_warn", "log_error", "log_block"):
        setattr(_main_stub, _name, lambda *args, **kwargs: None)
    _main_stub.get_task_info = lambda: {}
    sys.modules["main"] = _main_stub

from cloud_hotword import norm_for_match, norm_for_match_with_map  # noqa: E402
from term_selection import select_hotword_terms  # noqa: E402


def select(text: str, lang: str = "zh", **kwargs) -> list:
    norm_text, norm_map = norm_for_match_with_map(text, lang)
    return select_hotword_terms(
        text,
        lang,
        normalize=norm_for_match,
        norm_text=norm_text,
        norm_map=norm_map,
        **kwargs,
    )


class TermCandidateTests(unittest.TestCase):
    def test_keeps_repeated_nouns_only(self) -> None:
        # Arrange：动词、单字名词、只出现一次的名词都不该入选
        text = "沃尔走进森林，沃尔看见板甲。士兵拿着板甲，士兵笑了。巨蜘蛛出现。"

        # Act
        terms = select(text)

        # Assert
        self.assertEqual([term.text for term in terms], ["士兵", "板甲", "沃尔"])
        self.assertEqual([term.count for term in terms], [2, 2, 2])

    def test_offsets_follow_normalized_text(self) -> None:
        # Arrange：标点在归一化文本里被删掉，偏移必须是归一化下标
        text = "蜘蛛，蜘蛛。士兵，士兵。"

        # Act
        terms = select(text)

        # Assert
        spider = next(term for term in terms if term.text == "蜘蛛")
        self.assertEqual((spider.first_offset, spider.last_offset), (0, 2))
        self.assertEqual(spider.count, 2)

    def test_requires_two_occurrences(self) -> None:
        # Act
        terms = select("蜘蛛走了，森林很安静。")

        # Assert
        self.assertEqual(terms, [])

    def test_english_uses_word_tokens(self) -> None:
        # Arrange
        text = "The dragon guards the gate. The dragon sleeps. A dragon wakes."

        # Act
        terms = select(text, "en")

        # Assert
        self.assertEqual([term.text for term in terms], ["dragon"])
        self.assertEqual(terms[0].count, 3)


class TailFirstTests(unittest.TestCase):
    def test_tail_slots_ignore_count_and_use_last_occurrence(self) -> None:
        # Arrange：蜘蛛 3 次但末次位置更靠前，士兵 2 次但更靠后
        text = "蜘蛛蜘蛛蜘蛛。甲乙。士兵。士兵。"

        # Act
        terms = select(text)

        # Assert
        self.assertEqual([term.text for term in terms], ["士兵", "蜘蛛"])

    def test_remaining_slots_use_count_order(self) -> None:
        # Arrange
        text = "沃尔。沃尔。士兵。士兵。将军。将军。国王。国王。国王。"

        # Act
        terms = select(text)

        # Assert
        self.assertEqual(
            [term.text for term in terms], ["国王", "将军", "士兵", "沃尔"]
        )

    def test_fewer_than_six_candidates_is_allowed(self) -> None:
        # Act
        terms = select("蜘蛛蜘蛛")

        # Assert
        self.assertEqual([term.text for term in terms], ["蜘蛛"])

    def test_zero_candidates_is_allowed(self) -> None:
        # Act
        terms = select("蜘蛛走了")

        # Assert
        self.assertEqual(terms, [])

    def test_top_n_caps_result_length(self) -> None:
        # Arrange：6 个重复名词
        text = "沃尔。沃尔。士兵。士兵。将军。将军。国王。国王。蜘蛛。蜘蛛。板甲。板甲。"

        # Act
        terms = select(text, top_n=4)

        # Assert
        self.assertEqual(len(terms), 4)


if __name__ == "__main__":
    unittest.main()
