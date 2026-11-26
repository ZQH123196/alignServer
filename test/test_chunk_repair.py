"""块级修复判据（2026-10-27）。

覆盖三件事，全部是纯函数判定，不碰 FA / 模型 / 网络：
1. `_repair_chunk_words` 的均分形状：首尾贴块边界、按 token 长度加权、标点继承前词 end。
2. `_rebuild_words` 的词锚定与 `_diagnose_chunk` 的结构判据：未锚定字符、零时长
   有效 token、词流没覆盖块窗口。
3. `_decide_chunk_repairs` 的处置：容差带、每请求上限、无参考块。

判据只比相对比值（token/秒 的 r/R），故同一组断言对中文（逐字）与英文（按词）都成立 ——
这也是「多语言只写一套阈值」的守门：若有人把判据改成绝对 token/秒，英文用例会先炸。

为什么直接测私有方法：`process_audio` 需要 GPU、云端 ASR 与音频文件，单测无法驱动。
被直接测的 `_repair_chunk_words` / `_diagnose_chunk` 无 I/O、无状态，等价于纯函数；
`_decide_chunk_repairs` 的返回值就是判定结果，日志只是它的副产物。
"""

import sys
import types
import unittest
from types import SimpleNamespace

if "main" not in sys.modules:
    _main_stub = types.ModuleType("main")
    for _name in ("log", "log_info", "log_warn", "log_error", "log_block"):
        setattr(_main_stub, _name, lambda *args, **kwargs: None)
    _main_stub.get_task_info = lambda: {}
    sys.modules["main"] = _main_stub

from cloud_hotword import (  # noqa: E402
    CHUNK_RATE_BAND_HIGH,
    CHUNK_RATE_BAND_LOW,
    CHUNK_ZERO_RUN_MIN,
    MAX_REPAIRED_CHUNKS_PER_REQUEST,
    CloudHotwordAlignPipeline,
)


def fa_word(start, end, text=""):
    return SimpleNamespace(start_time=start, end_time=end, text=text)


def record(index, valid_tokens, duration, verdict, audio_start=0.0):
    return {
        "index": index,
        "audio_start": audio_start,
        "duration": duration,
        "verdict": verdict,
        "valid_tokens": valid_tokens,
    }


class RepairChunkWordsTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def test_chinese_tokens_split_evenly_and_reach_block_end(self):
        # 中文每 token 长度为 1，权重相同 → 退化为均分。
        words = self.pipeline._repair_chunk_words("甲乙丙丁", 4.0, "zh")

        self.assertEqual([w["text"] for w in words], ["甲", "乙", "丙", "丁"])
        self.assertEqual(words[0]["start"], 0.0)
        self.assertAlmostEqual(words[0]["end"], 1.0)
        self.assertAlmostEqual(words[-1]["end"], 4.0)

    def test_punctuation_inherits_previous_end(self):
        words = self.pipeline._repair_chunk_words("甲乙，丙", 6.0, "zh")

        punctuation = next(w for w in words if w["text"] == "，")
        previous = next(w for w in words if w["text"] == "乙")
        self.assertEqual(punctuation["start"], punctuation["end"])
        self.assertAlmostEqual(punctuation["start"], previous["end"])
        # 标点不占时间：有效 token 仍把整块窗口摊满
        self.assertAlmostEqual(words[-1]["end"], 6.0)

    def test_leading_punctuation_inherits_block_start(self):
        words = self.pipeline._repair_chunk_words("「甲", 2.0, "zh")

        self.assertEqual(words[0]["start"], 0.0)
        self.assertEqual(words[0]["end"], 0.0)

    def test_english_words_weighted_by_length(self):
        # 长期望 > 短期望：均分不等于公平，长度才是可比的量。
        words = self.pipeline._repair_chunk_words("a extraordinarily", 10.0, "en")
        short = next(w for w in words if w["text"] == "a")
        long = next(w for w in words if w["text"] == "extraordinarily")

        self.assertGreater(long["end"] - long["start"], short["end"] - short["start"])
        self.assertAlmostEqual(words[-1]["end"], 10.0)

    def test_punctuation_only_tail_collapses_to_window_start(self):
        # 切掉重叠区后只剩标点：它们不占时间，一起塌到窗口起点，不返回 None。
        words = self.pipeline._repair_chunk_words("……", 3.0, "zh")

        self.assertEqual([(w["text"], w["start"], w["end"]) for w in words],
                         [("…", 0.0, 0.0), ("…", 0.0, 0.0)])

    def test_step_below_one_millisecond_returns_none(self):
        # 1000 个 token 摊 0.5s → 每 token 0.5ms，低于下限
        self.assertIsNone(self.pipeline._repair_chunk_words("甲" * 1000, 0.5, "zh"))


class RepairSeamTests(unittest.TestCase):
    """均分必须尊重「已输出」的边界：重叠区由前块输出，接缝不能倒流。"""

    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def test_keep_from_drops_already_emitted_prefix(self):
        words = self.pipeline._repair_chunk_words("甲乙丙丁", 4.0, "zh", keep_from=2)

        self.assertEqual([w["text"] for w in words], ["丙", "丁"])

    def test_start_sec_never_precedes_previous_chunk_end(self):
        # chunk[1] 从 18.26s 开始，但前块已输出到 19.70s：均分必须从 19.70s 起。
        words = self.pipeline._repair_chunk_words("甲乙丙丁", 23.8, "zh", keep_from=2, start_sec=1.44)

        self.assertAlmostEqual(words[0]["start"], 1.44)
        self.assertAlmostEqual(words[-1]["end"], 23.8)

    def test_end_sec_caps_the_repair_so_the_next_chunk_does_not_regress(self):
        # ch8 实测：块尾部铺到 46.86s，而下一块首个保留词在 46.22s → 回退 0.64s 越过容差。
        # 上限取下一块首个保留词的起点后，本块尾部不会越过它。
        words = self.pipeline._repair_chunk_words("甲乙丙丁", 24.68, "zh", start_sec=1.44, end_sec=8.0)

        self.assertAlmostEqual(words[-1]["end"], 8.0)

    def test_window_collapsed_returns_none(self):
        self.assertIsNone(self.pipeline._repair_chunk_words("甲乙", 1.0, "zh", start_sec=1.0))


class DiagnoseChunkTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def rebuild(self, ground_truth, fa_words, duration=3.0, lang="zh"):
        return self.pipeline._rebuild_words(ground_truth, fa_words, lang, duration)

    def diagnose(self, ground_truth, fa_words, duration=3.0, lang="zh"):
        rebuilt, stats = self.rebuild(ground_truth, fa_words, duration, lang)
        return self.pipeline._diagnose_chunk(
            ground_truth, rebuilt, fa_words, duration, lang, stats["unanchored"]
        )

    def test_healthy_chunk_passes(self):
        ground_truth = "甲乙丙"
        fa_words = [fa_word(0.1, 1.0, "甲"), fa_word(1.0, 2.0, "乙"), fa_word(2.0, 2.9, "丙")]

        verdict, valid = self.diagnose(ground_truth, fa_words)

        self.assertIsNone(verdict)
        self.assertEqual(valid, 3)

    def test_number_span_is_anchored_not_index_paired(self):
        # 2026-10-27 回归守门：`1500` 在 FA 词表里是**一个**词，参考文本按字切成四份。
        # 旧实现按序号硬贴：数字之后的字全部错位，块尾落成零时长（实测三章 54 块里
        # 15 块因此被判 fa-short，进而整章作废、回退重跑 tts）。
        ground_truth = "价1500元"
        fa_words = [fa_word(0.0, 0.3, "价"), fa_word(0.3, 1.5, "1500"), fa_word(1.5, 2.0, "元")]

        rebuilt, stats = self.rebuild(ground_truth, fa_words, 2.0)
        verdict = self.pipeline._diagnose_chunk(
            ground_truth, rebuilt, fa_words, 2.0, "zh", stats["unanchored"]
        )[0]

        self.assertIsNone(verdict)
        self.assertEqual(stats["unanchored"], 0)
        self.assertEqual([round(w["start"], 3) for w in rebuilt], [0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
        self.assertEqual([round(w["end"], 3) for w in rebuilt], [0.3, 0.6, 0.9, 1.2, 1.5, 2.0])

    def test_fa_dropped_words_become_unanchored(self):
        # FA 真丢字：剩余的字由缺口夹摊开，但数量超过阈值说明块内时间轴不可信。
        ground_truth = "甲乙丙丁戊己庚辛壬癸"
        fa_words = [fa_word(0.0, 0.5, "甲"), fa_word(0.5, 1.0, "乙"), fa_word(1.0, 1.5, "丙")]

        verdict, valid = self.diagnose(ground_truth, fa_words, 1.5)

        self.assertEqual(verdict, "unanchored(7)")
        self.assertEqual(valid, 10)

    def test_small_fa_gap_is_tolerated(self):
        # 少 2 个字只是尾端毫秒级截断：为它整块均分会毁掉块内本来准确的时间。
        ground_truth = "甲乙丙丁戊"
        fa_words = [fa_word(0.0, 1.0, "甲"), fa_word(1.0, 2.0, "乙"), fa_word(2.0, 2.9, "丙")]

        verdict, _ = self.diagnose(ground_truth, fa_words, 3.0)

        self.assertIsNone(verdict)

    def test_zero_length_run_is_a_defect(self):
        # 成段零时长：FA 对这整段没给出对齐区间。
        ground_truth = "甲" + "乙" * CHUNK_ZERO_RUN_MIN + "丙"
        fa_words = [fa_word(0.0, 0.5, "甲")]
        fa_words += [fa_word(0.5, 0.5, "乙") for _ in range(CHUNK_ZERO_RUN_MIN)]
        fa_words.append(fa_word(0.5, 1.0, "丙"))

        verdict, _ = self.diagnose(ground_truth, fa_words, 1.0)

        self.assertEqual(verdict, f"zero-run({CHUNK_ZERO_RUN_MIN})")

    def test_scattered_zero_length_tokens_are_tolerated(self):
        # 2026-10-27 回归守门：FA 逐字估边，零星 start == end 是常态。
        # 按「存在即坏」判定时实测 16 块里 12 块被误报，正常块被均分反而毁掉准确时间。
        ground_truth = "甲乙丙丁戊己庚辛"
        fa_words = [
            fa_word(0.0, 0.5, "甲"), fa_word(0.5, 0.5, "乙"),  # 单个零时长
            fa_word(0.5, 1.0, "丙"), fa_word(1.0, 1.5, "丁"),
            fa_word(1.5, 1.5, "戊"),                            # 再来一个
            fa_word(1.5, 2.0, "己"), fa_word(2.0, 2.5, "庚"), fa_word(2.5, 2.9, "辛"),
        ]

        verdict, _ = self.diagnose(ground_truth, fa_words, 3.0)

        self.assertIsNone(verdict)

    def test_tail_gap_beyond_tolerance(self):
        # 字全覆盖、无零长，但末词离块终点 3s：这段音频的发音没被消费。
        ground_truth = "甲乙"
        fa_words = [fa_word(0.0, 1.0, "甲"), fa_word(1.0, 2.0, "乙")]

        verdict, _ = self.diagnose(ground_truth, fa_words, 5.0)

        self.assertTrue(verdict.startswith("tail-gap"))

    def test_punctuation_only_chunk_is_not_a_defect(self):
        # 纯标点块没有可发音 token：没有时间可摊，也不算零长有效 token。
        ground_truth = "……"

        verdict, valid = self.diagnose(ground_truth, [], 1.0)

        self.assertIsNone(verdict)
        self.assertEqual(valid, 0)

    def test_gap_split_by_punctuation_still_gets_time(self):
        # 2026-10-27 审查 M2 回归守门：缺口被标点打断时，第二段不能塌成零时长。
        # 按「连续有效段」逐个填的写法里，前一段先把窗口铺到下一个锚点，
        # 后一段就只能从合成值铺到同一点（实测 `甲乙，丙丁戊` 只剩 戊 时，丙丁落成 (5.0, 5.0)）。
        ground_truth = "甲乙，丙丁戊"
        fa_words = [fa_word(5.0, 6.0, "戊")]

        rebuilt, stats = self.rebuild(ground_truth, fa_words, 6.0)

        self.assertEqual(stats["unanchored"], 4)
        # 甲 乙 丙 丁 共用同一对锚点（0.0, 5.0），一起摊
        self.assertEqual(
            [round(w["start"], 3) for w in rebuilt],
            [0.0, 1.25, 2.5, 2.5, 3.75, 5.0],
        )
        self.assertEqual(
            [round(w["end"], 3) for w in rebuilt],
            [1.25, 2.5, 2.5, 3.75, 5.0, 6.0],
        )
        for word in rebuilt:
            if word["text"] in "，":
                continue
            self.assertLess(word["start"], word["end"])

    def test_no_fa_words_on_short_block_is_a_defect(self):
        # 2026-10-27 审查 M3 回归守门：FA 一个词都没给、块又短（有效字 < 阈值）时，
        # 只靠 `unanchored >= 5` 会放行，块尾落成零时长。
        ground_truth = "甲乙，丙"

        verdict, valid = self.diagnose(ground_truth, [], 4.0)

        self.assertEqual(verdict, "unanchored(3)")
        self.assertEqual(valid, 3)

    def test_word_never_anchors_inside_another_word(self):
        # 前跳查找必须守住词边界：`art` 不能锚进 `party` 中间。
        ground_truth = "the party started"
        fa_words = [fa_word(0.0, 0.5, "the"), fa_word(0.5, 1.0, "art")]

        _, stats = self.rebuild(ground_truth, fa_words, 1.0, "en")

        self.assertEqual(stats["located_words"], 1)
        self.assertGreater(stats["unanchored"], 0)

    def test_digit_word_never_anchors_inside_a_longer_number(self):
        # 同一类错锚：`500` 不能锚进 `15000` 中间。
        ground_truth = "共15000元"
        fa_words = [fa_word(0.0, 0.5, "共"), fa_word(0.5, 1.0, "500")]

        _, stats = self.rebuild(ground_truth, fa_words, 1.0)

        self.assertEqual(stats["located_words"], 1)

    def test_english_word_token_takes_its_own_char_span(self):
        # 英文按词切：一个 token 覆盖多个字符，时间取它覆盖字符的首尾。
        ground_truth = "price 1500 ok"
        fa_words = [fa_word(0.0, 0.5, "price"), fa_word(0.5, 1.5, "1500"), fa_word(1.5, 2.0, "ok")]

        rebuilt, stats = self.rebuild(ground_truth, fa_words, 2.0, "en")

        self.assertEqual(stats["unanchored"], 0)
        self.assertEqual([round(w["start"], 3) for w in rebuilt], [0.0, 0.5, 0.5, 1.5, 1.5])
        self.assertEqual([round(w["end"], 3) for w in rebuilt], [0.5, 0.5, 1.5, 1.5, 2.0])


class DecideChunkRepairsTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def test_healthy_request_has_no_defect(self):
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 50, 10.0, None), record(1, 50, 10.0, None)], "req",
        )

        self.assertIsNone(defect)
        self.assertEqual(repaired, set())

    def test_bad_chunk_within_band_is_repaired(self):
        # R = 50/10 = 5 token/s；坏块 45/10 = 4.5 → r/R = 0.9，在带内
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 50, 10.0, None), record(1, 45, 10.0, "unanchored")], "req",
        )

        self.assertIsNone(defect)
        self.assertEqual(repaired, {1})

    def test_rate_out_of_band_is_defect_and_not_repaired(self):
        # 坏块 10/10 = 1 token/s，R=5 → r/R = 0.2，远低于下界
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 50, 10.0, None), record(1, 10, 10.0, "unanchored(7)")], "req",
        )

        self.assertEqual(defect["reason"], "rate-out-of-band")
        self.assertEqual(repaired, set())
        self.assertIn(f"[{CHUNK_RATE_BAND_LOW}, {CHUNK_RATE_BAND_HIGH}]", defect["message"])
        # 结构原因不许被处置理由盖掉：两者要同时出现在诊断里
        self.assertIn("unanchored(7)", defect["message"])
        self.assertIn("chunk[1] 0.0-10.0s", defect["message"])

    def test_more_than_cap_repaired_is_defect(self):
        records = [record(0, 50, 10.0, None)]
        for index in range(1, MAX_REPAIRED_CHUNKS_PER_REQUEST + 2):
            records.append(record(index, 45, 10.0, "unanchored"))

        defect, repaired = self.pipeline._decide_chunk_repairs(records, "req")

        self.assertEqual(defect["reason"], "too-many-repaired")
        self.assertEqual(len(repaired), MAX_REPAIRED_CHUNKS_PER_REQUEST + 1)

    def test_no_good_chunk_is_defect(self):
        # 全部块判坏 → 没有参考语速，一律不摊（摊了也无从判断是否合理）
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 20, 10.0, "unanchored"), record(1, 15, 10.0, "zero-run(5)")], "req",
        )

        self.assertEqual(defect["reason"], "no-good-chunk")
        self.assertEqual(repaired, set())

    def test_cap_is_two(self):
        # 上限的**唯一归属**是服务端常量：分块是服务端内部概念，客户端无从知道块数。
        self.assertEqual(MAX_REPAIRED_CHUNKS_PER_REQUEST, 2)


if __name__ == "__main__":
    unittest.main()
