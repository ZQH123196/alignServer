"""块级修复判据（2026-10-27）。

覆盖三件事，全部是纯函数判定，不碰 FA / 模型 / 网络：
1. `_repair_chunk_words` 的局部修复：只重摊零时长段、健康时间原样保留、标点继承前词 end。
2. `_rebuild_words` 的词锚定与 `_diagnose_chunk` 的结构判据：无词覆盖的字、零时长
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
    CHUNK_UNCOVERED_MIN_CHARS,
    CHUNK_ZERO_RUN_MIN,
    MAX_REPAIRED_CHUNKS_PER_REQUEST,
    CloudHotwordAlignPipeline,
    format_chunk_word_order,
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
    """坏块修复只重摊缺陷段（2026-10-28）。

    缺陷段 = 有效且零时长的连续 token。健康 token 的观测时间必须原样保留：
    整块推平会为了 18 个字毁掉同块 62 个可信时间（第 8 章 chunk[2]）。
    """

    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def repair(self, ground_truth, spans, duration=10.0, lang="zh", **kwargs):
        rebuilt = [{"text": text, "start": start, "end": end} for text, start, end in spans]
        return self.pipeline._repair_chunk_words(
            ground_truth, rebuilt, duration, lang, **kwargs
        )

    def test_healthy_block_keeps_its_observation_times(self):
        # 整块都有 FA 时间：修复不得改动任何时间（旧实现会把整块铺满 10s 窗口）。
        spans = [("甲", 0.0, 1.0), ("乙", 1.0, 2.0), ("丙", 2.0, 3.0)]

        words = self.repair("甲乙丙", spans)

        self.assertEqual([(w["text"], w["start"], w["end"]) for w in words], spans)

    def test_zero_run_is_spread_between_its_two_neighbours_only(self):
        # 5 个字里只有中间 3 个零时长：只摊 [甲.end=1.0, 戊.start=6.0] 这一窗。
        spans = [
            ("甲", 0.0, 1.0),
            ("乙", 1.0, 1.0),
            ("丙", 1.0, 1.0),
            ("丁", 1.0, 1.0),
            ("戊", 6.0, 7.0),
        ]

        words = self.repair("甲乙丙丁戊", spans)

        self.assertEqual((words[0]["start"], words[0]["end"]), (0.0, 1.0))
        self.assertEqual((words[4]["start"], words[4]["end"]), (6.0, 7.0))
        self.assertAlmostEqual(words[1]["start"], 1.0)
        self.assertAlmostEqual(words[1]["end"], 1.0 + 5.0 / 3)
        self.assertAlmostEqual(words[3]["end"], 6.0)

    def test_zero_run_at_the_tail_stops_at_end_sec(self):
        # 块尾零时长：右边界取 end_sec（下一块首个保留词），不铺到块窗口终点。
        spans = [("甲", 0.0, 1.0), ("乙", 1.0, 1.0), ("丙", 1.0, 1.0)]

        words = self.repair("甲乙丙", spans, duration=10.0, end_sec=4.0)

        self.assertAlmostEqual(words[2]["end"], 4.0)
        self.assertAlmostEqual(words[1]["end"] - words[1]["start"], 1.5)

    def test_zero_run_at_the_head_starts_at_start_sec(self):
        # 块首零时长：左边界取 start_sec（前块已输出最大 end）。
        spans = [("甲", 0.0, 0.0), ("乙", 0.0, 0.0), ("丙", 5.0, 6.0)]

        words = self.repair("甲乙丙", spans, start_sec=4.0)

        self.assertAlmostEqual(words[0]["start"], 4.0)
        self.assertAlmostEqual(words[1]["end"], 5.0)
        self.assertEqual((words[2]["start"], words[2]["end"]), (5.0, 6.0))

    def test_start_sec_does_not_move_internal_gaps(self):
        # start_sec 只作用于块首：块内缺口一律取邻居时间，不能被整块平移。
        spans = [
            ("甲", 20.0, 21.0),
            ("乙", 21.0, 21.0),
            ("丙", 21.0, 21.0),
            ("丁", 21.0, 21.0),
            ("戊", 26.0, 27.0),
        ]

        words = self.repair("甲乙丙丁戊", spans, start_sec=19.7)

        self.assertAlmostEqual(words[1]["start"], 21.0)
        self.assertAlmostEqual(words[3]["end"], 26.0)

    def test_punctuation_inherits_the_repaired_previous_end(self):
        spans = [("甲", 0.0, 1.0), ("乙", 1.0, 1.0), ("，", 1.0, 1.0), ("丙", 6.0, 7.0)]

        words = self.repair("甲乙，丙", spans)

        punctuation = words[2]
        self.assertEqual(punctuation["start"], punctuation["end"])
        self.assertAlmostEqual(punctuation["start"], words[1]["end"])
        self.assertAlmostEqual(words[1]["end"], 6.0)

    def test_leading_punctuation_follows_the_block_head_window(self):
        spans = [("「", 0.0, 0.0), ("甲", 2.0, 3.0)]

        words = self.repair("「甲", spans, start_sec=1.0)

        self.assertEqual((words[0]["start"], words[0]["end"]), (1.0, 1.0))

    def test_block_without_any_anchor_spreads_over_the_window(self):
        # 一个锚点都没有时没有可保留的信息：整段铺在 [start_sec, end_sec] 内。
        spans = [("甲", 0.0, 0.0), ("乙", 0.0, 0.0), ("丙", 0.0, 0.0), ("丁", 0.0, 0.0)]

        words = self.repair("甲乙丙丁", spans, duration=4.0, start_sec=1.0, end_sec=3.0)

        self.assertAlmostEqual(words[0]["start"], 1.0)
        self.assertAlmostEqual(words[-1]["end"], 3.0)
        self.assertAlmostEqual(words[0]["end"] - words[0]["start"], 0.5)

    def test_dense_zero_run_returns_none(self):
        # 1000 个 token 摊 0.5s → 每 token 0.5ms，低于 CHUNK_TOKEN_MIN_SPAN_SEC。
        spans = [("甲", 0.0, 0.0)] * 1000

        self.assertIsNone(self.repair("甲" * 1000, spans, duration=0.5))


class RepairSeamTests(unittest.TestCase):
    """修复必须尊重「已输出」的边界：重叠区由前块输出，接缝不能倒流。"""

    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def repair(self, ground_truth, spans, duration=10.0, **kwargs):
        rebuilt = [{"text": text, "start": start, "end": end} for text, start, end in spans]
        return self.pipeline._repair_chunk_words(ground_truth, rebuilt, duration, "zh", **kwargs)

    def test_keep_from_drops_already_emitted_prefix(self):
        spans = [("甲", 0.0, 1.0), ("乙", 1.0, 2.0), ("丙", 2.0, 2.0), ("丁", 2.0, 2.0)]

        words = self.repair("甲乙丙丁", spans, keep_from=2)

        self.assertEqual([w["text"] for w in words], ["丙", "丁"])

    def test_block_end_zero_run_never_passes_next_chunk(self):
        # ch8 实测：修复段尾部越过下一块首个保留词会造成回退。
        spans = [("甲", 0.0, 1.0), ("乙", 1.0, 1.0), ("丙", 1.0, 1.0), ("丁", 1.0, 1.0)]

        words = self.repair("甲乙丙丁", spans, duration=24.68, start_sec=1.44, end_sec=8.0)

        self.assertAlmostEqual(words[-1]["end"], 8.0)


class DiagnoseChunkTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def rebuild(self, ground_truth, fa_words, duration=3.0, lang="zh"):
        return self.pipeline._rebuild_words(ground_truth, fa_words, lang, duration)

    def diagnose(self, ground_truth, fa_words, duration=3.0, lang="zh"):
        rebuilt, stats = self.rebuild(ground_truth, fa_words, duration, lang)
        return self.pipeline._diagnose_chunk(
            ground_truth, rebuilt, fa_words, duration, lang, stats["uncovered"]
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
            ground_truth, rebuilt, fa_words, 2.0, "zh", stats["uncovered"]
        )[0]

        self.assertIsNone(verdict)
        self.assertEqual(stats["uncovered"], 0)
        self.assertEqual([round(w["start"], 3) for w in rebuilt], [0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
        self.assertEqual([round(w["end"], 3) for w in rebuilt], [0.3, 0.6, 0.9, 1.2, 1.5, 2.0])

    def test_many_uncovered_chars_is_a_defect(self):
        # FA 真丢字：剩余的字由缺口夹摊开，但数量达到阈值说明块内时间轴不可信。
        ground_truth = "甲乙丙丁戊己庚辛壬癸子丑"
        fa_words = [fa_word(0.0, 0.5, "甲"), fa_word(0.5, 1.0, "乙")]

        verdict, valid = self.diagnose(ground_truth, fa_words, 1.5)

        self.assertEqual(verdict, f"无词覆盖({CHUNK_UNCOVERED_MIN_CHARS}字)")
        self.assertEqual(valid, 12)

    def test_uncovered_chars_below_the_limit_are_tolerated(self):
        # 阈值边界：差 1 个字不判坏。2026-10-28 第 8 章（阈值 5 时三块被判坏，
        # 而同一章 ASR 覆盖率 98.6%，音频并没有缺字）。
        ground_truth = "甲乙丙丁戊己庚辛壬癸子"
        fa_words = [fa_word(0.0, 0.5, "甲"), fa_word(0.5, 1.0, "乙")]

        verdict, _ = self.diagnose(ground_truth, fa_words, 1.5)

        self.assertIsNone(verdict)

    def test_gap_is_spread_between_the_two_nearest_anchors_only(self):
        # 用户场景：30 个字里只有中间 11..22 出问题。两头夹只能摊这一段，
        # 前面 11 个字与后面 7 个字必须保留 FA 给出的时间。
        ground_truth = "".join(chr(0x4e00 + i) for i in range(30))
        fa_words = [
            fa_word(0.0, 1.1, ground_truth[0:11]),
            fa_word(2.4, 3.0, ground_truth[23:30]),
        ]

        rebuilt, stats = self.rebuild(ground_truth, fa_words, 3.0)

        self.assertEqual(stats["uncovered"], 12)
        # 锚点两侧保留观测时间
        self.assertAlmostEqual(rebuilt[0]["start"], 0.0, places=6)
        self.assertAlmostEqual(rebuilt[10]["end"], 1.1, places=6)
        self.assertAlmostEqual(rebuilt[23]["start"], 2.4, places=6)
        self.assertAlmostEqual(rebuilt[29]["end"], 3.0, places=6)
        # 缺口整段只用 [前锚 end, 后锚 start] 这一个窗口
        self.assertAlmostEqual(rebuilt[11]["start"], 1.1, places=6)
        self.assertAlmostEqual(rebuilt[22]["end"], 2.4, places=6)
        step = (2.4 - 1.1) / 12
        self.assertAlmostEqual(rebuilt[11]["end"], 1.1 + step, places=6)
        self.assertAlmostEqual(rebuilt[22]["start"], 2.4 - step, places=6)

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

        self.assertEqual(stats["uncovered"], 4)
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
        # 只靠 `无词覆盖 >= 10` 会放行，块尾落成零时长。
        ground_truth = "甲乙，丙"

        verdict, valid = self.diagnose(ground_truth, [], 4.0)

        self.assertEqual(verdict, "无词覆盖(3字)")
        self.assertEqual(valid, 3)

    def test_word_never_anchors_inside_another_word(self):
        # 前跳查找必须守住词边界：`art` 不能锚进 `party` 中间。
        ground_truth = "the party started"
        fa_words = [fa_word(0.0, 0.5, "the"), fa_word(0.5, 1.0, "art")]

        _, stats = self.rebuild(ground_truth, fa_words, 1.0, "en")

        self.assertEqual(stats["located_words"], 1)
        self.assertGreater(stats["uncovered"], 0)

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

        self.assertEqual(stats["uncovered"], 0)
        self.assertEqual([round(w["start"], 3) for w in rebuilt], [0.0, 0.5, 0.5, 1.5, 1.5])
        self.assertEqual([round(w["end"], 3) for w in rebuilt], [0.5, 0.5, 1.5, 1.5, 2.0])


class ChunkWordOrderTests(unittest.TestCase):
    """块被判坏时的词序对照（2026-10-28 第 8 章）。

    判「FA 丢词」还是「锚定拒绝」只能看这一行：词表里没有该段的词 → 模型侧丢词；
    词在表里但没对位 → `_locate_word` 的词边界拒绝。
    """

    def setUp(self):
        self.pipeline = CloudHotwordAlignPipeline(None, None, {})

    def word_order(self, ground_truth, fa_words):
        _, stats = self.pipeline._rebuild_words(ground_truth, fa_words, "zh", 3.0)
        return format_chunk_word_order(ground_truth, fa_words, stats["word_spans"])

    def test_lists_words_that_failed_to_locate(self):
        # 丁 不在原文里：词在表内、定位失败 —— 属锚定拒绝而不是丢词。
        fa_words = [fa_word(0.0, 1.0, "甲"), fa_word(1.0, 2.0, "丁"), fa_word(2.0, 3.0, "丙")]

        text = self.word_order("甲乙丙", fa_words)

        self.assertIn("FA词=3", text)
        self.assertIn("未对位词=1", text)
        self.assertIn("未对位词文本：「丁」", text)
        self.assertIn("块原文：「甲乙丙」", text)

    def test_reports_no_unlocated_word_explicitly(self):
        fa_words = [fa_word(0.0, 1.0, "甲"), fa_word(1.0, 3.0, "乙")]

        text = self.word_order("甲乙", fa_words)

        self.assertIn("未对位词=0", text)
        self.assertIn("未对位词文本：「无」", text)


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
            [record(0, 50, 10.0, None), record(1, 45, 10.0, "无词覆盖(18字)")], "req",
        )

        self.assertIsNone(defect)
        self.assertEqual(repaired, {1})

    def test_rate_out_of_band_is_defect_and_not_repaired(self):
        # 坏块 10/10 = 1 token/s，R=5 → r/R = 0.2，远低于下界
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 50, 10.0, None), record(1, 10, 10.0, "无词覆盖(7字)")], "req",
        )

        self.assertEqual(defect["reason"], "rate-out-of-band")
        self.assertEqual(repaired, set())
        self.assertIn(f"[{CHUNK_RATE_BAND_LOW}, {CHUNK_RATE_BAND_HIGH}]", defect["message"])
        # 结构原因不许被处置理由盖掉：两者要同时出现在诊断里
        self.assertIn("无词覆盖(7字)", defect["message"])
        self.assertIn("chunk[1] 0.0-10.0s", defect["message"])

    def test_more_than_cap_repaired_is_defect(self):
        records = [record(0, 50, 10.0, None)]
        for index in range(1, MAX_REPAIRED_CHUNKS_PER_REQUEST + 2):
            records.append(record(index, 45, 10.0, "无词覆盖(18字)"))

        defect, repaired = self.pipeline._decide_chunk_repairs(records, "req")

        self.assertEqual(defect["reason"], "too-many-repaired")
        self.assertEqual(len(repaired), MAX_REPAIRED_CHUNKS_PER_REQUEST + 1)

    def test_no_good_chunk_is_defect(self):
        # 全部块判坏 → 没有参考语速，一律不摊（摊了也无从判断是否合理）
        defect, repaired = self.pipeline._decide_chunk_repairs(
            [record(0, 20, 10.0, "无词覆盖(18字)"), record(1, 15, 10.0, "zero-run(5)")], "req",
        )

        self.assertEqual(defect["reason"], "no-good-chunk")
        self.assertEqual(repaired, set())

    def test_cap_is_two(self):
        # 上限的**唯一归属**是服务端常量：分块是服务端内部概念，客户端无从知道块数。
        self.assertEqual(MAX_REPAIRED_CHUNKS_PER_REQUEST, 2)


if __name__ == "__main__":
    unittest.main()
