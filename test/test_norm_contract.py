"""对齐侧归一化判据（2026-10-18 第 8 章复读事故的配套改动）。

事故链路：TTS 把 `〖〖` 之类不可发音字符原样送进 MOSS-TTS-v1.5，模型念出噪声音节；
而对齐侧 `norm_for_match` 只看「不是 P/S 类就保留」，`C` 组（零宽空格、BOM）会留在
参考文本里，与实送文本不一致。

本文件锁三件事：
1. 归一化只保留 `L`/`N`/`M` 三类 —— 不可见字符不再凭空进参考文本；
2. 全角字母数字先转半角再匹配，与 server 端 `normalizeFullwidthAlnum` 逐字同源；
3. 标点仍然全部删除，且不做任何读音替换（替换会让 ASR 多出参考文本没有的字）。
"""

import sys
import types

# cloud_hotword 顶层 `from main import log...`；pytest 下 `__main__` 是 pytest 本身，
# 其兼容分支会把 pytest 塞成 `main` 导致 ImportError。这里先注入只含日志函数的桩。
if "main" not in sys.modules:
    _main_stub = types.ModuleType("main")
    for _name in ("log", "log_info", "log_warn", "log_error", "log_block"):
        setattr(_main_stub, _name, lambda *args, **kwargs: None)
    _main_stub.get_task_info = lambda: {}
    sys.modules["main"] = _main_stub

from cloud_hotword import (  # noqa: E402
    _halfwidth_alnum,
    _is_match_char,
    norm_for_match,
    norm_for_match_with_map,
    norm_span_to_original_span,
)


def test_panel_brackets_are_not_matchable() -> None:
    # 对齐侧本来就删 P/S 类；这里锁住 `〖〗` 确实属于 P 类
    for ch in "〖〗【】〔〕〈〉《》「」『』":
        assert not _is_match_char(ch), ch


def test_invisible_chars_are_not_matchable() -> None:
    # 零宽空格、BOM、软连字符：旧判据（not P/S）会保留它们
    for ch in "\u200b\ufeff\u00ad\u200e\u200f":
        assert not _is_match_char(ch), ch


def test_letters_digits_marks_are_matchable() -> None:
    for ch in "沃尔abc１２3":
        assert _is_match_char(ch), ch
    # 组合标记（声调、变音符号）必须保留，否则越南语 / 泰语会丢字
    assert _is_match_char("\u0301")


def test_norm_drops_invisible_and_punctuation() -> None:
    assert norm_for_match("甲\u200b乙\ufeff丙。", "zh") == "甲乙丙"


def test_norm_keeps_halfwidth_alnum_only() -> None:
    # 全角字母数字转半角；全角标点仍然删除
    assert norm_for_match("ＡＢＣ１２３，ｄｅｆ", "zh") == "abc123def"


def test_halfwidth_alnum_is_one_to_one() -> None:
    # 长度不变，`norm_for_match_with_map` 的下标映射才不会被顶歪
    for ch in "Ａ１ａ　":
        assert len(_halfwidth_alnum(ch)) == 1, ch


def test_norm_does_not_expand_symbols_into_words() -> None:
    # 硬约束：只删不替换成词。`/` 与 `%` 都必须消失，不能变成「分之」「百分之」
    assert norm_for_match("士兵12/25", "zh") == "士兵1225"
    assert norm_for_match("进度50%", "zh") == "进度50"


def test_norm_and_index_map_have_equal_length() -> None:
    # 契约：idx_map 必须能逐字回映射原文。`İ`.lower() 是 2 个码位，
    # 若按输入码位补 map，norm 会比 map 长，其后下标全部错位。
    for text, lang in (
        ("İstanbul 十二个士兵", "tr"),
        ("十二个士兵", "zh"),
        ("他说十句话，ＡＢＣ。", "zh"),
        ("ＡＢ", "zh"),
    ):
        norm, idx_map = norm_for_match_with_map(text, lang)
        assert len(norm) == len(idx_map), (text, norm, idx_map)


def test_span_inside_numeral_expansion_covers_whole_run() -> None:
    # 「十二」在 norm 里是两个锚定到段首的字，区间止于段内时必须给到段尾，
    # 否则 FA 参考文本只拿到「十」，丢掉「二」。
    text = "十二个士兵"
    _, idx_map = norm_for_match_with_map(text, "zh")
    assert norm_span_to_original_span(idx_map, 0, 2, len(text)) == (0, 2)


def test_norm_still_handles_chinese_numerals() -> None:
    # 既有行为不得回归：中文数字仍改写为阿拉伯数字
    assert norm_for_match("十二个士兵", "zh") == "12个士兵"
