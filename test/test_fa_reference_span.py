"""FA 参考文本的 norm→原文区间映射（2026-10-18 第 44 章事故）。

事故：`_norm_text_to_orig_map` 在「数字改写后的文本」上直接记下标，而
`_cn_number_to_digits` 会把「十」改写成「10」等，每个改写点都让后续位置右移。
第 44 章累计漂移 12 字，FA 参考文本越过 chunk 音频窗口 3.1s，被判成伪对齐。
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
    norm_for_match,
    norm_for_match_with_map,
    norm_span_to_original_span,
)


TEXT = "他说十几分钟后，天上便开始下起蒙蒙细雨，靠近骏鹰山脉的地方就是这样。"
TARGET = "靠近骏鹰山脉的地方就是这样"


def _norm_range(text: str, sub: str) -> tuple[int, int]:
    norm = norm_for_match(text, "zh")
    sub_norm = norm_for_match(sub, "zh")
    start = norm.index(sub_norm)
    return start, start + len(sub_norm)


def test_span_after_numeral_expansion_maps_back_exactly() -> None:
    # Arrange: 「十」→「10」发生在目标区间之前，旧映射会整体右移
    norm, idx_map = norm_for_match_with_map(TEXT, "zh")
    start, end = _norm_range(TEXT, TARGET)

    # Act
    orig_start, orig_end = norm_span_to_original_span(idx_map, start, end, len(TEXT))

    # Assert
    assert TEXT[orig_start:orig_end] == TARGET


def test_span_reaching_text_end_stops_after_the_last_meaningful_char() -> None:
    # Arrange: 末尾标点不属于 norm 流，区间止于最后一个有意义字符之后
    norm, idx_map = norm_for_match_with_map(TEXT, "zh")

    # Act
    orig_start, orig_end = norm_span_to_original_span(idx_map, 0, len(norm), len(TEXT))

    # Assert
    assert (orig_start, orig_end) == (0, TEXT.index("。"))
    assert TEXT[orig_start:orig_end] == "他说十几分钟后，天上便开始下起蒙蒙细雨，靠近骏鹰山脉的地方就是这样"
