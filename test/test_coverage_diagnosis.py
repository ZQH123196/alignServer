"""覆盖率诊断文案：面向不懂服务内部指标的调用方（2026-10-18 第 8 章事故）。

两条约束：
1. 缺失位置决定判断分支。中间缺段不能拿「尾部未转写时长」推断成因。
2. 不出现 refLen / matched / refNorm / 流尾 等内部指标名。
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
    coverage_gap_verdict,
    format_coverage_diagnosis,
)


class TestCoverageGapVerdict:
    def test_internal_gap_does_not_reason_from_tail_silence(self) -> None:
        verdict = coverage_gap_verdict(
            missing_chars=240,
            physical_duration_ms=473_280,
            asr_end_time_ms=473_070,
            lang="zh",
            gap_kinds=["internal"],
        )
        assert "没有出现在转写里" in verdict
        assert "缺尾段" not in verdict
        assert "先查上游 TTS 产物是否漏读" not in verdict

    def test_trailing_gap_still_checks_the_audio_tail(self) -> None:
        verdict = coverage_gap_verdict(
            missing_chars=240,
            physical_duration_ms=473_280,
            asr_end_time_ms=473_070,
            lang="zh",
            gap_kinds=["trailing"],
        )
        assert "音频结尾" in verdict
        assert "先查上游 TTS 产物是否漏读" in verdict

    def test_mixed_gaps_use_the_internal_branch(self) -> None:
        verdict = coverage_gap_verdict(
            missing_chars=240,
            physical_duration_ms=473_280,
            asr_end_time_ms=473_070,
            lang="zh",
            gap_kinds=["internal", "trailing"],
        )
        assert "先听音频这一段" in verdict
        assert "缺尾段" not in verdict

    def test_no_gap_kinds_falls_back_to_the_internal_branch(self) -> None:
        verdict = coverage_gap_verdict(240, 473_280, 473_070, "zh")
        assert "先听音频这一段" in verdict
        assert "缺尾段" not in verdict


class TestCoverageDiagnosis:
    def _coverage(self) -> dict:
        return {
            "ok": False,
            "reasons": ["internal_reference_gap"],
            "offenders": [
                {
                    "kind": "internal",
                    "start": 164,
                    "end": 404,
                    "length": 240,
                    "text": "沃尔很想立刻开始研究，但也不能将刚招募的士兵撂着不管。",
                }
            ],
            "referenceLength": 1919,
            "asrLength": 1919,
            "matchedLength": 1614,
            "coverageRatio": 0.8411,
            "leadingGap": 0,
            "internalGap": 240,
            "trailingGap": 0,
            "allowedGap": 32,
        }

    def _diagnosis(self) -> str:
        return format_coverage_diagnosis(
            self._coverage(),
            asr_norm_text="发现酒馆已经有了不少人坐着吃饭而门口还站着十个背着包裹的年轻人",
            physical_duration_ms=473_280,
            asr_end_time_ms=473_070,
            lang="zh",
        )

    def test_uses_plain_labels_not_internal_metric_names(self) -> None:
        diagnosis = self._diagnosis()
        for jargon in ("refLen", "asrLen", "matched", "refNorm", "流尾", "internal_reference_gap"):
            assert jargon not in diagnosis
        assert "有效字" in diagnosis
        assert "84.1%" in diagnosis

    def test_reports_the_missing_region_in_one_based_positions(self) -> None:
        diagnosis = self._diagnosis()
        assert "从第 165 个有效字到第 404 个" in diagnosis
        assert "共 240 字" in diagnosis

    def test_labels_the_asr_side_as_a_comparison_sample(self) -> None:
        diagnosis = self._diagnosis()
        assert "ASR 转写结尾 160 个有效字（对照用）" in diagnosis

    def test_states_that_the_check_runs_before_chunking(self) -> None:
        # 2026-10-28 第 34 章：调用方问「不是分块的吗，为什么没兜住」。
        # 覆盖率判定在切块之前整章比对，分块只约束后续强制对齐窗口。
        diagnosis = self._diagnosis()
        assert "切块之前" in diagnosis
        assert "不吸收整章文本缺失" in diagnosis

    def test_empty_asr_is_explained_without_a_fake_missing_count(self) -> None:
        diagnosis = format_coverage_diagnosis(
            {
                "offenders": [],
                "referenceLength": 1919,
                "asrLength": 0,
                "matchedLength": 0,
                "coverageRatio": 0.0,
                "allowedGap": 32,
            },
            "",
            physical_duration_ms=1000,
            asr_end_time_ms=0,
            lang="zh",
        )
        assert "ASR 没有转写出任何文字" in diagnosis
        assert "缺了 0 字" not in diagnosis

    def test_empty_reference_is_explained(self) -> None:
        diagnosis = format_coverage_diagnosis(
            {
                "offenders": [],
                "referenceLength": 0,
                "asrLength": 10,
                "matchedLength": 0,
                "coverageRatio": 0.0,
                "allowedGap": 32,
            },
            "abcdefghij",
            physical_duration_ms=1000,
            asr_end_time_ms=500,
            lang="zh",
        )
        assert "参考文本为空" in diagnosis
