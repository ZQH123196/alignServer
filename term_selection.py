"""即时热词选词：重复出现的名词术语。

选法（2026-10-18 定案，依据 docs/research/hotword-selection-terms）：

1. 分词在原文上做：jieba 词性标注只留名词（词性以 `n` 开头），长度 >= 2，出现 >= 2 次。
   不能先删标点再分词 —— 「沃尔沃尔」会被粘成「沃尔沃」+「尔」，次数与位置都失真。
2. 尾部 3 条按末次出现位置降序取，优先保证章尾有锚点。
3. 剩余槽位按出现次数降序、长度降序、首次位置升序补齐，最多 6 条。
4. 不设下限：候选不足就少给，0 条也给。

候选文本与偏移一律落在归一化文本上，与切块、锚点、FA 参考文本同源。
次数按分词整词出现计，不按子串计；锚点匹配另按子串出现，多出的出现由 LCS 剔除。
"""

import bisect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import jieba.posseg as pseg

HOTWORDS_PER_CHAPTER = 6
TAIL_HOTWORDS = 3
MIN_TERM_LENGTH = 2
MIN_TERM_COUNT = 2
MAX_TERM_LENGTH = 15
EN_MIN_TERM_LENGTH = 4

_ZH_TERM_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_EN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:['\u2019][A-Za-z]+)?")

Normalize = Callable[[str, str], str]


@dataclass(frozen=True)
class TermCandidate:
    """一个热词候选。text 与两个偏移都在归一化文本上。"""

    text: str
    count: int
    first_offset: int
    last_offset: int


def _term_tokens(text: str, lang: str) -> Iterable[tuple[str, int]]:
    """产出 (候选词, 原文起始偏移)。

    zh 用 jieba 词性标注取名词；en 按单词切，至少 4 个字母。
    """

    if lang != "zh":
        for match in _EN_TERM_RE.finditer(text):
            word = match.group(0)
            if len(word) >= EN_MIN_TERM_LENGTH:
                yield word, match.start()
        return

    cursor = 0
    for word, flag in pseg.cut(text):
        start = cursor
        cursor += len(word)
        word = word.strip()
        if len(word) < MIN_TERM_LENGTH:
            continue
        if not _ZH_TERM_RE.fullmatch(word):
            continue
        if not flag.startswith("n"):
            continue
        yield word, start


def _collect_candidates(
    reference_text: str,
    lang: str,
    normalize: Normalize,
    norm_text: str,
    norm_map: list[int],
) -> list[TermCandidate]:
    stats: dict[str, dict[str, int]] = {}
    for word, start in _term_tokens(reference_text, lang):
        norm = normalize(word, lang)
        if not norm or len(norm) > MAX_TERM_LENGTH:
            continue
        offset = bisect.bisect_left(norm_map, start)
        if offset >= len(norm_text) or not norm_text.startswith(norm, offset):
            continue
        entry = stats.setdefault(
            norm, {"count": 0, "first_offset": offset, "last_offset": offset}
        )
        entry["count"] += 1
        entry["first_offset"] = min(entry["first_offset"], offset)
        entry["last_offset"] = max(entry["last_offset"], offset)

    return [
        TermCandidate(
            text=norm,
            count=entry["count"],
            first_offset=entry["first_offset"],
            last_offset=entry["last_offset"],
        )
        for norm, entry in stats.items()
        if entry["count"] >= MIN_TERM_COUNT
    ]


def select_hotword_terms(
    reference_text: str,
    lang: str,
    *,
    normalize: Normalize,
    norm_text: str,
    norm_map: list[int],
    top_n: int = HOTWORDS_PER_CHAPTER,
    tail_n: int = TAIL_HOTWORDS,
) -> list[TermCandidate]:
    """取尾部 3 条 + 频次 3 条，合并去重后最多 top_n 条。

    normalize / norm_text / norm_map 由调用方传入，避免本模块反向依赖 cloud_hotword。
    """

    candidates = _collect_candidates(
        reference_text, lang, normalize, norm_text, norm_map
    )

    tail = sorted(
        candidates,
        key=lambda term: (-term.last_offset, -term.count, -len(term.text), term.first_offset),
    )[:tail_n]
    chosen = {term.text for term in tail}
    rest = [term for term in candidates if term.text not in chosen]
    rest.sort(key=lambda term: (-term.count, -len(term.text), term.first_offset))
    return tail + rest[: max(0, top_n - len(tail))]
