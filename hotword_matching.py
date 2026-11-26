"""Pure helpers for ordered and tail-aware hotword matching."""

from difflib import SequenceMatcher
from typing import Any, Callable


Normalize = Callable[[str, str], str]


def build_asr_stream(
    words: list[dict[str, Any]], language: str, normalize: Normalize
) -> tuple[str, list[int]]:
    stream_parts: list[str] = []
    char_to_word: list[int] = []
    for word_index, word in enumerate(words):
        token = normalize(
            f"{word.get('text', '')}{word.get('punctuation', '')}", language
        )
        stream_parts.append(token)
        char_to_word.extend([word_index] * len(token))
    return "".join(stream_parts), char_to_word


def assess_reference_coverage(
    reference_text: str,
    words: list[dict[str, Any]],
    language: str,
    normalize: Normalize,
    *,
    max_gap_ratio: float = 0.01,
    minimum_gap: int = 32,
) -> dict[str, Any]:
    """Detect large reference-text regions absent from the ASR word stream."""
    reference_stream = normalize(reference_text, language)
    asr_stream, _ = build_asr_stream(words, language, normalize)
    allowed_gap = max(minimum_gap, int(len(reference_stream) * max_gap_ratio))
    if not reference_stream:
        return {
            "ok": False,
            "reasons": ["empty_reference"],
            "referenceLength": 0,
            "asrLength": len(asr_stream),
            "matchedLength": 0,
            "coverageRatio": 0.0,
            "leadingGap": 0,
            "internalGap": 0,
            "trailingGap": 0,
            "allowedGap": allowed_gap,
        }
    if not asr_stream:
        return {
            "ok": False,
            "reasons": ["empty_asr"],
            "referenceLength": len(reference_stream),
            "asrLength": 0,
            "matchedLength": 0,
            "coverageRatio": 0.0,
            "leadingGap": len(reference_stream),
            "internalGap": 0,
            "trailingGap": len(reference_stream),
            "allowedGap": allowed_gap,
        }

    blocks = [
        block
        for block in SequenceMatcher(
            None, reference_stream, asr_stream, autojunk=False
        ).get_matching_blocks()
        if block.size > 0
    ]
    matched_length = sum(block.size for block in blocks)
    gaps: list[tuple[int, int, int]] = (
        [(0, len(reference_stream), len(reference_stream))] if not blocks else []
    )
    reference_cursor = 0
    for block in blocks:
        if block.a > reference_cursor:
            gaps.append((reference_cursor, block.a, block.a - reference_cursor))
        reference_cursor = block.a + block.size
    if reference_cursor < len(reference_stream):
        gaps.append((reference_cursor, len(reference_stream), len(reference_stream) - reference_cursor))

    leading_gap = gaps[0][2] if gaps and gaps[0][0] == 0 else 0
    trailing_gap = gaps[-1][2] if gaps and gaps[-1][1] == len(reference_stream) else 0
    internal_gap = max(
        (gap[2] for gap in gaps if gap[0] > 0 and gap[1] < len(reference_stream)),
        default=0,
    )
    reasons: list[str] = []
    if leading_gap > allowed_gap:
        reasons.append("leading_reference_gap")
    if internal_gap > allowed_gap:
        reasons.append("internal_reference_gap")
    if trailing_gap > allowed_gap:
        reasons.append("trailing_reference_gap")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "referenceLength": len(reference_stream),
        "asrLength": len(asr_stream),
        "matchedLength": matched_length,
        "coverageRatio": matched_length / len(reference_stream),
        "leadingGap": leading_gap,
        "internalGap": internal_gap,
        "trailingGap": trailing_gap,
        "allowedGap": allowed_gap,
    }


def _exact_positions(
    needle: str, haystack: str, start: int, end: int
) -> list[int]:
    positions: list[int] = []
    cursor = start
    while cursor <= end - len(needle):
        position = haystack.find(needle, cursor, end)
        if position < 0:
            break
        positions.append(position)
        cursor = position + 1
    return positions


def find_best_span(
    needle: str,
    haystack: str,
    start: int = 0,
    end: int | None = None,
    *,
    min_ratio: float = 0.8,
    min_length: int = 5,
    prefer_last: bool = False,
) -> dict[str, Any] | None:
    """Find an exact or small-edit-distance span in a normalized stream."""
    if not needle or not haystack:
        return None

    search_end = len(haystack) if end is None else min(end, len(haystack))
    search_start = max(0, start)
    exact = _exact_positions(needle, haystack, search_start, search_end)
    if exact:
        if len(exact) == 1 or prefer_last:
            position = exact[-1] if prefer_last else exact[0]
            return {
                "start": position,
                "end": position + len(needle),
                "score": 1.0,
                "matchType": "exact",
            }
        return None

    if len(needle) < min_length:
        return None

    min_window = max(1, int(len(needle) * 0.7))
    max_window = min(
        search_end - search_start,
        max(min_window, int(len(needle) * 1.4) + 2),
    )
    if min_window > max_window:
        return None

    best: tuple[float, int, int] | None = None
    for window_start in range(search_start, search_end - min_window + 1):
        for window_length in range(min_window, max_window + 1):
            window_end = window_start + window_length
            if window_end > search_end:
                break
            score = SequenceMatcher(
                None, needle, haystack[window_start:window_end]
            ).ratio()
            candidate = (score, window_start, window_end)
            if best is None or score > best[0] or (
                prefer_last and score == best[0] and window_start > best[1]
            ):
                best = candidate

    if best is None or best[0] < min_ratio:
        return None
    return {
        "start": best[1],
        "end": best[2],
        "score": best[0],
        "matchType": "fuzzy",
    }


def find_tail_matches(
    hotwords: list[dict[str, Any]],
    words: list[dict[str, Any]],
    language: str,
    normalize: Normalize,
    existing_matches: list[dict[str, Any]],
    *,
    min_ratio: float = 0.8,
    min_length: int = 5,
) -> list[dict[str, Any]]:
    """Match unmatched reference hotwords backwards from the ASR tail.

    The caller supplies already matched anchors. The search is restricted to
    the ASR region after the latest reference anchor, so a fuzzy tail match
    cannot jump back into an earlier chapter section.
    """
    stream, char_to_word = build_asr_stream(words, language, normalize)
    if not stream:
        return []

    matched_norms = {item.get("normText") for item in existing_matches}
    latest = max(
        existing_matches,
        key=lambda item: item.get("offsetInFullNorm", -1),
        default=None,
    )
    tail_start = 0
    latest_ref_offset = -1
    if latest is not None:
        latest_ref_offset = latest.get("offsetInFullNorm", -1)
        latest_asr_start = latest.get("asrOffset", 0)
        latest_asr_end = latest.get("asrEndOffset")
        if latest_asr_end is None:
            latest_asr_end = latest_asr_start + len(latest.get("normText", ""))
        tail_start = max(0, int(latest_asr_end))

    unmatched_tail = [
        item
        for item in hotwords
        if item.get("normText") not in matched_norms
        and item.get("offsetInFullNorm", -1) > latest_ref_offset
    ]
    unmatched_tail.sort(key=lambda item: item["offsetInFullNorm"], reverse=True)

    cursor = len(stream)
    results: list[dict[str, Any]] = []
    for hotword in unmatched_tail:
        span = find_best_span(
            hotword["normText"],
            stream,
            tail_start,
            cursor,
            min_ratio=min_ratio,
            min_length=min_length,
            prefer_last=True,
        )
        if span is None:
            continue

        start = int(span["start"])
        end = int(span["end"])
        if start >= len(char_to_word) or end <= start:
            continue
        start_word = char_to_word[start]
        end_word = char_to_word[min(end - 1, len(char_to_word) - 1)]
        results.append(
            {
                **hotword,
                "asrOffset": start,
                "asrEndOffset": end,
                "startWordIndex": start_word,
                "endWordIndex": end_word,
                "matchScore": span["score"],
                "matchType": span["matchType"],
            }
        )
        cursor = start

    return list(reversed(results))


def select_monotonic_chain(
    matched: list[dict[str, Any]], time_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the highest-scoring text/time ordered anchor chain."""
    if not matched:
        return [], []

    ordered = sorted(
        enumerate(matched), key=lambda item: item[1].get("offsetInFullNorm", -1)
    )
    path_scores: list[float] = []
    predecessors: list[int | None] = []
    for current_index, (_, current) in enumerate(ordered):
        current_score = float(current.get("matchScore", 0.5)) + 0.5
        best_score = current_score
        best_predecessor: int | None = None
        for previous_index in range(current_index):
            _, previous = ordered[previous_index]
            if previous.get(time_key, 0) > current.get(time_key, 0):
                continue
            if previous.get("asrOffset", 0) >= current.get("asrOffset", 0):
                continue
            candidate_score = path_scores[previous_index] + current_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_predecessor = previous_index
        path_scores.append(best_score)
        predecessors.append(best_predecessor)

    end_index = max(range(len(ordered)), key=lambda index: path_scores[index])
    selected_indices: set[int] = set()
    while end_index is not None:
        selected_indices.add(ordered[end_index][0])
        end_index = predecessors[end_index]

    valid = [item for index, item in enumerate(matched) if index in selected_indices]
    discarded = [item for index, item in enumerate(matched) if index not in selected_indices]
    valid.sort(key=lambda item: item.get(time_key, 0))
    for index, item in enumerate(valid):
        item["anchorConfidence"] = (
            "high" if 0 < index < len(valid) - 1 else "medium"
        )
    return valid, discarded
