"""Locating the machine readable zone inside an OCR result.

The detector is geometry-aware: it regroups recognition boxes into visual
lines, scores each line on how much of it already belongs to the MRZ alphabet,
and then tries the ICAO line-count/width combinations against the lowest
plausible group of lines. The best-scoring parse wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.services.mrz.charset import mrz_ratio, normalize_line
from app.services.mrz.models import MRZResult
from app.services.mrz.parser import parse

# The line-scanning and ranking logic lives in textscan, which knows nothing
# about images or OCR engines. This module only adds the geometry on top.
from app.services.mrz.textscan import (
    MIN_LINE_LENGTH,
    parse_text,
)
from app.services.mrz.textscan import formats_for as _formats_for
from app.services.mrz.textscan import rank_result as _rank
from app.services.ocr.base import TextBlock

__all__ = [
    "MRZCandidate",
    "detect_and_parse",
    "find_candidates",
    "group_into_lines",
    "parse_text",
    "score_line",
]

MIN_MRZ_RATIO = 0.72


@dataclass
class MRZCandidate:
    lines: List[str]
    confidence: float
    y_position: float
    source_blocks: List[TextBlock] = field(default_factory=list)


def group_into_lines(
    blocks: Sequence[TextBlock], tolerance: float = 0.6
) -> List[List[TextBlock]]:
    """Group recognition boxes that sit on the same visual text line."""
    usable = [b for b in blocks if b.text.strip() and b.polygon]
    if not usable:
        return [[b] for b in blocks if b.text.strip()]

    usable = sorted(usable, key=lambda b: (b.center[1], b.x_min))
    lines: List[List[TextBlock]] = []
    for block in usable:
        placed = False
        for line in lines:
            reference = line[-1]
            limit = tolerance * max(reference.height, block.height, 1.0)
            if abs(block.center[1] - reference.center[1]) <= limit:
                line.append(block)
                placed = True
                break
        if not placed:
            lines.append([block])
    for line in lines:
        line.sort(key=lambda b: b.x_min)
    lines.sort(key=lambda line: min(b.center[1] for b in line))
    return lines


def _line_text(line: Sequence[TextBlock]) -> str:
    return "".join(b.text.strip() for b in line)


def _line_confidence(line: Sequence[TextBlock]) -> float:
    scored = [b.confidence for b in line if b.text.strip()]
    return sum(scored) / len(scored) if scored else 0.0


def score_line(text: str) -> Tuple[str, float]:
    """Normalise a line and score how likely it is to be an MRZ line."""
    normalized = normalize_line(text)
    if len(normalized) < MIN_LINE_LENGTH:
        return normalized, 0.0
    ratio = mrz_ratio(text)
    # MRZ lines are dense with fillers; ordinary printed text almost never is.
    filler_share = normalized.count("<") / max(len(normalized), 1)
    length_fit = min(len(normalized) / 44.0, 1.0)
    return normalized, ratio * 0.6 + min(filler_share * 3.0, 1.0) * 0.25 + length_fit * 0.15


def find_candidates(blocks: Sequence[TextBlock]) -> List[MRZCandidate]:
    """Return MRZ line groups, most promising first."""
    grouped = group_into_lines(blocks)
    scored: List[Tuple[int, str, float, float, List[TextBlock]]] = []
    for index, line in enumerate(grouped):
        raw = _line_text(line)
        normalized, score = score_line(raw)
        if score >= MIN_MRZ_RATIO * 0.6 and len(normalized) >= MIN_LINE_LENGTH:
            y = sum(b.center[1] for b in line) / len(line)
            scored.append((index, normalized, score, y, list(line)))

    candidates: List[MRZCandidate] = []
    if not scored:
        return candidates

    # Split a line that swallowed two MRZ rows in one box (2 x 44 or 2 x 36).
    for index, normalized, score, y, source in list(scored):
        for _, width in ((2, 44), (2, 36)):
            if len(normalized) in (2 * width, 2 * width - 1, 2 * width + 1):
                scored.append(
                    (
                        index,
                        normalized[:width],
                        score,
                        y,
                        source,
                    )
                )
                scored.append(
                    (
                        index + 1,
                        normalized[width:],
                        score,
                        y + 1,
                        source,
                    )
                )

    scored.sort(key=lambda item: item[0])
    # Consecutive runs of MRZ-looking lines.
    runs: List[List[Tuple[int, str, float, float, List[TextBlock]]]] = []
    current: List[Tuple[int, str, float, float, List[TextBlock]]] = []
    for item in scored:
        if current and item[0] - current[-1][0] > 1:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)

    for run in runs:
        for count in (3, 2):
            if len(run) < count:
                continue
            # Prefer the last lines of the run: the MRZ sits at the bottom.
            window = run[-count:]
            candidates.append(
                MRZCandidate(
                    lines=[item[1] for item in window],
                    confidence=sum(item[2] for item in window) / count,
                    y_position=max(item[3] for item in window),
                    source_blocks=[b for item in window for b in item[4]],
                )
            )
    # Lower on the page and higher scoring first.
    candidates.sort(key=lambda c: (-c.confidence, -c.y_position))
    return candidates


def detect_and_parse(
    blocks: Sequence[TextBlock], ocr_confidence: Optional[float] = None
) -> Optional[MRZResult]:
    """Find the MRZ in an OCR result and parse it.

    Every plausible line grouping is tried against every format; the parse with
    the most satisfied check digits wins. Returns ``None`` when no grouping
    yields a structurally valid MRZ.
    """
    best: Optional[MRZResult] = None
    best_score = -1.0
    for candidate in find_candidates(blocks):
        confidence = (
            ocr_confidence
            if ocr_confidence is not None
            else _line_confidence(candidate.source_blocks)
        )
        for mrz_type in _formats_for(candidate.lines):
            result = parse(candidate.lines, mrz_type, ocr_confidence=confidence)
            if result is None:
                continue
            score = _rank(result)
            if score > best_score:
                best, best_score = result, score
    return best
