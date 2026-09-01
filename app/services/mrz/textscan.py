"""Finding and parsing a machine-readable zone in plain text.

This module deliberately knows nothing about images, OCR engines or HTTP. It
takes text lines - however they were produced - and works out which of them
form a machine-readable zone.

``detector.py`` builds on this to work from positioned OCR boxes; everything
here operates on strings alone.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from app.services.mrz.charset import FILLER, MRZ_ALPHABET, mrz_ratio, normalize_line
from app.services.mrz.models import FORMAT_SHAPES, MRV_A, MRV_B, TD1, TD2, TD3, MRZResult
from app.services.mrz.parser import parse

#: Shorter than this and a line cannot be a zone line, whatever it contains.
MIN_LINE_LENGTH = 24
#: Formats are tried in order of how common they are.
FORMAT_ORDER = (TD3, TD2, TD1, MRV_A, MRV_B)
#: How far a recognised line may be from its nominal width and still be tried.
WIDTH_TOLERANCE = 6


def strict_mrz_ratio(text: str) -> float:
    """Share of characters already in the zone alphabet, *without* folding.

    Unlike :func:`~app.services.mrz.charset.mrz_ratio`, this does not upper-case
    first. That matters for detection: the zone is upper case by definition, so
    lower-case letters are evidence *against* a line being one. Folding first
    would make any sentence of prose look like a zone.
    """
    cleaned = [c for c in str(text) if not c.isspace()]
    if not cleaned:
        return 0.0
    return sum(1 for c in cleaned if c in MRZ_ALPHABET) / len(cleaned)


def looks_like_mrz_line(text: str, min_ratio: float = 0.85) -> bool:
    """True when a line is dense enough in zone characters to be one.

    Ordinary printed text fails this: it has lower case, spaces and punctuation,
    none of which belong to the zone alphabet. A filler is also required, since
    every real zone line is padded to a fixed width with them.
    """
    normalized = normalize_line(text)
    if len(normalized) < MIN_LINE_LENGTH:
        return False
    if FILLER not in str(text):
        return False
    return strict_mrz_ratio(text) >= min_ratio


def formats_for(lines: Sequence[str]) -> Iterable[str]:
    """Formats whose shape is compatible with this grouping of lines."""
    count = len(lines)
    width = max((len(line) for line in lines), default=0)
    out: List[str] = []
    for mrz_format in FORMAT_ORDER:
        rows, expected = FORMAT_SHAPES[mrz_format]
        if rows != count:
            continue
        # Allow OCR to have lost or gained a few characters at the edges.
        if abs(width - expected) <= WIDTH_TOLERANCE:
            out.append(mrz_format)
    return out


def rank_result(result: MRZResult) -> float:
    """Ranking score used to choose between competing parses of one candidate."""
    if not result.structure_valid:
        return -0.5 + 0.1 * result.check_digit_score
    return 1.0 + result.check_digit_score - 0.02 * len(result.corrections)


def candidate_groups(lines: Sequence[str]) -> List[List[str]]:
    """Groupings of consecutive lines that could form a zone, best first.

    A zone sits at the end of the text it belongs to, so the last lines of each
    run of zone-like lines are tried before earlier ones.
    """
    normalized = [normalize_line(line) for line in lines]
    indexed = [
        (index, text)
        for index, (text, original) in enumerate(zip(normalized, lines))
        if len(text) >= MIN_LINE_LENGTH and mrz_ratio(original) >= 0.6
    ]
    if not indexed:
        return []

    # Split into runs of consecutive lines.
    runs: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    for item in indexed:
        if current and item[0] - current[-1][0] > 1:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)

    groups: List[List[str]] = []
    for run in runs:
        for count in (3, 2):
            if len(run) >= count:
                groups.append([text for _, text in run[-count:]])
    return groups


def split_joined_lines(text: str) -> List[List[str]]:
    """Recover two zone lines that OCR merged into one string.

    A 2 x 44 zone read as a single 88 character run is common when the two rows
    end up in one recognition box.
    """
    normalized = normalize_line(text)
    out: List[List[str]] = []
    for width in (44, 36):
        for total in (2 * width - 1, 2 * width, 2 * width + 1):
            if len(normalized) == total:
                out.append([normalized[:width], normalized[width:]])
    return out


def parse_text(
    text: str, ocr_confidence: float = 0.0, mrz_type: Optional[str] = None
) -> Optional[MRZResult]:
    """Find and parse a zone inside a block of text.

    Every plausible grouping is tried against every compatible format and the
    parse with the most satisfied check digits wins. Returns ``None`` when no
    grouping yields a structurally valid zone - never a partial guess.
    """
    raw_lines = [line for line in str(text).splitlines() if line.strip()]
    if not raw_lines:
        return None

    groups = candidate_groups(raw_lines)
    for line in raw_lines:
        groups.extend(split_joined_lines(line))
    if not groups:
        return None

    best: Optional[MRZResult] = None
    best_score = -1.0
    for group in groups:
        formats = [mrz_type] if mrz_type else list(formats_for(group))
        for candidate_format in formats:
            result = parse(group, candidate_format, ocr_confidence=ocr_confidence)
            if result is None:
                continue
            score = rank_result(result)
            if score > best_score:
                best, best_score = result, score
    return best


def parse_lines(
    lines: Sequence[str], ocr_confidence: float = 0.0, mrz_type: Optional[str] = None
) -> Optional[MRZResult]:
    """Parse lines that are already known to be the zone, in order."""
    return parse_text("\n".join(lines), ocr_confidence=ocr_confidence, mrz_type=mrz_type)
