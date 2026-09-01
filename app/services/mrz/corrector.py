"""Constrained OCR error correction for MRZ fields.

Two mechanisms, deliberately kept separate:

1. **Type coercion** - where ICAO 9303 mandates a digit we map letter
   look-alikes to digits, and vice versa. This is safe because the standard,
   not a guess, defines the field type.

2. **Check-digit guided repair** - for alphanumeric fields (document number,
   optional data) no direction is known, so candidate substitutions are
   enumerated and accepted *only* when exactly one candidate satisfies the
   field's check digit. If several candidates validate, or none does, the
   original text is kept and the ambiguity is reported. Blind correction is
   never applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product
from typing import Callable, List, Optional, Tuple

from app.services.mrz.charset import (
    AMBIGUOUS,
    DIGIT_TO_LETTER,
    FILLER,
    coerce_alpha,
    coerce_numeric,
)
from app.services.mrz.checkdigit import verify_check_digit

# Upper bound on simultaneous substitutions. Two covers the overwhelming
# majority of real OCR slips while keeping the search space tiny and the
# false-positive risk negligible.
MAX_SUBSTITUTIONS = 2
MAX_CANDIDATES = 4096

VALID_SEX = {"M", "F", "X"}


def coerce_field(value: str, kind: str) -> str:
    """Apply the type coercion mandated by the field's definition."""
    if kind in ("num", "date"):
        return coerce_numeric(value)
    if kind in ("alpha", "name", "doc_code"):
        return coerce_alpha(value)
    if kind == "sex":
        return coerce_sex(value)
    return value


def coerce_sex(value: str) -> str:
    """Normalise the sex field to M / F / X / filler.

    ``<`` is the standard's 'unspecified' value, so an unreadable glyph becomes
    a filler rather than an invented letter.
    """
    char = (value or FILLER).strip().upper()[:1] or FILLER
    if char in VALID_SEX or char == FILLER:
        return char
    mapped = DIGIT_TO_LETTER.get(char, char)
    if mapped in VALID_SEX:
        return mapped
    # 'H' (hombre) and 'N' are seen in the wild on badly printed documents.
    return {"H": "M", "N": FILLER, "0": FILLER}.get(char, FILLER)


def _substitution_candidates(field: str, max_subs: int) -> List[str]:
    """Every variant of ``field`` reachable with <= ``max_subs`` swaps."""
    positions = [i for i, ch in enumerate(field) if ch in AMBIGUOUS]
    if not positions:
        return []
    out: List[str] = []
    for count in range(1, max_subs + 1):
        for combo in combinations(positions, count):
            alternatives = [[AMBIGUOUS[field[i]]] for i in combo]
            for choice in product(*alternatives):
                chars = list(field)
                for idx, replacement in zip(combo, choice):
                    chars[idx] = replacement
                out.append("".join(chars))
                if len(out) >= MAX_CANDIDATES:
                    return out
    return out


def repair_with_check_digit(
    field: str,
    digit: Optional[str],
    *,
    max_subs: int = MAX_SUBSTITUTIONS,
    validator: Optional[Callable[[str], bool]] = None,
) -> Tuple[str, bool, bool]:
    """Try to make ``field`` agree with ``digit``.

    Returns ``(field, corrected, ambiguous)``. The field is only changed when a
    single candidate both validates against the check digit and passes the
    optional semantic ``validator``.
    """
    if not field or digit is None or digit == "" or not str(digit).isdigit():
        return field, False, False
    if verify_check_digit(field, digit) and (validator is None or validator(field)):
        return field, False, False

    # Staged search: a single-character slip is by far the most likely, so a
    # unique 1-substitution answer wins outright. Only if nothing validates at
    # that distance do we widen the search, which keeps the false-positive rate
    # of the wider search from swamping the obvious fix.
    ambiguous = False
    for distance in range(1, max(1, max_subs) + 1):
        matches = {
            candidate
            for candidate in _substitution_candidates(field, distance)
            if verify_check_digit(candidate, digit)
            and (validator is None or validator(candidate))
        }
        if len(matches) == 1:
            return matches.pop(), True, False
        if matches:
            ambiguous = True
            break
    return field, False, ambiguous


def repair_date(raw: str, digit: Optional[str]) -> Tuple[str, bool, bool]:
    """Repair a ``YYMMDD`` field, constrained to calendar-plausible results."""
    coerced = coerce_numeric(raw or "")
    return repair_with_check_digit(coerced, digit, validator=is_plausible_mrz_date)


def is_plausible_mrz_date(value: str) -> bool:
    if len(value) != 6 or not value.isdigit():
        return False
    month, day = int(value[2:4]), int(value[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def is_plausible_country_code(value: str) -> bool:
    return len(value) == 3 and all(c.isalpha() or c == FILLER for c in value)


def correct_line(line: str, spec: Sequence[Tuple[int, int, str]]) -> Tuple[str, List[str]]:
    """Apply per-field type coercion across one MRZ line.

    ``spec`` is a sequence of ``(start, end, kind)`` slices covering the line.
    """
    chars = list(line)
    corrections: List[str] = []
    for start, end, kind in spec:
        original = line[start:end]
        if not original:
            continue
        fixed = coerce_field(original, kind)
        if fixed != original:
            corrections.append(kind)
            chars[start:end] = list(fixed)
    return "".join(chars), corrections
