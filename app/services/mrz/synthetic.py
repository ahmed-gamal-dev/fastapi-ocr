"""Synthetic zone generation, for tests and documentation.

This is the inverse of the parser: plain field values in, a syntactically valid
machine-readable zone out, with every check digit computed correctly. It exists
so the test-suite can cover the parser exhaustively without a single real
document, and so damaged input can be simulated deterministically.

Nothing here reads or writes real data.
"""

from __future__ import annotations

from typing import Dict, List

from app.services.mrz.charset import FILLER, name_to_mrz, pad_or_trim
from app.services.mrz.checkdigit import compute_check_digit
from app.services.mrz.models import FORMAT_SHAPES, MRV_A, MRV_B, TD1, TD2, TD3

#: ``(optional_start, optional_end, has_optional_check, has_composite)``
OPTIONAL_LAYOUT = {
    TD3: (28, 42, True, True),
    TD2: (28, 35, False, True),
    MRV_A: (28, 44, False, False),
    MRV_B: (28, 36, False, False),
}


def name_field(surname: str = "", given_names: str = "", width: int = 39) -> str:
    """Encode a name the way a zone does: ``SURNAME<<GIVEN<NAMES``."""
    encoded = name_to_mrz(surname) + "<<" + name_to_mrz(given_names)
    return pad_or_trim(encoded, width)


def build_mrz(fields: Dict[str, str], mrz_type: str = TD3) -> List[str]:
    """Build a complete, check-digit-correct zone from field values.

    Recognised keys: ``document_code``, ``issuing_state``, ``surname``,
    ``given_names``, ``document_number``, ``nationality``, ``birth_date``
    (YYMMDD), ``sex``, ``expiry_date`` (YYMMDD), ``optional_data``.
    """
    if mrz_type not in FORMAT_SHAPES:
        raise ValueError(f"Unsupported format: {mrz_type}")
    if mrz_type == TD1:
        return _build_td1(fields)
    return _build_two_line(fields, mrz_type)


def _build_two_line(fields: Dict[str, str], mrz_type: str) -> List[str]:
    width = FORMAT_SHAPES[mrz_type][1]
    opt_start, opt_end, has_opt_check, has_composite = OPTIONAL_LAYOUT[mrz_type]

    line1 = pad_or_trim(
        pad_or_trim(fields.get("document_code", "P"), 2)
        + pad_or_trim(fields.get("issuing_state", "UTO"), 3)
        + name_field(fields.get("surname", ""), fields.get("given_names", ""), width - 5),
        width,
    )

    number = pad_or_trim(fields.get("document_number", ""), 9)
    birth = pad_or_trim(fields.get("birth_date", ""), 6)
    expiry = pad_or_trim(fields.get("expiry_date", ""), 6)
    optional = pad_or_trim(fields.get("optional_data", ""), opt_end - opt_start)

    line2 = (
        number
        + compute_check_digit(number)
        + pad_or_trim(fields.get("nationality", "UTO"), 3)
        + birth
        + compute_check_digit(birth)
        + (fields.get("sex") or FILLER)[:1]
        + expiry
        + compute_check_digit(expiry)
        + optional
    )
    if has_opt_check:
        line2 += compute_check_digit(optional)
    if has_composite:
        line2 = pad_or_trim(line2, width - 1)
        source = pad_or_trim(line2, width)
        tail_end = 43 if mrz_type == TD3 else 35
        composite = source[0:10] + source[13:20] + source[21:tail_end]
        line2 += compute_check_digit(composite)
    return [line1, pad_or_trim(line2, width)]


def _build_td1(fields: Dict[str, str]) -> List[str]:
    number = pad_or_trim(fields.get("document_number", ""), 9)
    optional_1 = pad_or_trim(fields.get("optional_data", ""), 15)
    line1 = pad_or_trim(
        pad_or_trim(fields.get("document_code", "I"), 2)
        + pad_or_trim(fields.get("issuing_state", "UTO"), 3)
        + number
        + compute_check_digit(number)
        + optional_1,
        30,
    )

    birth = pad_or_trim(fields.get("birth_date", ""), 6)
    expiry = pad_or_trim(fields.get("expiry_date", ""), 6)
    optional_2 = pad_or_trim(fields.get("optional_data_2", ""), 11)
    line2_head = (
        birth
        + compute_check_digit(birth)
        + (fields.get("sex") or FILLER)[:1]
        + expiry
        + compute_check_digit(expiry)
        + pad_or_trim(fields.get("nationality", "UTO"), 3)
        + optional_2
    )
    composite = line1[5:30] + line2_head[0:7] + line2_head[8:15] + line2_head[18:29]
    line2 = pad_or_trim(line2_head, 29) + compute_check_digit(composite)

    line3 = name_field(fields.get("surname", ""), fields.get("given_names", ""), 30)
    return [line1, pad_or_trim(line2, 30), line3]


def damage(line: str, position: int, replacement: str) -> str:
    """Replace one character, to simulate a recognition slip."""
    if not 0 <= position < len(line):
        raise IndexError(f"position {position} is outside the line")
    return line[:position] + replacement + line[position + 1 :]
