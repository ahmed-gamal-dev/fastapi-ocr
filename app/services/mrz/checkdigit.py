"""ICAO 9303 check digit computation and verification.

Weights cycle 7-3-1 over the field. Digits score their face value, ``A``-``Z``
score 10-35, and the filler ``<`` scores 0. The check digit is the sum mod 10.
"""

from __future__ import annotations

from typing import Optional

from app.services.mrz.charset import FILLER

WEIGHTS = (7, 3, 1)


def char_value(char: str) -> Optional[int]:
    if char == FILLER:
        return 0
    if "0" <= char <= "9":
        return ord(char) - 48
    if "A" <= char <= "Z":
        return ord(char) - 55  # 'A' -> 10
    return None


def compute_check_digit(field: str) -> Optional[str]:
    """Return the check digit for ``field``, or None if it holds a bad glyph."""
    if field is None:
        return None
    total = 0
    for index, char in enumerate(str(field)):
        value = char_value(char)
        if value is None:
            return None
        total += value * WEIGHTS[index % 3]
    return str(total % 10)


def verify_check_digit(field: str, digit: Optional[str]) -> Optional[bool]:
    """Verify a field against its check digit.

    Returns ``None`` when verification is not applicable - an absent digit on an
    empty optional field is not a failure, it is simply not checkable.
    """
    if digit is None or digit == "":
        return None
    if digit == FILLER:
        # A filler check digit is only meaningful over an empty field.
        return None if set(str(field)) <= {FILLER} else False
    if not digit.isdigit():
        return False
    expected = compute_check_digit(field)
    if expected is None:
        return False
    return expected == digit
