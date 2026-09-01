"""MRZ character set handling per ICAO Doc 9303 Part 3.

The MRZ alphabet is exactly ``A-Z``, ``0-9`` and the filler ``<``. Anything an
OCR engine produces outside that set is an artefact, and the maps below encode
the substitutions that are actually plausible in the OCR-B font.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

MRZ_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
FILLER = "<"

# Glyphs that are never MRZ characters but are commonly emitted for a filler.
_FILLER_LOOKALIKES = {
    "«": "<<",
    "»": "<<",
    "‹": "<",
    "›": "<",
    "≪": "<<",
    "＜": "<",
    " ": "<",
    "\t": "<",
    "_": "<",
    "-": "<",
    "—": "<",
    "–": "<",
    "*": "<",
    "+": "<",
    ".": "<",
    ",": "<",
    "'": "<",
    '"': "<",
    "|": "<",
    "/": "<",
    "\\": "<",
    "(": "<",
    ")": "<",
    "[": "<",
    "]": "<",
    "{": "<",
    "}": "<",
    ":": "<",
    ";": "<",
    "!": "<",
    "?": "<",
    "=": "<",
    "^": "<",
    "~": "<",
    "#": "<",
    "%": "<",
    "&": "<",
    "@": "<",
    "$": "<",
    "°": "<",
}

# Substitutions applied where the standard mandates a digit.
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "J": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

# Substitutions applied where the standard mandates a letter.
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

# Pairs a glyph can legitimately be confused with, used by the constrained
# corrector for alphanumeric fields where neither direction is known upfront.
AMBIGUOUS: Dict[str, str] = {
    "0": "O",
    "O": "0",
    "1": "I",
    "I": "1",
    "2": "Z",
    "Z": "2",
    "4": "A",
    "A": "4",
    "5": "S",
    "S": "5",
    "6": "G",
    "G": "6",
    "7": "T",
    "T": "7",
    "8": "B",
    "B": "8",
    "Q": "0",
    "D": "0",
    "U": "0",
    "L": "1",
    "J": "1",
}

_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_ARABIC_INDIC.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})


def normalize_char(char: str) -> str:
    """Map a single OCR glyph onto the MRZ alphabet, or drop it."""
    if char in MRZ_ALPHABET:
        return char
    replacement = _FILLER_LOOKALIKES.get(char)
    if replacement is not None:
        return replacement
    folded = unicodedata.normalize("NFKD", char)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).upper()
    if len(folded) == 1 and folded in MRZ_ALPHABET:
        return folded
    return ""


def normalize_line(line: str) -> str:
    """Normalise a whole candidate MRZ line to the MRZ alphabet."""
    if not line:
        return ""
    text = unicodedata.normalize("NFKC", str(line)).translate(_ARABIC_INDIC).upper()
    return "".join(normalize_char(ch) for ch in text)


def pad_or_trim(line: str, length: int) -> str:
    """Force a line to its standard length using the filler character."""
    if len(line) < length:
        return line + FILLER * (length - len(line))
    return line[:length]


def strip_fillers(value: Optional[str]) -> str:
    """Turn an MRZ field into a plain value ('' when the field is empty)."""
    if not value:
        return ""
    return value.replace(FILLER, " ").strip()


def is_filler_only(value: Optional[str]) -> bool:
    return not value or set(value) <= {FILLER}


def coerce_numeric(value: str) -> str:
    """Force a field the standard defines as numeric into digits."""
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in value)


def coerce_alpha(value: str) -> str:
    """Force a field the standard defines as alphabetic into letters."""
    return "".join(DIGIT_TO_LETTER.get(ch, ch) for ch in value)


def mrz_ratio(text: str) -> float:
    """Fraction of characters already inside the MRZ alphabet."""
    cleaned = [c for c in str(text).upper() if not c.isspace()]
    if not cleaned:
        return 0.0
    return sum(1 for c in cleaned if c in MRZ_ALPHABET) / len(cleaned)


def candidates_for(char: str) -> List[str]:
    """Every glyph the corrector may try in place of ``char``."""
    out = [char]
    alt = AMBIGUOUS.get(char)
    if alt and alt not in out:
        out.append(alt)
    return out


def name_to_mrz(value: str) -> str:
    """Encode a plain name the way it appears in an MRZ name field."""
    folded = unicodedata.normalize("NFKD", value or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c)).upper()
    folded = re.sub(r"[^A-Z ]", "", folded)
    return re.sub(r"\s+", "<", folded.strip())
