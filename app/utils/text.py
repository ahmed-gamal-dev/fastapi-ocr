"""Text normalisation helpers for Arabic and Latin passport fields."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import List, Optional

# ---------------------------------------------------------------- digits
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(ARABIC_INDIC_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(EXTENDED_ARABIC_INDIC_DIGITS)})

# ---------------------------------------------------------------- arabic
# Harakat / tatweel / tashkeel: never printed on a passport data page, so any
# occurrence is an OCR artefact and is safe to drop.
_ARABIC_DIACRITICS = re.compile(
    "["
    "ؐ-ؚ"
    "ً-ٟ"
    "ٰ"
    "ۖ-ۜ"
    "۟-ۨ"
    "۪-ۭ"
    "ـ"  # tatweel
    "]"
)

_ARABIC_LETTERS = re.compile(r"[ء-غف-يٮ-ۓ]")
_ARABIC_KEEP = re.compile(r"[^ء-يٮ-ۓٹ-ۓ\s]")

_LATIN_KEEP = re.compile(r"[^A-Za-z' \-]")
_WS = re.compile(r"\s+")


def normalise_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII digits."""
    return text.translate(_DIGIT_MAP)


def strip_control(text: str) -> str:
    return "".join(
        ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C"
    )


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTERS.search(text))


def arabic_ratio(text: str) -> float:
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return len(_ARABIC_LETTERS.findall("".join(stripped))) / len(stripped)


def latin_ratio(text: str) -> float:
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return sum(1 for c in stripped if "A" <= c.upper() <= "Z") / len(stripped)


def normalise_arabic_name(text: Optional[str]) -> Optional[str]:
    """Clean an Arabic name without changing its spelling.

    Removes diacritics, tatweel, punctuation, Latin characters and duplicate
    whitespace. Letter forms (أ إ آ ا / ة ه / ى ي) are deliberately preserved:
    rewriting them would change the name as printed in the passport.
    """
    if not text:
        return None
    out = strip_control(unicodedata.normalize("NFC", text))
    out = _ARABIC_DIACRITICS.sub("", out)
    out = out.replace("‏", " ").replace("‎", " ").replace("؜", " ")
    out = _ARABIC_KEEP.sub(" ", out)
    out = collapse_ws(out)
    if not out or not has_arabic(out):
        return None
    # A single stray letter is noise, not a name.
    if len(out.replace(" ", "")) < 2:
        return None
    return out


def fold_arabic(text: Optional[str]) -> str:
    """Aggressive folding used ONLY for comparing two Arabic strings."""
    if not text:
        return ""
    out = normalise_arabic_name(text) or ""
    for src, dst in (
        ("أ", "ا"),  # أ -> ا
        ("إ", "ا"),  # إ -> ا
        ("آ", "ا"),  # آ -> ا
        ("ة", "ه"),  # ة -> ه
        ("ى", "ي"),  # ى -> ي
        ("ؤ", "و"),  # ؤ -> و
        ("ئ", "ي"),  # ئ -> ي
    ):
        out = out.replace(src, dst)
    return out.replace(" ", "")


def normalise_latin_name(text: Optional[str]) -> Optional[str]:
    """Uppercase, ASCII-fold and clean a Latin name."""
    if not text:
        return None
    out = unicodedata.normalize("NFKD", text)
    out = "".join(c for c in out if not unicodedata.combining(c))
    out = out.upper().replace("<", " ")
    out = _LATIN_KEEP.sub(" ", out)
    out = collapse_ws(out)
    if len(out.replace(" ", "")) < 2:
        return None
    return out


def normalise_passport_number(text: Optional[str]) -> Optional[str]:
    """Uppercase alphanumerics only. Filler and separators are dropped."""
    if not text:
        return None
    out = normalise_digits(str(text)).upper()
    out = re.sub(r"[^A-Z0-9]", "", out)
    return out or None


def similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def token_set(text: Optional[str]) -> set:
    if not text:
        return set()
    return {t for t in collapse_ws(text).split(" ") if t}


def longest_common_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
