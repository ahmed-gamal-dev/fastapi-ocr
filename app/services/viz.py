"""Label-anchored extraction of visual-inspection-zone fields.

The machine-readable zone carries the document number, the latin name and the
dates, all check-digit protected. It does not carry the Arabic name or the
issuing authority: those are printed only in the visual zone, next to a label.

So they are found the way a reader finds them - by the label. Locate the label,
then take the text beside or below it. Two arrangements occur in practice, and
both appear across designs of the same issuing state:

    الاسم   سالم بن محمد          label and value in separate boxes
    الاسم/ سالم بن محمد           label and value merged into one box

Nothing here reads images. It takes recognised blocks with their geometry and
returns values with the confidence of the box they came from. A field whose
label was not found is absent - never guessed, and never filled in from another
field that happened to be nearby.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.services.ocr.base import TextBlock

# ------------------------------------------------------------------ script

_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN = re.compile(r"[A-Za-z]")
_ARABIC_DIACRITICS = re.compile(r"[ً-ْـ]")


def is_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


def is_latin(text: str) -> bool:
    return bool(_LATIN.search(text))


def normalise_arabic(text: str) -> str:
    """Fold the spelling variants that separate what was printed from what was read.

    Alef and yeh carry several forms that OCR moves freely between, and the
    diacritics it invents on a blurred glyph are never part of the word. This is
    for *matching* only; values are returned as they were read.
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    for variant in "أإآٱ":
        text = text.replace(variant, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip()


def split_scripts(text: str) -> Tuple[str, str]:
    """``("DAMMAM", "الدمام")`` - the two halves of a bilingual box.

    A value printed in both scripts often comes back as one run of glyphs with
    no separator at all, so the split is by script rather than by whitespace.
    """
    latin = " ".join(re.findall(r"[A-Za-z][A-Za-z'\-.]*", text)).strip()
    arabic_chars = re.findall(r"[؀-ۿݐ-ݿ\s]+", text)
    arabic = re.sub(r"\s+", " ", "".join(arabic_chars)).strip()
    return latin, arabic


# ------------------------------------------------------------------- model


@dataclass
class VizField:
    """One field read out of the visual zone."""

    value: str
    confidence: float
    #: "merged" when label and value shared a box, "adjacent" when they did not.
    source: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class FieldSpec:
    """How to find one field, and what counts as a plausible value for it."""

    name: str
    #: Matched against the normalised text of every box, longest label first.
    labels: Tuple[str, ...]
    #: "arabic" or "latin" - a value in the wrong script is not this field.
    script: str
    #: Reject anything longer than this; a run-on box is a failed read, not a name.
    max_chars: int = 80


NAME_AR = FieldSpec(
    name="name_ar",
    labels=("الاسم",),
    script="arabic",
)

ISSUING_AUTHORITY_AR = FieldSpec(
    name="issuing_authority_ar",
    labels=("مكان الاصدار", "جهه الاصدار"),
    script="arabic",
    max_chars=40,
)

ISSUING_AUTHORITY = FieldSpec(
    name="issuing_authority",
    # Labelled in either script: the place is often printed in both, in one box,
    # under whichever label the design happens to put beside it.
    labels=(
        "issuingauthority",
        "issuing authority",
        "place of issue",
        "مكان الاصدار",
        "جهه الاصدار",
    ),
    script="latin",
    max_chars=40,
)

FIELD_SPECS: Tuple[FieldSpec, ...] = (
    NAME_AR,
    ISSUING_AUTHORITY_AR,
    ISSUING_AUTHORITY,
)

#: Labels of *other* fields. A value that is really the next label is no value.
_ALL_LABELS: Tuple[str, ...] = tuple(
    label for spec in FIELD_SPECS for label in spec.labels
) + (
    "تاريخ الميلاد",
    "تاريخ الاصدار",
    "تاريخ الانتهاء",
    "الجنسيه",
    "الجنس",
    "رقم الجواز",
    "جواز سفر",
    "date of birth",
    "date of issue",
    "date of expiry",
    "nationality",
    "passport no",
    "country code",
)


# ----------------------------------------------------------------- matching


def _key(text: str) -> str:
    """The form of a box's text that labels are compared against.

    Both scripts are folded the same way whatever field is being looked for: a
    value printed in one script is routinely labelled in the other, so the
    script of the label says nothing about the script of the value.
    """
    return re.sub(r"\s+", " ", normalise_arabic(text)).strip().lower()


def _label_at_start(text: str, spec: FieldSpec) -> Optional[Tuple[str, str]]:
    """``(label, remainder)`` when this box opens with one of the labels.

    The remainder is sliced out of the *original* text, not out of the folded
    form the label was matched against - folding is for recognising the label,
    and a value must be returned as it was read. Slicing it out of the folded
    form is how an upper-case place name silently becomes lower-case.
    """
    key = _key(text)
    for label in sorted(spec.labels, key=len, reverse=True):
        if not key.startswith(label):
            continue
        # Folding can change length, so walk the original until its folded
        # prefix is the label; that boundary is where the value begins.
        limit = min(len(text), len(label) * 2 + 16)
        for end in range(1, limit + 1):
            if _key(text[:end]) == label:
                return label, text[end:]
        return label, ""
    return None


def _looks_like_another_label(text: str) -> bool:
    key = _key(text)
    return any(label in key for label in _ALL_LABELS)


#: Enough letters to be a word rather than a fragment of a misread glyph.
_MIN_LETTERS = 3

#: A date landing next to a label is the commonest wrong answer, because dates
#: are printed all over this part of the page.
_MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def _plausible(value: str, spec: FieldSpec) -> bool:
    """Is this a value for this field, or just whatever sat closest to the label?

    Proximity alone is a weak signal on a dense page, so a candidate has to look
    like the kind of word the field holds: long enough to be a word, free of
    digits, and not the label of the field printed next to it.
    """
    if not value or len(value) > spec.max_chars:
        return False
    if any(character.isdigit() for character in value):
        return False
    if _looks_like_another_label(value):
        return False

    tokens = value.split()
    if not tokens:
        return False

    if spec.script == "arabic":
        if not is_arabic(value):
            return False
        letters = re.sub(r"\s", "", value)
    else:
        if not is_latin(value):
            return False
        if any(token.lower().strip(".") in _MONTHS for token in tokens):
            return False
        letters = re.sub(r"[^A-Za-z]", "", value)

    # One stray letter among the tokens means the box was misread, not that the
    # value has a one-letter word in it - neither script has those here.
    if any(len(token.strip(".,'-")) < 2 for token in tokens):
        return False
    return len(letters) >= _MIN_LETTERS


def _clean_remainder(remainder: str, spec: FieldSpec) -> str:
    """Strip what sits between a label and its value.

    The separator is usually a slash, and a slash on a worn page is routinely
    recognised as a single letter - so a one-character leading token is dropped
    with it. No name or place name in either script is one character long.
    """
    value = remainder.strip(" :/\\-|.,")
    tokens = value.split()
    while tokens and len(tokens[0]) == 1:
        tokens.pop(0)
    value = " ".join(tokens)
    if spec.script == "latin":
        latin, _ = split_scripts(value)
        return latin
    _, arabic = split_scripts(value)
    return arabic


# ---------------------------------------------------------------- geometry


def _vertical_overlap(a: TextBlock, b: TextBlock) -> float:
    top = max(a.y_min, b.y_min)
    bottom = min(a.y_max, b.y_max)
    shorter = min(a.y_max - a.y_min, b.y_max - b.y_min) or 1.0
    return max(0.0, bottom - top) / shorter


def _horizontal_overlap(a: TextBlock, b: TextBlock) -> float:
    left = max(a.x_min, b.x_min)
    right = min(a.x_max, b.x_max)
    narrower = min(a.x_max - a.x_min, b.x_max - b.x_min) or 1.0
    return max(0.0, right - left) / narrower


def _candidates(label: TextBlock, blocks: Sequence[TextBlock]) -> List[TextBlock]:
    """Boxes that could hold this label's value, nearest first.

    Two positions are accepted, because both are used: beside the label on the
    same line, and on the line under it. Arabic reads right to left, so a value
    beside its label is to the *left* of it.
    """
    height = max(label.y_max - label.y_min, 1.0)
    beside: List[Tuple[float, TextBlock]] = []
    below: List[Tuple[float, TextBlock]] = []

    for block in blocks:
        if block is label:
            continue
        if _vertical_overlap(label, block) > 0.5:
            if block.x_max <= label.x_max:
                beside.append((label.x_min - block.x_max, block))
            continue
        gap = block.y_min - label.y_max
        if 0 <= gap <= 2.0 * height and _horizontal_overlap(label, block) > 0.25:
            below.append((gap, block))

    beside.sort(key=lambda pair: abs(pair[0]))
    below.sort(key=lambda pair: pair[0])
    return [block for _, block in beside] + [block for _, block in below]


# ------------------------------------------------------------------ public


def extract_field(
    blocks: Sequence[TextBlock], spec: FieldSpec
) -> Optional[VizField]:
    """Find one labelled field, or return ``None`` when its label is not there."""
    for block in blocks:
        found = _label_at_start(block.text, spec)
        if found is None:
            continue
        _, remainder = found

        # The label and its value share a box more often than not.
        merged = _clean_remainder(remainder, spec)
        if _plausible(merged, spec):
            return VizField(merged, block.confidence, "merged")

        # Otherwise it is the nearest box beside or below the label.
        for candidate in _candidates(block, blocks):
            value = _clean_remainder(candidate.text, spec)
            if _plausible(value, spec):
                return VizField(value, candidate.confidence, "adjacent")
    return None


def extract(blocks: Sequence[TextBlock]) -> Dict[str, VizField]:
    """Every visual-zone field that could be found, keyed by field name."""
    found: Dict[str, VizField] = {}
    for spec in FIELD_SPECS:
        field = extract_field(blocks, spec)
        if field is not None:
            found[spec.name] = field
    return found


def missing_labels(blocks: Sequence[TextBlock]) -> List[FieldSpec]:
    """Specs whose label is on the page but whose value was not readable.

    These are the fields worth a second, enhanced look: the label proves the
    region exists, so a failed read is a recognition problem rather than an
    absent field.
    """
    pending: List[FieldSpec] = []
    for spec in FIELD_SPECS:
        if extract_field(blocks, spec) is not None:
            continue
        if any(_label_at_start(block.text, spec) for block in blocks):
            pending.append(spec)
    return pending


def label_block(
    blocks: Sequence[TextBlock], spec: FieldSpec
) -> Optional[TextBlock]:
    """The box holding this field's label, if it is on the page."""
    for block in blocks:
        if _label_at_start(block.text, spec) is not None:
            return block
    return None
