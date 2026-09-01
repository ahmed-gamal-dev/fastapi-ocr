"""Per-field parse results.

A field carries four independent things: the value, whether the standard's own
check could confirm it, how much confidence that justifies, and what went wrong.
They are deliberately separate - a field can be present but unverifiable, or
verifiable and wrong.

The cardinal rule of this module: an unreadable field is ``None``. It is never
filled with a guess, a partial reading, or a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------- field names
DOCUMENT_CODE = "document_code"
DOCUMENT_CATEGORY = "document_category"
ISSUING_STATE = "issuing_state"
DOCUMENT_NUMBER = "document_number"
NATIONALITY = "nationality"
BIRTH_DATE = "birth_date"
SEX = "sex"
EXPIRY_DATE = "expiry_date"
OPTIONAL_DATA = "optional_data"
PERSONAL_NUMBER = "personal_number"
SURNAME = "surname"
GIVEN_NAMES = "given_names"

#: Every field this parser can produce, in the order a document prints them.
FIELD_NAMES = (
    DOCUMENT_CODE,
    DOCUMENT_CATEGORY,
    ISSUING_STATE,
    DOCUMENT_NUMBER,
    NATIONALITY,
    BIRTH_DATE,
    SEX,
    EXPIRY_DATE,
    OPTIONAL_DATA,
    PERSONAL_NUMBER,
    SURNAME,
    GIVEN_NAMES,
)

#: Fields the standard protects with their own check digit.
CHECK_PROTECTED = (DOCUMENT_NUMBER, BIRTH_DATE, EXPIRY_DATE, OPTIONAL_DATA)


@dataclass
class ParsedField:
    """One field extracted from a machine-readable zone."""

    name: str
    #: The interpreted value, or ``None`` when the field could not be read.
    value: Optional[Any] = None
    #: The characters exactly as they appeared in the zone, fillers included.
    raw: str = ""
    #: ``True``/``False`` from the field's check digit; ``None`` when the
    #: standard defines no check for it, so validity is simply unknown.
    valid: Optional[bool] = None
    confidence: float = 0.0
    errors: List[str] = field(default_factory=list)
    #: True when conservative OCR correction changed this field.
    corrected: bool = False

    @property
    def present(self) -> bool:
        return self.value is not None

    @property
    def verified(self) -> bool:
        """Present *and* confirmed by a check digit."""
        return self.value is not None and self.valid is True

    def add_error(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "valid": self.valid,
            "confidence": round(self.confidence, 4),
            "errors": list(self.errors),
            "corrected": self.corrected,
        }


def missing(name: str, reason: str) -> ParsedField:
    """A field that could not be read. Value stays ``None`` by construction."""
    return ParsedField(name=name, value=None, raw="", valid=None, confidence=0.0,
                       errors=[reason])
