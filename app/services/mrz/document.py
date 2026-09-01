"""The structured result returned by an :class:`~app.services.mrz.base.MRZParser`.

An :class:`MRZDocument` is a bag of :class:`~app.services.mrz.fields.ParsedField`
objects plus zone-level validity. Callers read fields by name; nothing is
positional and nothing is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.mrz.fields import (
    FIELD_NAMES,
    GIVEN_NAMES,
    SURNAME,
    ParsedField,
)


@dataclass
class MRZDocument:
    """Everything a parser could establish about one machine-readable zone."""

    mrz_type: str
    fields: Dict[str, ParsedField] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)
    #: Structural conformance: right shape, right field types, plausible dates.
    structure_valid: bool = False
    #: Every applicable check digit agrees with the field it protects.
    check_digits_valid: bool = False
    #: Per check digit, ``True``/``False``/``None`` (not applicable).
    check_digits: Dict[str, Optional[bool]] = field(default_factory=dict)
    confidence: float = 0.0
    #: Conditions that make the zone untrustworthy.
    errors: List[str] = field(default_factory=list)
    #: Observations that do not invalidate the parse.
    warnings: List[str] = field(default_factory=list)
    #: Conservative OCR repairs that were applied, for audit.
    corrections: List[str] = field(default_factory=list)
    parser: str = ""

    # ------------------------------------------------------------- accessors
    @property
    def valid(self) -> bool:
        return self.structure_valid and self.check_digits_valid

    def get(self, name: str) -> Optional[ParsedField]:
        return self.fields.get(name)

    def value(self, name: str, default: Any = None) -> Any:
        """The parsed value, or ``default`` when the field could not be read."""
        found = self.fields.get(name)
        return found.value if found is not None and found.value is not None else default

    def confidence_of(self, name: str) -> float:
        found = self.fields.get(name)
        return found.confidence if found else 0.0

    def is_valid(self, name: str) -> Optional[bool]:
        found = self.fields.get(name)
        return found.valid if found else None

    @property
    def present_fields(self) -> List[str]:
        return [name for name in FIELD_NAMES if self.value(name) is not None]

    @property
    def verified_fields(self) -> List[str]:
        """Fields a check digit independently confirmed."""
        return [
            name
            for name, item in self.fields.items()
            if item.verified
        ]

    @property
    def full_name(self) -> Optional[str]:
        """Given names followed by surname, or ``None`` if neither was read."""
        parts = [self.value(GIVEN_NAMES), self.value(SURNAME)]
        joined = " ".join(part for part in parts if part)
        return joined or None

    @property
    def raw(self) -> str:
        return "\n".join(self.lines)

    # --------------------------------------------------------- serialisation
    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """Serialise the document.

        ``include_raw`` is off by default: the raw zone text reproduces the
        whole document in one string, so callers have to ask for it rather than
        receive it by accident.
        """
        data: Dict[str, Any] = {
            "mrz_type": self.mrz_type,
            "parser": self.parser,
            "valid": self.valid,
            "structure_valid": self.structure_valid,
            "check_digits_valid": self.check_digits_valid,
            "check_digits": dict(self.check_digits),
            "confidence": round(self.confidence, 4),
            "fields": {name: item.to_dict() for name, item in self.fields.items()},
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "corrections": list(self.corrections),
        }
        if include_raw:
            data["raw"] = self.raw
            data["lines"] = list(self.lines)
        return data

    def to_flat_dict(self) -> Dict[str, Any]:
        """Just the values, for callers that do not need the per-field detail."""
        return {name: self.value(name) for name in FIELD_NAMES}

    def summary(self) -> str:
        """A short, non-disclosing description, safe to log."""
        return (
            f"{self.mrz_type} valid={self.valid} "
            f"confidence={self.confidence:.2f} "
            f"fields={len(self.present_fields)}/{len(FIELD_NAMES)} "
            f"verified={len(self.verified_fields)}"
        )
