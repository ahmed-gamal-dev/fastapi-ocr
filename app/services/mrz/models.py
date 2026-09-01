"""Result types produced by the MRZ parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ICAO 9303 machine readable travel document formats.
TD1 = "TD1"  # 3 x 30, identity cards
TD2 = "TD2"  # 2 x 36, older travel documents
TD3 = "TD3"  # 2 x 44, passports
MRV_A = "MRV_A"  # 2 x 44, visas
MRV_B = "MRV_B"  # 2 x 36, visas

FORMAT_SHAPES = {
    TD1: (3, 30),
    TD2: (2, 36),
    TD3: (2, 44),
    MRV_A: (2, 44),
    MRV_B: (2, 36),
}


@dataclass
class CheckResult:
    """One check digit and whether the field it protects agrees with it."""

    name: str
    field_value: str
    digit: Optional[str]
    expected: Optional[str]
    valid: Optional[bool]  # None = not applicable / not checkable

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.name, "valid": self.valid}


@dataclass
class MRZResult:
    mrz_type: str
    lines: List[str] = field(default_factory=list)
    document_code: str = ""
    document_category: str = "unknown"  # passport | visa | identity_card | other
    issuing_state: str = ""
    document_number: str = ""
    nationality: str = ""
    birth_date: str = ""  # raw YYMMDD as printed in the MRZ
    sex: Optional[str] = None
    expiry_date: str = ""  # raw YYMMDD
    optional_data: str = ""
    personal_number: str = ""
    surname: str = ""
    given_names: str = ""
    checks: List[CheckResult] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structure_valid: bool = False
    ocr_confidence: float = 0.0

    # ------------------------------------------------------------- derived
    @property
    def raw(self) -> str:
        return "\n".join(self.lines)

    @property
    def checkable(self) -> List[CheckResult]:
        return [c for c in self.checks if c.valid is not None]

    @property
    def check_digits_valid(self) -> bool:
        """True when every applicable check digit agrees with its field."""
        applicable = self.checkable
        return bool(applicable) and all(c.valid for c in applicable)

    @property
    def check_digit_score(self) -> float:
        applicable = self.checkable
        if not applicable:
            return 0.0
        return sum(1 for c in applicable if c.valid) / len(applicable)

    @property
    def valid(self) -> bool:
        return self.structure_valid and self.check_digits_valid

    def check(self, name: str) -> Optional[CheckResult]:
        for item in self.checks:
            if item.name == name:
                return item
        return None

    def field_valid(self, name: str) -> Optional[bool]:
        found = self.check(name)
        return found.valid if found else None

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.given_names, self.surname) if p]
        return " ".join(parts)

    @property
    def confidence(self) -> float:
        """Structural confidence, before it is blended with OCR confidence."""
        if not self.structure_valid:
            return 0.0
        score = 0.45 + 0.55 * self.check_digit_score
        if self.corrections:
            score -= 0.05 * min(len(self.corrections), 3)
        return max(0.0, min(1.0, score))

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.mrz_type,
            "valid": self.valid,
            "structure_valid": self.structure_valid,
            "check_digits_valid": self.check_digits_valid,
            "check_digits": {c.name: c.valid for c in self.checks},
            "corrections_applied": list(self.corrections),
            "confidence": round(self.confidence, 4),
        }
        if self.warnings:
            data["warnings"] = list(self.warnings)
        if include_raw:
            data["raw"] = self.raw
            data["lines"] = list(self.lines)
        return data
