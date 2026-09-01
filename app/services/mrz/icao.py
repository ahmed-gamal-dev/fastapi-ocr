"""ICAO Doc 9303 implementation of :class:`~app.services.mrz.base.MRZParser`.

This is the layer that turns the low-level parse (``parser.py``) into the
structured, per-field result the interface promises: values resolved to their
final types, validity taken from the standard's check digits, a confidence for
each field, and an explicit list of what went wrong.

Design rule inherited from the whole service: **never invent a value.** A field
that cannot be read is ``None`` with a recorded reason. Correction is applied
only where the standard itself makes it safe (a field defined as numeric) or
where a check digit confirms a single unambiguous candidate.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

from app.services.mrz import textscan
from app.services.mrz.base import MRZParser, register_parser
from app.services.mrz.charset import FILLER
from app.services.mrz.confidence import document_confidence, field_confidence
from app.services.mrz.document import MRZDocument
from app.services.mrz.fields import (
    BIRTH_DATE,
    DOCUMENT_CATEGORY,
    DOCUMENT_CODE,
    DOCUMENT_NUMBER,
    EXPIRY_DATE,
    GIVEN_NAMES,
    ISSUING_STATE,
    NATIONALITY,
    OPTIONAL_DATA,
    PERSONAL_NUMBER,
    SEX,
    SURNAME,
    ParsedField,
    missing,
)
from app.services.mrz.models import MRV_A, MRV_B, TD1, TD2, TD3, MRZResult
from app.services.mrz.parser import detect_format as _detect_format
from app.services.mrz.parser import parse as _parse
from app.utils.dates import parse_mrz_date, to_iso, validate_date_set

VALID_SEX = ("M", "F", "X")

#: ``validate_date_set`` reports against document-level date names; map those
#: onto the zone field names used here.
_DATE_FIELD_BY_VALIDATOR = {
    "date_of_birth": BIRTH_DATE,
    "date_of_expiry": EXPIRY_DATE,
}


class ICAOMRZParser(MRZParser):
    """Parses two-line (TD2, TD3, MRV) and three-line (TD1) zones."""

    name = "icao9303"

    def __init__(
        self, reference_date: Optional[date] = None, strict: bool = True
    ) -> None:
        #: Anchors century resolution for two-digit years. Injectable so the
        #: tests are not sensitive to the day they run.
        self.reference_date = reference_date
        #: When strict (the default), lines that fail structural validation
        #: produce ``None`` rather than a document. Without this a page of
        #: ordinary prose of roughly the right length yields a document full of
        #: values scraped out of the words - exactly the invention this parser
        #: must never do. A *real* zone with a bad check digit is structurally
        #: valid and is still returned, flagged invalid.
        #: Set ``strict=False`` only to inspect why something failed to parse.
        self.strict = strict

    # ---------------------------------------------------------- interface
    def supported_formats(self) -> Sequence[str]:
        return (TD1, TD2, TD3, MRV_A, MRV_B)

    def detect_format(self, lines: Sequence[str]) -> Optional[str]:
        from app.services.mrz.charset import normalize_line

        cleaned = [normalize_line(line) for line in lines if normalize_line(line)]
        return _detect_format(cleaned) if cleaned else None

    def parse_lines(
        self,
        lines: Sequence[str],
        mrz_type: Optional[str] = None,
        ocr_confidence: float = 0.0,
    ) -> Optional[MRZDocument]:
        if not lines:
            return None
        result = _parse(list(lines), mrz_type, ocr_confidence=ocr_confidence)
        if result is None:
            # The lines may still contain a zone surrounded by noise.
            result = textscan.parse_lines(
                lines, ocr_confidence=ocr_confidence, mrz_type=mrz_type
            )
        return self._finalise(result, ocr_confidence)

    def parse_text(
        self, text: str, ocr_confidence: float = 0.0
    ) -> Optional[MRZDocument]:
        result = textscan.parse_text(text, ocr_confidence=ocr_confidence)
        return self._finalise(result, ocr_confidence)

    def _finalise(
        self, result: Optional[MRZResult], ocr_confidence: float
    ) -> Optional[MRZDocument]:
        if result is None:
            return None
        if self.strict and not result.structure_valid:
            return None
        return self.build_document(result, ocr_confidence)

    # ------------------------------------------------------- construction
    def build_document(
        self, result: MRZResult, ocr_confidence: float = 0.0
    ) -> MRZDocument:
        """Turn a low-level parse into the structured public result."""
        composite = result.field_valid("composite")
        fields: Dict[str, ParsedField] = {}

        def score(name: str, present: bool, check: Optional[bool]) -> float:
            return field_confidence(
                present=present,
                check_valid=check,
                composite_valid=composite,
                corrected=_was_corrected(result, name),
                ocr_confidence=ocr_confidence,
            )

        # --- document type -------------------------------------------------
        code = result.document_code.strip()
        if code:
            fields[DOCUMENT_CODE] = ParsedField(
                name=DOCUMENT_CODE,
                value=code,
                raw=code,
                valid=None,  # the standard defines no check digit for it
                confidence=score(DOCUMENT_CODE, True, None),
            )
            fields[DOCUMENT_CATEGORY] = ParsedField(
                name=DOCUMENT_CATEGORY,
                value=result.document_category,
                raw=code,
                valid=None,
                confidence=score(DOCUMENT_CATEGORY, True, None),
            )
        else:
            fields[DOCUMENT_CODE] = missing(DOCUMENT_CODE, "document code is empty")
            fields[DOCUMENT_CATEGORY] = missing(
                DOCUMENT_CATEGORY, "document code is empty"
            )

        # --- issuing authority and nationality -----------------------------
        fields[ISSUING_STATE] = _code_field(
            ISSUING_STATE, result.issuing_state, score(ISSUING_STATE, True, None)
        )
        fields[NATIONALITY] = _code_field(
            NATIONALITY, result.nationality, score(NATIONALITY, True, None)
        )

        # --- document number -----------------------------------------------
        number_check = result.field_valid(DOCUMENT_NUMBER)
        number = result.document_number.strip()
        if number:
            item = ParsedField(
                name=DOCUMENT_NUMBER,
                value=number,
                raw=number,
                valid=number_check,
                confidence=score(DOCUMENT_NUMBER, True, number_check),
                corrected=_was_corrected(result, DOCUMENT_NUMBER),
            )
            if number_check is False:
                item.add_error("check digit does not match the document number")
            fields[DOCUMENT_NUMBER] = item
        else:
            fields[DOCUMENT_NUMBER] = missing(
                DOCUMENT_NUMBER, "document number is empty"
            )

        # --- dates ----------------------------------------------------------
        birth = self._date_field(
            BIRTH_DATE, result.birth_date, result.field_valid(BIRTH_DATE), "birth",
            score(BIRTH_DATE, True, result.field_valid(BIRTH_DATE)), result,
        )
        expiry = self._date_field(
            EXPIRY_DATE, result.expiry_date, result.field_valid(EXPIRY_DATE), "expiry",
            score(EXPIRY_DATE, True, result.field_valid(EXPIRY_DATE)), result,
        )
        fields[BIRTH_DATE] = birth
        fields[EXPIRY_DATE] = expiry

        # --- sex -------------------------------------------------------------
        if result.sex in VALID_SEX:
            fields[SEX] = ParsedField(
                name=SEX,
                value=result.sex,
                raw=result.sex,
                valid=None,
                confidence=score(SEX, True, None),
            )
        else:
            # ``<`` is the standard's "unspecified"; anything else was unreadable.
            fields[SEX] = missing(SEX, "sex is unspecified or unreadable")

        # --- names and optional data ----------------------------------------
        fields[SURNAME] = _text_field(
            SURNAME, result.surname, score(SURNAME, bool(result.surname), None)
        )
        fields[GIVEN_NAMES] = _text_field(
            GIVEN_NAMES, result.given_names, score(GIVEN_NAMES, bool(result.given_names), None)
        )
        optional_check = result.field_valid(OPTIONAL_DATA)
        fields[OPTIONAL_DATA] = _text_field(
            OPTIONAL_DATA,
            result.optional_data,
            score(OPTIONAL_DATA, bool(result.optional_data), optional_check),
            valid=optional_check,
        )
        fields[PERSONAL_NUMBER] = _text_field(
            PERSONAL_NUMBER,
            result.personal_number,
            score(PERSONAL_NUMBER, bool(result.personal_number), optional_check),
            valid=optional_check,
        )

        errors = self._collect_errors(result, fields)
        self._cross_validate_dates(fields)

        document = MRZDocument(
            mrz_type=result.mrz_type,
            fields=fields,
            lines=list(result.lines),
            structure_valid=result.structure_valid,
            check_digits_valid=result.check_digits_valid,
            check_digits={c.name: c.valid for c in result.checks},
            confidence=document_confidence(
                structure_valid=result.structure_valid,
                check_digit_score=result.check_digit_score,
                corrections=result.corrections,
                ocr_confidence=ocr_confidence,
            ),
            errors=errors,
            warnings=list(result.warnings),
            corrections=list(result.corrections),
            parser=self.name,
        )
        return document

    # ------------------------------------------------------------- helpers
    def _date_field(
        self,
        name: str,
        raw: str,
        check: Optional[bool],
        kind: str,
        confidence: float,
        result: MRZResult,
    ) -> ParsedField:
        """Resolve a ``YYMMDD`` field to an ISO date, century included."""
        parsed = parse_mrz_date(raw, kind=kind, reference=self.reference_date)
        if parsed is None:
            return missing(name, f"{name} is not a readable YYMMDD date")
        item = ParsedField(
            name=name,
            value=to_iso(parsed),
            raw=raw,
            valid=check,
            confidence=confidence,
            corrected=_was_corrected(result, name),
        )
        if check is False:
            item.add_error(f"check digit does not match {name}")
        return item

    def _cross_validate_dates(self, fields: Dict[str, ParsedField]) -> None:
        """Apply the relationships between dates that the zone cannot check.

        A check digit only proves the digits were *read* correctly, not that the
        date they encode is possible. A birth date in the future is wrong even
        with a perfect check digit, so the calendar overrides the digit.
        """
        birth = _as_date(fields.get(BIRTH_DATE))
        expiry = _as_date(fields.get(EXPIRY_DATE))
        if birth is None and expiry is None:
            return

        warnings, invalid = validate_date_set(
            birth, expiry, None, reference=self.reference_date
        )
        for validator_name in invalid:
            field_name = _DATE_FIELD_BY_VALIDATOR.get(validator_name)
            item = fields.get(field_name) if field_name else None
            if item is None:
                continue
            for message in warnings:
                if validator_name in message:
                    item.add_error(message)
            # A date the calendar contradicts cannot be trusted, whatever its
            # check digit says.
            item.valid = False
            item.confidence = min(item.confidence, 0.25)

    def _collect_errors(
        self, result: MRZResult, fields: Dict[str, ParsedField]
    ) -> List[str]:
        errors: List[str] = []
        if not result.structure_valid:
            errors.append("machine-readable zone failed structural validation")
        for check in result.checks:
            if check.valid is False:
                errors.append(f"{check.name} check digit mismatch")
        return errors


# ------------------------------------------------------------------ builders
def _was_corrected(result: MRZResult, name: str) -> bool:
    return any(correction.startswith(f"{name}:") for correction in result.corrections)


def _code_field(name: str, raw: str, confidence: float) -> ParsedField:
    """A three letter state or nationality code.

    The zone carries no check digit for these, so validity stays ``None``; a
    malformed code is reported as an error rather than silently repaired.
    """
    value = (raw or "").replace(FILLER, "").strip().upper()
    if not value:
        return missing(name, f"{name} is empty")
    item = ParsedField(name=name, value=value, raw=raw, valid=None, confidence=confidence)
    if len(value) != 3 or not value.isalpha():
        item.add_error(f"{name} is not a three-letter code")
        item.confidence = min(item.confidence, 0.3)
    return item


def _text_field(
    name: str, raw: str, confidence: float, valid: Optional[bool] = None
) -> ParsedField:
    value = (raw or "").strip()
    if not value:
        # Optional fields are legitimately empty; that is not an error.
        return ParsedField(name=name, value=None, raw="", valid=None, confidence=0.0)
    return ParsedField(name=name, value=value, raw=raw, valid=valid, confidence=confidence)


def _as_date(item: Optional[ParsedField]) -> Optional[date]:
    if item is None or not item.value:
        return None
    try:
        return date.fromisoformat(str(item.value))
    except ValueError:  # pragma: no cover - value is always ISO by construction
        return None


register_parser("icao9303", ICAOMRZParser)
register_parser("icao", ICAOMRZParser)
