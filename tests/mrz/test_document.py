"""Per-field results, the document container and confidence scoring."""

from __future__ import annotations

import pytest

from app.services.mrz import (
    BIRTH_DATE,
    DOCUMENT_CATEGORY,
    DOCUMENT_CODE,
    DOCUMENT_NUMBER,
    EXPIRY_DATE,
    FIELD_NAMES,
    GIVEN_NAMES,
    ISSUING_STATE,
    NATIONALITY,
    OPTIONAL_DATA,
    SEX,
    SURNAME,
)
from app.services.mrz.confidence import (
    CHECK_FAILED,
    CHECK_PASSED,
    UNPROTECTED_COMPOSITE_OK,
    document_confidence,
    field_confidence,
)
from app.services.mrz.fields import ParsedField, missing
from tests.mrz.conftest import ICAO_TD3_SPECIMEN, make_lines


# ------------------------------------------------------------- ParsedField
def test_a_parsed_field_reports_presence():
    assert ParsedField("x", value="A").present is True
    assert ParsedField("x", value=None).present is False


def test_verified_means_present_and_confirmed():
    assert ParsedField("x", value="A", valid=True).verified is True
    assert ParsedField("x", value="A", valid=False).verified is False
    assert ParsedField("x", value="A", valid=None).verified is False
    assert ParsedField("x", value=None, valid=True).verified is False


def test_a_missing_field_has_no_value_and_a_reason():
    item = missing(DOCUMENT_NUMBER, "document number is empty")
    assert item.value is None
    assert item.confidence == 0.0
    assert item.errors == ["document number is empty"]


def test_errors_are_not_duplicated():
    item = ParsedField("x")
    item.add_error("bad")
    item.add_error("bad")
    assert item.errors == ["bad"]


def test_field_serialisation_exposes_the_four_facets():
    item = ParsedField("x", value="A", valid=True, confidence=0.98, errors=["e"])
    assert item.to_dict() == {
        "value": "A",
        "valid": True,
        "confidence": 0.98,
        "errors": ["e"],
        "corrected": False,
    }


# -------------------------------------------------------------- confidence
def test_confidence_ordering_follows_the_evidence():
    verified = field_confidence(present=True, check_valid=True, composite_valid=True)
    unprotected = field_confidence(present=True, check_valid=None, composite_valid=True)
    contradicted = field_confidence(present=True, check_valid=False, composite_valid=True)
    assert verified > unprotected > contradicted
    assert verified == CHECK_PASSED
    assert contradicted == CHECK_FAILED
    assert unprotected == UNPROTECTED_COMPOSITE_OK


def test_an_absent_field_scores_zero():
    assert field_confidence(present=False, check_valid=True, composite_valid=True) == 0.0


def test_a_failing_composite_lowers_unprotected_fields():
    good = field_confidence(present=True, check_valid=None, composite_valid=True)
    bad = field_confidence(present=True, check_valid=None, composite_valid=False)
    none = field_confidence(present=True, check_valid=None, composite_valid=None)
    assert good > none > bad


def test_a_correction_costs_confidence():
    plain = field_confidence(present=True, check_valid=True, composite_valid=True)
    fixed = field_confidence(
        present=True, check_valid=True, composite_valid=True, corrected=True
    )
    assert fixed < plain


def test_recognition_quality_tempers_the_score():
    perfect = field_confidence(
        present=True, check_valid=True, composite_valid=True, ocr_confidence=1.0
    )
    poor = field_confidence(
        present=True, check_valid=True, composite_valid=True, ocr_confidence=0.1
    )
    assert perfect > poor > 0.0


def test_an_unsupplied_ocr_confidence_does_not_penalise():
    """Zero means 'not supplied', not 'recognised badly'."""
    assert field_confidence(
        present=True, check_valid=True, composite_valid=True, ocr_confidence=0.0
    ) == CHECK_PASSED


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_document_confidence_stays_in_range(score):
    value = document_confidence(structure_valid=True, check_digit_score=score)
    assert 0.0 <= value <= 1.0


def test_a_structurally_invalid_zone_is_capped_low():
    assert document_confidence(structure_valid=False, check_digit_score=1.0) <= 0.25


# ---------------------------------------------------------------- document
def test_every_declared_field_is_produced(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert set(document.fields) == set(FIELD_NAMES)


def test_values_are_reachable_by_name(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert document.value(DOCUMENT_CODE) == "P"
    assert document.value(DOCUMENT_CATEGORY) == "passport"
    assert document.value(ISSUING_STATE) == "UTO"
    assert document.value(DOCUMENT_NUMBER) == "L898902C3"
    assert document.value(NATIONALITY) == "UTO"
    assert document.value(BIRTH_DATE) == "1974-08-12"
    assert document.value(SEX) == "F"
    assert document.value(EXPIRY_DATE) == "2012-04-15"
    assert document.value(SURNAME) == "ERIKSSON"
    assert document.value(GIVEN_NAMES) == "ANNA MARIA"


def test_dates_are_returned_as_iso_strings(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert document.value(BIRTH_DATE) == "1974-08-12"
    assert document.get(BIRTH_DATE).raw == "740812"


def test_an_unknown_field_returns_the_default(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert document.value("no_such_field") is None
    assert document.value("no_such_field", "fallback") == "fallback"
    assert document.get("no_such_field") is None
    assert document.confidence_of("no_such_field") == 0.0
    assert document.is_valid("no_such_field") is None


def test_check_digit_protected_fields_are_marked_verified(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert set(document.verified_fields) >= {DOCUMENT_NUMBER, BIRTH_DATE, EXPIRY_DATE}
    # A field the standard does not protect is present but unverified.
    assert document.is_valid(ISSUING_STATE) is None
    assert document.value(ISSUING_STATE) == "UTO"


def test_full_name_joins_the_identifiers(parser):
    assert parser.parse_lines(ICAO_TD3_SPECIMEN).full_name == "ANNA MARIA ERIKSSON"


def test_full_name_is_none_when_no_name_was_read(parser):
    document = parser.parse_lines(make_lines(surname="", given_names=""))
    assert document.full_name is None


def test_present_fields_lists_only_what_was_read(parser):
    document = parser.parse_lines(make_lines(optional_data=""))
    assert OPTIONAL_DATA not in document.present_fields
    assert DOCUMENT_NUMBER in document.present_fields


def test_serialisation_round_trip(parser):
    data = parser.parse_lines(ICAO_TD3_SPECIMEN).to_dict()
    assert data["mrz_type"] == "TD3"
    assert data["parser"] == "icao9303"
    assert data["valid"] is True
    assert data["fields"][DOCUMENT_NUMBER]["value"] == "L898902C3"
    assert data["fields"][DOCUMENT_NUMBER]["valid"] is True
    assert data["check_digits"]["composite"] is True


def test_raw_zone_text_is_withheld_unless_asked_for(parser):
    """The raw lines reproduce the whole zone, so callers opt in rather than
    receive them by accident."""
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert "raw" not in document.to_dict()
    assert "lines" not in document.to_dict()
    assert document.to_dict(include_raw=True)["raw"].startswith("P<UTO")


def test_flat_dict_gives_just_the_values(parser):
    flat = parser.parse_lines(ICAO_TD3_SPECIMEN).to_flat_dict()
    assert flat[DOCUMENT_NUMBER] == "L898902C3"
    assert set(flat) == set(FIELD_NAMES)


def test_summary_is_safe_to_log(parser):
    """It must describe the parse without disclosing any of its content."""
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    summary = document.summary()
    assert "TD3" in summary and "valid=True" in summary
    for secret in ("ERIKSSON", "ANNA", "L898902C3", "740812"):
        assert secret not in summary
