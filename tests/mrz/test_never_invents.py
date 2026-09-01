"""The cardinal rule: the parser never invents or guesses a value.

Every test here takes damaged or absent input and asserts that the parser
reports the gap rather than filling it.
"""

from __future__ import annotations

import pytest

from app.services.mrz import (
    BIRTH_DATE,
    DOCUMENT_NUMBER,
    EXPIRY_DATE,
    GIVEN_NAMES,
    NATIONALITY,
    OPTIONAL_DATA,
    SEX,
    SURNAME,
)
from app.services.mrz.synthetic import damage
from tests.mrz.conftest import ICAO_TD3_SPECIMEN, make_lines


@pytest.mark.parametrize(
    "lines",
    [
        ["Dear customer, your order has shipped.", "Thank you for your business."],
        ["INVOICE 2026-09", "TOTAL DUE 1240.00"],
        ["", ""],
        ["<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"] * 2,
        ["A" * 44, "B" * 44],
    ],
)
def test_non_zone_input_yields_nothing_at_all(parser, lines):
    """Not a partial document, not empty fields - nothing."""
    assert parser.parse_lines(lines) is None


def test_an_unreadable_sex_is_none_not_a_default(parser):
    lines = make_lines()
    document = parser.parse_lines([lines[0], damage(lines[1], 20, "9")])
    assert document.value(SEX) is None
    assert document.confidence_of(SEX) == 0.0
    assert document.get(SEX).errors


def test_an_empty_optional_field_is_none_not_an_empty_string(parser):
    document = parser.parse_lines(make_lines(optional_data=""))
    assert document.value(OPTIONAL_DATA) is None


def test_an_absent_given_name_is_none(parser):
    document = parser.parse_lines(make_lines(given_names=""))
    assert document.value(GIVEN_NAMES) is None
    assert document.value(SURNAME) == "SPECIMEN"


def test_a_field_that_fails_its_check_digit_is_reported_not_repaired(parser):
    """W, K and H have no confusable twin, so no repair is possible. The value
    read is returned as-is, flagged invalid, with low confidence - neither
    silently corrected nor silently dropped."""
    lines = make_lines(document_number="WWWKKKHHH")
    broken = damage(lines[1], 9, str((int(lines[1][9]) + 5) % 10))
    document = parser.parse_lines([lines[0], broken])

    field = document.get(DOCUMENT_NUMBER)
    assert field.value == "WWWKKKHHH"
    assert field.valid is False
    assert field.confidence <= 0.25
    assert field.errors
    assert document.valid is False


def test_a_unique_repair_is_applied_and_disclosed(parser):
    """Correction is allowed when exactly one candidate satisfies the check
    digit - but it is recorded, and it costs confidence."""
    lines = make_lines(document_number="123456789")
    document = parser.parse_lines([lines[0], damage(lines[1], 0, "I")])

    field = document.get(DOCUMENT_NUMBER)
    assert field.value == "123456789"
    assert field.corrected is True
    assert "document_number:check_digit_repair" in document.corrections
    assert field.confidence < 0.98


def test_an_ambiguous_repair_is_refused(parser):
    """Two single-character fixes satisfy this check digit. The parser must keep
    what it read and say so, rather than pick one."""
    lines = make_lines(document_number="AB2134567")
    document = parser.parse_lines([lines[0], damage(lines[1], 1, "8")])

    assert document.get(DOCUMENT_NUMBER).value == "A82134567"  # exactly as read
    assert "document_number:check_digit_repair" not in document.corrections
    assert any("unambiguous" in w for w in document.warnings)
    assert document.valid is False


def test_an_impossible_date_is_not_coerced_into_a_real_one(parser):
    """Month 19 cannot be repaired into a valid month by guesswork."""
    lines = make_lines()
    broken = lines[1][:13] + "801901" + lines[1][19:]
    document = parser.parse_lines([lines[0], broken])
    assert document is None or document.value(BIRTH_DATE) is None


def test_a_calendar_contradiction_overrides_a_valid_check_digit(parser):
    """Every digit agrees, but an expiry before the birth date is impossible.
    The parser must not report it as trustworthy."""
    document = parser.parse_lines(
        make_lines(birth_date="260901", expiry_date="260101")
    )
    assert document.check_digits_valid is True  # the digits really do agree
    expiry = document.get(EXPIRY_DATE)
    assert expiry.valid is False
    assert expiry.confidence <= 0.25
    assert any("not after" in error for error in expiry.errors)


def test_a_malformed_country_code_is_flagged_not_fixed(parser):
    lines = make_lines()
    broken = damage(lines[1], 10, "1")  # nationality becomes '1TO'
    document = parser.parse_lines([lines[0], broken])
    nationality = document.get(NATIONALITY)
    # Type coercion maps '1' to 'I' because the standard says the field is
    # alphabetic - it never picks a plausible-looking country instead.
    assert nationality.value in ("ITO", "1TO")
    assert nationality.value != "UTO"


def test_confidence_never_exceeds_one(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN, ocr_confidence=1.0)
    assert 0.0 <= document.confidence <= 1.0
    for field in document.fields.values():
        assert 0.0 <= field.confidence <= 1.0


def test_a_wholly_damaged_zone_is_rejected_rather_than_half_parsed(parser):
    garbage = ["P<UTO" + "Q" * 39, "Q" * 44]
    document = parser.parse_lines(garbage)
    assert document is None or document.valid is False


def test_check_digits_are_never_reported_as_valid_when_absent(parser):
    """MRV formats define no composite digit. Absent must read as 'not
    applicable', never as 'passed'."""
    document = parser.parse_lines(
        make_lines("MRV_A", document_code="V"), mrz_type="MRV_A"
    )
    assert document.check_digits.get("composite") is None
