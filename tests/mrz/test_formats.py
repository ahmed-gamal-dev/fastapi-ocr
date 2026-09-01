"""Structural parsing of every supported format."""

from __future__ import annotations

import pytest

from app.services.mrz import MRV_A, MRV_B, TD1, TD2, TD3, build_mrz, detect_format, parse
from app.services.mrz.models import FORMAT_SHAPES
from tests.mrz.conftest import (
    ICAO_TD1_SPECIMEN,
    ICAO_TD2_SPECIMEN,
    ICAO_TD3_SPECIMEN,
    SPECIMEN,
    make_lines,
)


# ------------------------------------------------------- published specimens
def test_td3_specimen_from_the_standard_parses_completely():
    result = parse(ICAO_TD3_SPECIMEN)
    assert result.mrz_type == TD3
    assert result.valid is True
    assert result.issuing_state == "UTO"
    assert result.document_number == "L898902C3"
    assert result.nationality == "UTO"
    assert result.birth_date == "740812"
    assert result.sex == "F"
    assert result.expiry_date == "120415"
    assert result.surname == "ERIKSSON"
    assert result.given_names == "ANNA MARIA"
    assert result.document_category == "passport"


def test_td3_specimen_satisfies_every_check_digit():
    checks = {c.name: c.valid for c in parse(ICAO_TD3_SPECIMEN).checks}
    assert checks == {
        "document_number": True,
        "birth_date": True,
        "expiry_date": True,
        "optional_data": True,
        "composite": True,
    }


def test_td2_specimen_parses():
    result = parse(ICAO_TD2_SPECIMEN)
    assert (result.mrz_type, result.valid) == (TD2, True)
    assert result.document_number == "D23145890"
    assert result.document_category == "identity_card"


def test_td1_specimen_parses():
    result = parse(ICAO_TD1_SPECIMEN)
    assert (result.mrz_type, result.valid) == (TD1, True)
    assert result.document_number == "D23145890"
    assert result.surname == "ERIKSSON"
    assert result.given_names == "ANNA MARIA"


def test_td1_composite_covers_fields_with_no_digit_of_their_own():
    """The composite spans line 1's optional data, which has no check digit of
    its own, so damage there is caught only by the composite."""
    damaged = list(ICAO_TD1_SPECIMEN)
    damaged[0] = damaged[0][:20] + "Q" + damaged[0][21:]
    result = parse(damaged)
    assert result.field_valid("document_number") is True
    assert result.field_valid("composite") is False


def test_the_composite_catches_a_repair_that_its_own_digit_accepted():
    """A single-character slip can be 'repaired' into a different value that
    still satisfies the field's own check digit. The composite is independent,
    so it catches the wrong answer."""
    damaged = list(ICAO_TD1_SPECIMEN)
    damaged[0] = damaged[0][:5] + "Y" + damaged[0][6:]
    result = parse(damaged)
    assert result.field_valid("document_number") is True   # own digit satisfied
    assert result.field_valid("composite") is False        # but the zone disagrees
    assert result.valid is False


def test_a_mod_ten_coincidence_can_defeat_both_digits():
    """Honest limitation, not a defect: substituting D (13) for X (33) shifts
    every weighted sum by a multiple of 10, so no check digit can see it. The
    parser reports what the zone says; it cannot detect this class of error."""
    damaged = list(ICAO_TD1_SPECIMEN)
    damaged[0] = damaged[0][:5] + "X" + damaged[0][6:]
    result = parse(damaged)
    assert result.document_number == "X23145890"
    assert result.valid is True


# ------------------------------------------------------------ round-tripping
@pytest.mark.parametrize(
    "mrz_type,code",
    [(TD1, "I"), (TD2, "I"), (TD3, "P"), (MRV_A, "V"), (MRV_B, "V")],
)
def test_every_format_round_trips(mrz_type, code):
    lines = build_mrz({**SPECIMEN, "document_code": code}, mrz_type)
    rows, width = FORMAT_SHAPES[mrz_type]
    assert len(lines) == rows
    assert all(len(line) == width for line in lines)

    result = parse(lines)
    assert result.mrz_type == mrz_type
    assert result.valid is True
    assert result.document_number == "AB1234567"
    assert result.surname == "SPECIMEN"
    assert result.given_names == "SAMPLE TEST"


@pytest.mark.parametrize(
    "mrz_type,code", [(MRV_A, "V"), (MRV_B, "V")]
)
def test_visas_have_no_composite_check_digit(mrz_type, code):
    """MRV formats define no composite digit; absent is not the same as failing."""
    result = parse(build_mrz({**SPECIMEN, "document_code": code}, mrz_type))
    assert result.check("composite") is None
    assert result.document_category == "visa"


# ----------------------------------------------------------- format detection
@pytest.mark.parametrize(
    "lines,expected",
    [
        (ICAO_TD3_SPECIMEN, TD3),
        (ICAO_TD2_SPECIMEN, TD2),
        (ICAO_TD1_SPECIMEN, TD1),
        (["V" + ICAO_TD3_SPECIMEN[0][1:], ICAO_TD3_SPECIMEN[1]], MRV_A),
    ],
)
def test_format_detection(lines, expected):
    assert detect_format(lines) == expected


@pytest.mark.parametrize("lines", [[], ["short"], ["a" * 44], ["x" * 44] * 4])
def test_undetectable_shapes_return_none(lines):
    assert detect_format(lines) is None


# ------------------------------------------------------------- name encoding
@pytest.mark.parametrize(
    "field,surname,given",
    [
        ("SPECIMEN<<SAMPLE<TEST<<<<<<", "SPECIMEN", "SAMPLE TEST"),
        ("SPECIMEN<<SAMPLE<<<<<<<<<<<", "SPECIMEN", "SAMPLE"),
        ("SINGLENAME<<<<<<<<<<<<<<<<<", "SINGLENAME", ""),
        ("VAN<DER<BERG<<JAN<<<<<<<<<<", "VAN DER BERG", "JAN"),
        ("<<<<<<<<<<<<<<<<<<<<<<<<<<<", "", ""),
    ],
)
def test_name_field_splitting(field, surname, given):
    from app.services.mrz.parser import _parse_name

    assert _parse_name(field) == (surname, given)


def test_a_name_with_no_separator_is_treated_as_a_surname():
    """The standard prescribes this for single-identifier names."""
    from app.services.mrz.parser import _parse_name

    assert _parse_name("MONONYM<<<<<<<") == ("MONONYM", "")


# --------------------------------------------------- extended document number
def test_a_document_number_longer_than_nine_characters_is_recovered():
    """ICAO 9303-4 4.2.2: the number overflows into the optional data field,
    the in-place check digit becomes a filler, and the real digit follows the
    overflow."""
    from app.services.mrz.checkdigit import compute_check_digit

    number = "AB1234567890"
    tail = number[8:] + compute_check_digit(number)
    line2 = (
        number[:8] + "<" + "<" + "UTO"
        + "800101" + compute_check_digit("800101")
        + "M"
        + "301231" + compute_check_digit("301231")
        + tail + "<" * (14 - len(tail))
    )
    line2 += compute_check_digit(line2[28:42])
    line2 += compute_check_digit(line2[0:10] + line2[13:20] + line2[21:43])

    result = parse(["P<UTOSPECIMEN<<SAMPLE<<<<<<<<<<<<<<<<<<<<<<<", line2])
    assert result.document_number == number
    assert result.valid is True
    assert "document_number:extended_field" in result.corrections


# ------------------------------------------------------------------ hardening
@pytest.mark.parametrize("sex,expected", [("M", "M"), ("F", "F"), ("X", "X")])
def test_sex_values_are_preserved(sex, expected):
    assert parse(make_lines(sex=sex)).sex == expected


def test_an_unreadable_sex_is_reported_as_unknown():
    lines = make_lines()
    damaged = [lines[0], lines[1][:20] + "9" + lines[1][21:]]
    result = parse(damaged)
    assert result.sex is None
    assert any("sex" in w for w in result.warnings)


def test_short_lines_are_padded_rather_than_rejected():
    lines = make_lines()
    result = parse([lines[0][:-4], lines[1]])
    assert result is not None
    assert len(result.lines[0]) == 44


@pytest.mark.parametrize("lines", [[], [""], ["   ", "  "], None])
def test_empty_input_returns_none(lines):
    assert parse(lines or []) is None


def test_prose_is_not_parsed_as_a_zone():
    assert parse(["This is an ordinary sentence of text, not a zone at all."]) is None
