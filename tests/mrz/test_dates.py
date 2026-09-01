"""Date resolution: two digit years, centuries and ISO output."""

from __future__ import annotations

from datetime import date

import pytest

from app.services.mrz import BIRTH_DATE, EXPIRY_DATE
from app.services.mrz.icao import ICAOMRZParser
from tests.mrz.conftest import REFERENCE_DATE, make_lines


def parse_with(birth: str = "800101", expiry: str = "301231", **kwargs):
    parser = ICAOMRZParser(reference_date=REFERENCE_DATE, **kwargs)
    return parser.parse_lines(make_lines(birth_date=birth, expiry_date=expiry))


@pytest.mark.parametrize(
    "yymmdd,expected",
    [
        ("740812", "1974-08-12"),
        ("800101", "1980-01-01"),
        ("050101", "2005-01-01"),
        ("991231", "1999-12-31"),
        ("000229", "2000-02-29"),  # 2000 was a leap year, 1900 was not
    ],
)
def test_birth_year_century_resolution(yymmdd, expected):
    """A birth date cannot be in the future, which resolves the century."""
    assert parse_with(birth=yymmdd).value(BIRTH_DATE) == expected


@pytest.mark.parametrize(
    "yymmdd,expected",
    [
        ("301231", "2030-12-31"),
        ("260101", "2026-01-01"),
        ("120415", "2012-04-15"),  # an expired document is legitimate input
    ],
)
def test_expiry_year_century_resolution(yymmdd, expected):
    assert parse_with(expiry=yymmdd).value(EXPIRY_DATE) == expected


def test_dates_are_iso_formatted():
    document = parse_with()
    assert document.value(BIRTH_DATE) == "1980-01-01"
    assert document.value(EXPIRY_DATE) == "2030-12-31"


def test_the_raw_zone_digits_are_preserved_alongside_the_value():
    field = parse_with(birth="740812").get(BIRTH_DATE)
    assert field.raw == "740812"
    assert field.value == "1974-08-12"


def test_resolution_is_anchored_to_the_reference_date_not_the_clock():
    """The same zone must parse identically whenever the suite is run."""
    early = ICAOMRZParser(reference_date=date(2010, 1, 1))
    late = ICAOMRZParser(reference_date=date(2040, 1, 1))
    lines = make_lines(birth_date="050101")
    assert early.parse_lines(lines).value(BIRTH_DATE) == "2005-01-01"
    assert late.parse_lines(lines).value(BIRTH_DATE) == "2005-01-01"


def test_a_leap_day_in_a_non_leap_year_is_not_accepted():
    """29 February 2001 does not exist, so it must not be reported as a date."""
    lines = make_lines()
    broken = lines[1][:13] + "010229" + lines[1][19:]
    parser = ICAOMRZParser(reference_date=REFERENCE_DATE, strict=False)
    document = parser.parse_lines([lines[0], broken])
    assert document is None or document.value(BIRTH_DATE) != "2001-02-29"


def test_date_fields_carry_their_check_digit_verdict():
    document = parse_with()
    assert document.is_valid(BIRTH_DATE) is True
    assert document.is_valid(EXPIRY_DATE) is True
