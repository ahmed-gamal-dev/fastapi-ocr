"""ICAO 9303 check digit arithmetic."""

from __future__ import annotations

import pytest

from app.services.mrz.checkdigit import (
    WEIGHTS,
    char_value,
    compute_check_digit,
    verify_check_digit,
)


def test_weights_cycle_seven_three_one():
    assert WEIGHTS == (7, 3, 1)


@pytest.mark.parametrize(
    "char,expected",
    [("0", 0), ("9", 9), ("A", 10), ("B", 11), ("Z", 35), ("<", 0)],
)
def test_character_values(char, expected):
    assert char_value(char) == expected


@pytest.mark.parametrize("char", ["a", "-", " ", "!", "é"])
def test_characters_outside_the_alphabet_have_no_value(char):
    assert char_value(char) is None


@pytest.mark.parametrize(
    "field,digit",
    [
        # The worked examples printed in ICAO Doc 9303 part 3.
        ("123456789", "7"),
        ("740812", "2"),
        ("120415", "9"),
        ("L898902C3", "6"),
        ("D23145890", "7"),
        ("ZE184226B<<<<<", "1"),
    ],
)
def test_known_check_digits(field, digit):
    assert compute_check_digit(field) == digit


def test_all_filler_field_checks_to_zero():
    assert compute_check_digit("<<<<<<<<<") == "0"


def test_empty_field_checks_to_zero():
    assert compute_check_digit("") == "0"


def test_a_field_with_an_illegal_character_has_no_check_digit():
    assert compute_check_digit("ABC-123") is None
    assert compute_check_digit("abc") is None


def test_check_digit_is_position_sensitive():
    """The 7-3-1 weighting must make transpositions detectable."""
    assert compute_check_digit("AB1234567") != compute_check_digit("BA1234567")


def test_verification_accepts_the_right_digit():
    assert verify_check_digit("123456789", "7") is True


def test_verification_rejects_the_wrong_digit():
    assert verify_check_digit("123456789", "3") is False


def test_verification_is_not_applicable_without_a_digit():
    """Absent is not the same as wrong: it is simply not checkable."""
    assert verify_check_digit("123456789", None) is None
    assert verify_check_digit("123456789", "") is None


def test_filler_digit_over_an_empty_field_is_not_applicable():
    assert verify_check_digit("<<<<<<", "<") is None


def test_filler_digit_over_a_populated_field_is_a_failure():
    assert verify_check_digit("AB1234567", "<") is False


def test_non_numeric_digit_is_a_failure():
    assert verify_check_digit("123456789", "A") is False


def test_verification_fails_when_the_field_is_unparseable():
    assert verify_check_digit("ABC-123", "5") is False
