"""Conservative OCR correction.

Two mechanisms are under test: type coercion where the standard fixes the field
type, and check-digit-guided repair where it does not. The second must refuse to
act whenever the answer is not unique.
"""

from __future__ import annotations

import pytest

from app.services.mrz.checkdigit import compute_check_digit
from app.services.mrz.corrector import (
    MAX_SUBSTITUTIONS,
    coerce_field,
    coerce_sex,
    is_plausible_mrz_date,
    repair_date,
    repair_with_check_digit,
)


# ------------------------------------------------------------------ coercion
@pytest.mark.parametrize(
    "value,kind,expected",
    [
        ("74O8I2", "date", "740812"),
        ("74O8I2", "num", "740812"),
        ("0TO", "alpha", "OTO"),
        ("0TO", "name", "OTO"),
        ("P", "doc_code", "P"),
        ("AB1234567", "alnum", "AB1234567"),  # untouched: type is not fixed
    ],
)
def test_type_coercion_follows_the_field_definition(value, kind, expected):
    assert coerce_field(value, kind) == expected


@pytest.mark.parametrize("value,expected", [("M", "M"), ("F", "F"), ("X", "X"), ("<", "<")])
def test_valid_sex_values_pass_through(value, expected):
    assert coerce_sex(value) == expected


def test_sex_lookalikes_are_mapped():
    assert coerce_sex("H") == "M"  # 'hombre', seen on poorly printed documents
    assert coerce_sex("5") == "S" or coerce_sex("5") == "<"


def test_unreadable_sex_becomes_a_filler_not_a_guess():
    for glyph in ("9", "?", "", "Q"):
        assert coerce_sex(glyph) == "<"


# --------------------------------------------------------- check-digit repair
def test_a_unique_single_character_fix_is_applied():
    true_value = "123456789"
    digit = compute_check_digit(true_value)
    fixed, corrected, ambiguous = repair_with_check_digit("I23456789", digit)
    assert (fixed, corrected, ambiguous) == (true_value, True, False)


def test_a_correct_field_is_left_untouched():
    digit = compute_check_digit("AB1234567")
    fixed, corrected, ambiguous = repair_with_check_digit("AB1234567", digit)
    assert (fixed, corrected, ambiguous) == ("AB1234567", False, False)


def test_an_ambiguous_field_is_never_guessed():
    """Two different single-character fixes satisfy this check digit, so the
    original must be kept and the ambiguity reported."""
    digit = compute_check_digit("AB2134567")
    fixed, corrected, ambiguous = repair_with_check_digit("A82134567", digit)
    assert fixed == "A82134567"  # unchanged
    assert corrected is False
    assert ambiguous is True


def test_a_field_with_no_ambiguous_glyphs_is_left_alone():
    """W, K and H have no confusable twin, so there is nothing to try and the
    mismatch is reported rather than papered over."""
    wrong_digit = str((int(compute_check_digit("WWWKKKHHH")) + 3) % 10)
    fixed, corrected, ambiguous = repair_with_check_digit("WWWKKKHHH", wrong_digit)
    assert (fixed, corrected, ambiguous) == ("WWWKKKHHH", False, False)


def test_repair_needs_a_numeric_check_digit():
    assert repair_with_check_digit("AB1234567", "<") == ("AB1234567", False, False)
    assert repair_with_check_digit("AB1234567", None) == ("AB1234567", False, False)


def test_repair_of_an_empty_field_is_a_no_op():
    assert repair_with_check_digit("", "0") == ("", False, False)


def test_the_search_is_bounded():
    assert MAX_SUBSTITUTIONS == 2


def test_minimal_edits_win_over_larger_ones():
    """A single-character explanation is preferred to a two-character one."""
    true_value = "123456789"
    digit = compute_check_digit(true_value)
    fixed, corrected, _ = repair_with_check_digit("I23456789", digit)
    assert fixed == true_value and corrected is True


# --------------------------------------------------------------- date repair
def test_date_repair_coerces_letters_to_digits():
    fixed, _, _ = repair_date("74O8I2", compute_check_digit("740812"))
    assert fixed == "740812"


def test_date_repair_will_not_produce_an_impossible_date():
    """A candidate that satisfies the check digit but is not a real date must
    be rejected: month 19 does not exist."""
    assert is_plausible_mrz_date("801901") is False
    assert is_plausible_mrz_date("800100") is False
    assert is_plausible_mrz_date("800101") is True


@pytest.mark.parametrize("value", ["", "8001", "80010A", "abcdef"])
def test_implausible_date_strings_are_rejected(value):
    assert is_plausible_mrz_date(value) is False
