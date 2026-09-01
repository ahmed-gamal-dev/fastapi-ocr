"""The zone alphabet and conservative glyph normalisation."""

from __future__ import annotations

import pytest

from app.services.mrz.charset import (
    FILLER,
    MRZ_ALPHABET,
    coerce_alpha,
    coerce_numeric,
    is_filler_only,
    mrz_ratio,
    name_to_mrz,
    normalize_char,
    normalize_line,
    pad_or_trim,
    strip_fillers,
)


def test_alphabet_is_exactly_the_standard_set():
    assert MRZ_ALPHABET == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    assert FILLER == "<"


@pytest.mark.parametrize("char", list("ABZ019<"))
def test_alphabet_characters_pass_through(char):
    assert normalize_char(char) == char


def test_lower_case_is_folded_up():
    assert normalize_line("p<utoeriksson") == "P<UTOERIKSSON"


@pytest.mark.parametrize("glyph", ["«", "‹", "＜", " ", "_", "-", "*", "/", "|"])
def test_filler_lookalikes_become_fillers(glyph):
    assert set(normalize_char(glyph)) == {FILLER}


def test_double_chevron_expands_to_two_fillers():
    assert normalize_char("«") == "<<"
    assert normalize_line("P«EGY") == "P<<EGY"


def test_arabic_indic_digits_are_converted():
    assert normalize_line("٧٤٠٨١٢") == "740812"
    assert normalize_line("۷۴۰۸۱۲") == "740812"


def test_accents_are_folded_to_their_base_letter():
    assert normalize_line("MÜLLER") == "MULLER"
    assert normalize_line("JOSÉ") == "JOSE"


def test_characters_with_no_mapping_are_dropped():
    """A glyph that is neither alphabet nor a known look-alike is removed
    rather than guessed at."""
    assert normalize_line("AB中文CD") == "ABCD"


def test_normalising_empty_input_is_safe():
    assert normalize_line("") == ""
    assert normalize_line(None) == ""


def test_pad_extends_with_fillers():
    assert pad_or_trim("ABC", 6) == "ABC<<<"


def test_trim_cuts_to_length():
    assert pad_or_trim("ABCDEFGH", 3) == "ABC"


def test_pad_or_trim_leaves_an_exact_length_alone():
    assert pad_or_trim("ABC", 3) == "ABC"


def test_strip_fillers_turns_a_field_into_a_value():
    assert strip_fillers("ERIKSSON<<<<<") == "ERIKSSON"
    assert strip_fillers("ANNA<MARIA<<<") == "ANNA MARIA"
    assert strip_fillers("") == ""
    assert strip_fillers(None) == ""


def test_is_filler_only_detects_an_empty_field():
    assert is_filler_only("<<<<<<") is True
    assert is_filler_only("") is True
    assert is_filler_only("A<<<<<") is False


def test_numeric_coercion_maps_letter_lookalikes_to_digits():
    assert coerce_numeric("74O8I2") == "740812"
    # O->0 I->1 Z->2 A->4 S->5 G->6 T->7 B->8
    assert coerce_numeric("OIZASGTB") == "01245678"


def test_numeric_coercion_leaves_fillers_alone():
    assert coerce_numeric("<<<<<<") == "<<<<<<"


def test_alpha_coercion_maps_digit_lookalikes_to_letters():
    assert coerce_alpha("3GY") == "3GY"  # 3 has no confident letter twin
    assert coerce_alpha("0GY") == "OGY"
    assert coerce_alpha("U5A") == "USA"


def test_ratio_of_a_zone_line_is_one():
    assert mrz_ratio("P<UTOERIKSSON<<ANNA<MARIA<<<<") == 1.0


def test_ratio_falls_with_foreign_characters():
    assert mrz_ratio("Hello, world!") < 0.9
    assert mrz_ratio("") == 0.0


def test_name_encoding_matches_the_zone_convention():
    assert name_to_mrz("ANNA MARIA") == "ANNA<MARIA"
    assert name_to_mrz("O'BRIEN") == "OBRIEN"
    assert name_to_mrz("JOSÉ") == "JOSE"
    assert name_to_mrz("") == ""
