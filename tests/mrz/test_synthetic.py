"""The synthetic zone builder used by this suite.

If the builder is wrong, every test that relies on it is meaningless, so it is
verified independently against the published specimens.
"""

from __future__ import annotations

import pytest

from app.services.mrz import MRV_A, MRV_B, TD1, TD2, TD3, build_mrz, parse
from app.services.mrz.checkdigit import compute_check_digit
from app.services.mrz.models import FORMAT_SHAPES
from app.services.mrz.synthetic import damage, name_field


def test_the_builder_reproduces_the_published_td3_specimen():
    """Independent proof the builder is correct: it must reproduce the worked
    example from the standard, character for character."""
    lines = build_mrz(
        {
            "document_code": "P",
            "issuing_state": "UTO",
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "document_number": "L898902C3",
            "nationality": "UTO",
            "birth_date": "740812",
            "sex": "F",
            "expiry_date": "120415",
            "optional_data": "ZE184226B",
        },
        TD3,
    )
    assert lines == [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]


@pytest.mark.parametrize("mrz_type", [TD1, TD2, TD3, MRV_A, MRV_B])
def test_built_zones_have_the_right_shape(mrz_type):
    rows, width = FORMAT_SHAPES[mrz_type]
    lines = build_mrz({"document_number": "AB1234567", "birth_date": "800101",
                       "expiry_date": "301231", "sex": "M"}, mrz_type)
    assert len(lines) == rows
    assert all(len(line) == width for line in lines)


@pytest.mark.parametrize("mrz_type", [TD1, TD2, TD3, MRV_A, MRV_B])
def test_built_zones_always_satisfy_their_own_check_digits(mrz_type):
    code = "V" if mrz_type in (MRV_A, MRV_B) else "P"
    lines = build_mrz(
        {
            "document_code": code,
            "document_number": "AB1234567",
            "birth_date": "800101",
            "expiry_date": "301231",
            "sex": "M",
            "surname": "SPECIMEN",
            "given_names": "SAMPLE",
        },
        mrz_type,
    )
    result = parse(lines)
    assert result.valid is True


def test_an_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported format"):
        build_mrz({}, "TD9")


def test_name_encoding_pads_to_the_field_width():
    field = name_field("SPECIMEN", "SAMPLE TEST", 39)
    assert field.startswith("SPECIMEN<<SAMPLE<TEST")
    assert len(field) == 39


def test_long_names_are_truncated_to_the_field():
    assert len(name_field("A" * 40, "B" * 40, 39)) == 39


def test_damage_replaces_exactly_one_character():
    assert damage("ABCDEF", 2, "X") == "ABXDEF"


def test_damage_rejects_a_position_outside_the_line():
    with pytest.raises(IndexError):
        damage("ABC", 99, "X")


def test_the_builder_computes_digits_rather_than_copying_them():
    lines = build_mrz({"document_number": "AB1234567", "birth_date": "800101",
                       "expiry_date": "301231", "sex": "M"}, TD3)
    assert lines[1][9] == compute_check_digit("AB1234567")
    assert lines[1][19] == compute_check_digit("800101")
    assert lines[1][27] == compute_check_digit("301231")
