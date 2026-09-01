"""Finding a zone inside plain text, independent of any image or OCR engine."""

from __future__ import annotations

import pytest

from app.services.mrz import TD1, TD3
from app.services.mrz.textscan import (
    candidate_groups,
    formats_for,
    looks_like_mrz_line,
    parse_lines,
    parse_text,
    split_joined_lines,
    strict_mrz_ratio,
)
from tests.mrz.conftest import ICAO_TD1_SPECIMEN, ICAO_TD3_SPECIMEN


# ------------------------------------------------------------ line detection
@pytest.mark.parametrize("line", ICAO_TD3_SPECIMEN + ICAO_TD1_SPECIMEN)
def test_real_zone_lines_are_recognised(line):
    assert looks_like_mrz_line(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "Republic of Utopia, page 2.",
        "THIS IS AN ORDINARY UPPERCASE HEADING LINE",
        "Date of birth / Date de naissance",
        "SHORT<<LINE",
        "",
    ],
)
def test_ordinary_text_is_not_mistaken_for_a_zone_line(line):
    assert looks_like_mrz_line(line) is False


def test_lower_case_counts_against_a_line():
    """The zone is upper case by definition, so folding case before scoring
    would make any sentence look like one."""
    assert strict_mrz_ratio("P<UTOERIKSSON<<ANNA") == 1.0
    assert strict_mrz_ratio("p<utoeriksson<<anna") < 0.2


def test_a_line_without_fillers_is_not_a_zone_line():
    assert looks_like_mrz_line("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") is False


# ------------------------------------------------------------------ grouping
def test_candidate_groups_prefer_the_last_lines():
    lines = ["HEADER TEXT", "MORE HEADER", *ICAO_TD3_SPECIMEN]
    groups = candidate_groups(lines)
    assert any(group == ICAO_TD3_SPECIMEN for group in groups)


def test_no_candidates_in_ordinary_text():
    assert candidate_groups(["Hello there.", "Nothing to see."]) == []


@pytest.mark.parametrize(
    "lines,expected",
    [
        (["x" * 44] * 2, TD3),
        (["x" * 30] * 3, TD1),
    ],
)
def test_formats_are_offered_by_shape(lines, expected):
    assert expected in list(formats_for(lines))


def test_incompatible_shapes_offer_no_format():
    assert list(formats_for(["x" * 44])) == []
    assert list(formats_for(["x" * 10] * 2)) == []


# --------------------------------------------------------------- joined rows
def test_two_rows_merged_into_one_string_are_recovered():
    joined = "".join(ICAO_TD3_SPECIMEN)
    assert split_joined_lines(joined) == [ICAO_TD3_SPECIMEN]


def test_a_normal_line_is_not_split():
    assert split_joined_lines(ICAO_TD3_SPECIMEN[0]) == []


def test_a_joined_zone_still_parses():
    result = parse_text("".join(ICAO_TD3_SPECIMEN))
    assert result is not None and result.valid is True


# ------------------------------------------------------------------ parsing
def test_a_zone_is_found_among_surrounding_text():
    blob = "\n".join(
        [
            "REPUBLIC OF UTOPIA",
            "PASSPORT / PASSEPORT",
            "Surname / Nom",
            *ICAO_TD3_SPECIMEN,
            "",
        ]
    )
    result = parse_text(blob)
    assert result.mrz_type == TD3
    assert result.valid is True
    assert result.document_number == "L898902C3"


def test_a_three_line_zone_is_found_among_text():
    blob = "\n".join(["IDENTITY CARD", *ICAO_TD1_SPECIMEN])
    result = parse_text(blob)
    assert result.mrz_type == TD1
    assert result.valid is True


@pytest.mark.parametrize("text", ["", "   ", "Just an ordinary sentence.", "\n\n\n"])
def test_text_with_no_zone_returns_none(text):
    assert parse_text(text) is None


def test_parse_lines_is_the_list_form_of_parse_text():
    assert parse_lines(ICAO_TD3_SPECIMEN).document_number == "L898902C3"


def test_the_best_scoring_parse_wins():
    """Two candidate groupings are present; the one whose check digits agree
    must be chosen."""
    decoy = ["P<UTOWRONG<<DECOY<<<<<<<<<<<<<<<<<<<<<<<<<<<", "X" * 44]
    blob = "\n".join([*decoy, *ICAO_TD3_SPECIMEN])
    result = parse_text(blob)
    assert result.document_number == "L898902C3"
    assert result.valid is True
