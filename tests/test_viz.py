"""Label-anchored extraction of visual-zone fields.

Every fixture here is invented. The names and places are ordinary words chosen
to exercise the layouts - no real document data appears in this suite.
"""

from __future__ import annotations

import pytest

from app.services import viz
from tests.conftest import block

# The two arrangements that occur in practice.
LABEL_AR = "الاسم"
NAME_AR = "سالم بن محمد بن عبدالله الحارثي"
PLACE_AR = "الدمام"
PLACE_EN = "DAMMAM"


def separate_label_page():
    """Label in its own box, value beside it - Arabic reads right to left."""
    return [
        block(LABEL_AR, x=900, y=100, width=80, height=30, confidence=0.95),
        block(NAME_AR, x=560, y=98, width=330, height=30, confidence=0.97),
        block("مكان الإصدار", x=900, y=400, width=180, height=30, confidence=0.92),
        block(f"{PLACE_EN}{PLACE_AR}", x=905, y=440, width=170, height=30, confidence=0.94),
    ]


def merged_label_page():
    """Label and value recognised as one box, the separator misread as a letter."""
    return [
        block(f"{LABEL_AR}ا {NAME_AR}", x=560, y=100, width=420, height=30, confidence=0.93),
        block(f"مكان الإصدار {PLACE_AR}", x=800, y=400, width=280, height=30, confidence=0.91),
        block(f"IssuingAuthority/ {PLACE_EN}", x=300, y=402, width=300, height=28, confidence=0.96),
    ]


# ----------------------------------------------------------------- the name
def test_a_name_beside_its_label_is_found():
    found = viz.extract(separate_label_page())

    assert found["name_ar"].value == NAME_AR
    assert found["name_ar"].source == "adjacent"
    assert found["name_ar"].confidence == pytest.approx(0.97)


def test_a_name_sharing_a_box_with_its_label_is_found():
    found = viz.extract(merged_label_page())

    assert found["name_ar"].value == NAME_AR
    assert found["name_ar"].source == "merged"


def test_the_misread_separator_is_not_kept_as_part_of_the_name():
    """A slash read as a lone letter must not survive into the value."""
    found = viz.extract(merged_label_page())

    assert not found["name_ar"].value.startswith("ا ")


# ------------------------------------------------------- issuing authority
def test_a_bilingual_place_is_split_into_both_scripts():
    found = viz.extract(separate_label_page())

    assert found["issuing_authority"].value == PLACE_EN
    assert found["issuing_authority_ar"].value == PLACE_AR


def test_a_place_below_its_label_is_found():
    """The value sits under the label here, not beside it."""
    found = viz.extract(separate_label_page())

    assert found["issuing_authority_ar"].source == "adjacent"


def test_the_latin_and_arabic_forms_are_read_from_their_own_labels():
    found = viz.extract(merged_label_page())

    assert found["issuing_authority"].value == PLACE_EN
    assert found["issuing_authority_ar"].value == PLACE_AR


# -------------------------------------------------------------- never invent
def test_a_page_without_labels_yields_nothing():
    found = viz.extract(
        [
            block("SOME OTHER DOCUMENT", x=100, y=100, width=300, height=30),
            block("نص عربي بلا عنوان", x=100, y=200, width=300, height=30),
        ]
    )

    assert found == {}


def test_a_label_with_no_readable_value_yields_nothing():
    """A label alone is not a field. Nothing nearby is pressed into service."""
    found = viz.extract([block(LABEL_AR, x=900, y=100, width=80, height=30)])

    assert "name_ar" not in found


def test_a_date_next_to_a_label_is_not_taken_as_the_value():
    """Dates are printed all over this part of the page; they are not names."""
    found = viz.extract(
        [
            block("IssuingAuthority/", x=600, y=400, width=200, height=28),
            block("22 Oct 2035", x=300, y=402, width=160, height=28),
        ]
    )

    assert "issuing_authority" not in found


def test_a_neighbouring_label_is_not_taken_as_a_value():
    found = viz.extract(
        [
            block(LABEL_AR, x=900, y=100, width=80, height=30),
            block("تاريخ الميلاد", x=700, y=100, width=140, height=30),
        ]
    )

    assert "name_ar" not in found


def test_a_single_stray_letter_is_not_a_value():
    found = viz.extract(
        [
            block("مكان الإصدار ا", x=800, y=400, width=200, height=30),
        ]
    )

    assert "issuing_authority_ar" not in found


def test_a_value_in_the_wrong_script_is_not_the_arabic_name():
    found = viz.extract(
        [
            block(LABEL_AR, x=900, y=100, width=80, height=30),
            block("SALEM AL HARTHI", x=560, y=98, width=330, height=30),
        ]
    )

    assert "name_ar" not in found


# ----------------------------------------------------------- normalisation
def test_alef_variants_do_not_stop_a_label_matching():
    """OCR moves freely between the alef forms; the label still has to match."""
    found = viz.extract(
        [
            block(f"الأسم {NAME_AR}", x=560, y=100, width=420, height=30),
        ]
    )

    assert found["name_ar"].value == NAME_AR


def test_scripts_split_even_without_a_separator():
    latin, arabic = viz.split_scripts(f"{PLACE_EN}{PLACE_AR}")

    assert latin == PLACE_EN
    assert arabic == PLACE_AR


# -------------------------------------------------------------- gap finding
def test_a_field_whose_label_is_present_but_unread_is_reported_as_pending():
    """This is what tells the pipeline a second, enhanced look is worth taking."""
    pending = viz.missing_labels([block(LABEL_AR, x=900, y=100, width=80, height=30)])

    assert viz.NAME_AR in pending


def test_nothing_is_pending_once_every_field_is_read():
    pending = viz.missing_labels(separate_label_page())

    assert pending == []
