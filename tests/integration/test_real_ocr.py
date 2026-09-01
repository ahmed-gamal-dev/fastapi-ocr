"""End-to-end proof that image -> OpenCV -> PaddleOCR -> structured text works.

Every page here is generated at test time from invented strings.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import encode, joined, render_latin, render_mixed, rotate

pytestmark = pytest.mark.integration


# --------------------------------------------------------------- English
def test_english_text_is_recognised(recognize):
    result = recognize(encode(render_latin(["HELLO WORLD", "SECOND LINE"])))
    text = joined(result).upper()
    assert "HELLO WORLD" in text
    assert "SECOND LINE" in text


def test_recognition_is_confident_on_clean_text(recognize):
    result = recognize(encode(render_latin(["CLEAN SAMPLE TEXT"])))
    assert result.confidence["mean"] > 0.8
    assert result.confidence["min"] > 0.5


def test_every_block_carries_a_confidence_in_range(recognize):
    result = recognize(encode(render_latin(["ALPHA BRAVO", "CHARLIE DELTA"])))
    assert result.blocks
    for block in result.blocks:
        assert 0.0 < block.confidence <= 1.0


# ---------------------------------------------------------------- numbers
def test_digits_are_recognised(recognize):
    result = recognize(encode(render_latin(["1234567890"])))
    assert "1234567890" in joined(result)


def test_decimal_amounts_are_recognised(recognize):
    result = recognize(encode(render_latin(["TOTAL 1240.00 USD"])))
    text = joined(result)
    assert "1240.00" in text
    assert "TOTAL" in text.upper()


def test_iso_dates_are_recognised(recognize):
    result = recognize(encode(render_latin(["2026-09-01", "1990-01-15"])))
    text = joined(result)
    assert "2026-09-01" in text
    assert "1990-01-15" in text


def test_slash_dates_are_recognised(recognize):
    result = recognize(encode(render_latin(["01/09/2026"])))
    assert "01/09/2026" in joined(result).replace(" ", "")


def test_alphanumeric_reference_codes_are_recognised(recognize):
    result = recognize(encode(render_latin(["REF AB1234567"])))
    assert "AB1234567" in joined(result).replace(" ", "")


# ----------------------------------------------------------------- Arabic
def test_arabic_text_is_recognised(recognize, arabic_font):
    image = render_mixed([("مرحبا بالعالم", "rtl")])
    result = recognize(encode(image), languages=["arabic"])
    assert result.lines
    assert "مرحبا" in joined(result)


def test_arabic_lines_read_right_to_left(recognize, arabic_font):
    """A long Arabic line is detected as several boxes. They must be regrouped
    into one line, ordered right to left, not left to right."""
    image = render_mixed([("جمهورية مصر العربية", "rtl")])
    result = recognize(encode(image), languages=["arabic"])
    text = joined(result)
    assert "جمهورية" in text
    # The first word of the line must come before the last one in the output.
    assert text.index("جمهورية") < text.index("العربية")


def test_multiple_arabic_lines_keep_top_to_bottom_order(recognize, arabic_font):
    image = render_mixed(
        [("مرحبا بالعالم", "rtl"), ("جمهورية مصر العربية", "rtl")]
    )
    result = recognize(encode(image), languages=["arabic"])
    assert len(result.lines) >= 2
    assert "مرحبا" in result.lines[0].text
    assert "جمهورية" in result.lines[1].text
    assert [line.reading_order for line in result.lines] == sorted(
        line.reading_order for line in result.lines
    )


def test_arabic_indic_digits_are_recognised(recognize, arabic_font):
    result = recognize(
        encode(render_mixed([("١٢٣٤٥٦٧٨٩٠", "rtl")])), languages=["arabic"]
    )
    assert any(ch in joined(result) for ch in "١٢٣٤٥٦٧٨٩٠")


# ------------------------------------------------------ mixed Arabic/English
def test_mixed_arabic_and_english_on_one_line(recognize, arabic_font):
    image = render_mixed([("INVOICE رقم 12345", "ltr")])
    result = recognize(encode(image), languages=["arabic"])
    text = joined(result)
    assert "INVOICE" in text.upper()
    assert "12345" in text
    assert "رقم" in text


def test_mixed_content_across_several_lines(recognize, arabic_font):
    image = render_mixed(
        [("INVOICE رقم 12345", "ltr"), ("TOTAL المجموع 1240.00", "ltr")]
    )
    result = recognize(encode(image), languages=["arabic"])
    text = joined(result)
    assert "12345" in text and "1240.00" in text
    assert "رقم" in text and "المجموع" in text


def test_running_both_languages_covers_both_scripts(recognize, arabic_font):
    image = render_mixed([("PASSPORT جواز السفر", "ltr"), ("NUMBER 987654", "ltr")])
    result = recognize(encode(image), languages=["en", "arabic"])
    text = joined(result)
    assert "987654" in text
    assert set(result.languages) == {"en", "arabic"}


# ------------------------------------------------------------- geometry
def test_bounding_boxes_are_inside_the_page(recognize):
    image = render_latin(["BOUNDING BOX TEST"])
    result = recognize(encode(image))
    height, width = result.processed_size[1], result.processed_size[0]
    for block in result.blocks:
        x0, y0, x1, y1 = block.bbox
        assert 0 <= x0 < x1 <= width + 1
        assert 0 <= y0 < y1 <= height + 1


def test_bounding_boxes_have_four_corners(recognize):
    result = recognize(encode(render_latin(["POLYGON TEST"])))
    for block in result.blocks:
        assert len(block.polygon) == 4


def test_boxes_track_where_the_text_actually_is(recognize):
    """Text drawn near the top must produce boxes near the top."""
    image = render_latin(["TOP LINE"], line_height=110)
    result = recognize(encode(image))
    top_block = min(result.blocks, key=lambda b: b.bbox[1])
    assert top_block.bbox[1] < result.processed_size[1] * 0.5


def test_reading_order_is_top_to_bottom(recognize):
    result = recognize(encode(render_latin(["FIRST LINE", "SECOND LINE", "THIRD LINE"])))
    ys = [line.bbox[1] for line in result.lines]
    assert ys == sorted(ys)
    assert [line.reading_order for line in result.lines] == list(range(len(result.lines)))


def test_words_on_one_line_are_grouped_together(recognize):
    result = recognize(encode(render_latin(["ONE LINE OF SEVERAL WORDS"])))
    assert len(result.lines) == 1
    assert len(result.lines[0].text.split()) >= 4


# -------------------------------------------------------------- rotation
@pytest.mark.parametrize("angle", [-4.0, 3.0])
def test_slightly_rotated_text_is_still_read(recognize, angle):
    """Deskewing in the preprocessing chain has to earn its place."""
    image = rotate(render_latin(["ROTATED SAMPLE TEXT", "SECOND ROW"]), angle)
    result = recognize(encode(image))
    assert "ROTATED" in joined(result).upper()


def test_deskew_is_reported_for_a_rotated_page(recognize):
    image = rotate(render_latin(["SKEW DETECTION TEST", "ANOTHER ROW", "THIRD ROW"]), 4.0)
    result = recognize(encode(image))
    assert "enhance" in result.preprocessing.steps
    assert result.preprocessing.skew_angle != 0.0


def test_upside_down_text_is_recovered_by_orientation_retry(recognize):
    image = rotate(render_latin(["ORIENTATION TEST PAGE"]), 180)
    result = recognize(encode(image), detect_orientation=True)
    assert "ORIENTATION" in joined(result).upper()


# ------------------------------------------------------------ engine reuse
def test_models_are_loaded_once_and_reused(real_engine):
    """The provider object is created at startup and serves every request; a
    second startup must not build a second set of models."""
    engine, _ = real_engine
    provider = engine.provider
    assert provider.is_ready() is True
    loaded = set(provider.info()["loaded_languages"])
    assert loaded  # at least one language resident

    engines_before = dict(provider._engines)
    provider.warmup(list(loaded))
    # warmup is idempotent: the same model objects, not fresh ones.
    assert {k: id(v) for k, v in provider._engines.items()} == {
        k: id(v) for k, v in engines_before.items()
    }


def test_the_provider_reports_the_installed_engine(real_engine):
    engine, _ = real_engine
    info = engine.provider.info()
    assert info["provider"] == "paddleocr"
    assert info["paddleocr_version"]
    assert info["gpu"] is False


def test_an_empty_page_yields_no_text_rather_than_noise(recognize):
    import numpy as np

    blank = np.full((400, 1200, 3), 245, np.uint8)
    result = recognize(encode(blank))
    assert result.text.strip() == ""
    assert "no text was recognised in the image" in result.warnings
