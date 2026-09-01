"""The optional bridge from positioned OCR boxes to the parser.

``detector`` is the only module in this package that imports anything from the
OCR layer. Nothing else depends on it, so the parser stays usable on its own -
but the bridge itself still has to work, since it is how the generic pipeline
would eventually feed the parser.
"""

from __future__ import annotations

from app.services.mrz import TD3
from app.services.mrz.detector import (
    detect_and_parse,
    find_candidates,
    group_into_lines,
    score_line,
)
from app.services.ocr.base import TextBlock
from tests.mrz.conftest import ICAO_TD3_SPECIMEN


def block(text: str, y: float, x: float = 10, width: float = 440, confidence: float = 0.95):
    return TextBlock(text, confidence, [(x, y), (x + width, y), (x + width, y + 18), (x, y + 18)], "en")


def page_blocks():
    return [
        block("REPUBLIC OF UTOPIA", 40, 40, 300, 0.97),
        block("Surname / Nom", 130, 40, 200, 0.91),
        block(ICAO_TD3_SPECIMEN[0], 300),
        block(ICAO_TD3_SPECIMEN[1], 322),
    ]


def test_a_zone_is_found_among_page_text():
    result = detect_and_parse(page_blocks())
    assert result.mrz_type == TD3
    assert result.valid is True
    assert result.document_number == "L898902C3"


def test_boxes_split_across_one_line_are_regrouped():
    """A recognition engine may return one zone line as several boxes."""
    blocks = [
        block("REPUBLIC OF UTOPIA", 40, 40, 300, 0.97),
        block(ICAO_TD3_SPECIMEN[0][:19], 300, 10, 200),
        block(ICAO_TD3_SPECIMEN[0][19:], 301, 215, 235),
        block(ICAO_TD3_SPECIMEN[1], 322),
    ]
    result = detect_and_parse(blocks)
    assert result is not None and result.valid is True


def test_a_page_with_no_zone_yields_nothing():
    blocks = [block("INVOICE 2026-09", 40, 40, 200), block("TOTAL 1240.00", 90, 40, 200)]
    assert detect_and_parse(blocks) is None


def test_no_blocks_yields_nothing():
    assert detect_and_parse([]) is None


def test_recognition_confidence_is_carried_through():
    result = detect_and_parse(page_blocks(), ocr_confidence=0.5)
    assert result.ocr_confidence == 0.5


def test_boxes_are_grouped_into_visual_lines():
    lines = group_into_lines([block("A", 40, 10, 50), block("B", 42, 70, 50), block("C", 200, 10, 50)])
    assert len(lines) == 2


def test_candidates_are_ranked_with_the_zone_first():
    candidates = find_candidates(page_blocks())
    assert candidates
    assert candidates[0].lines[-1].startswith("L898902C3")


def test_zone_lines_score_above_ordinary_text():
    _, zone_score = score_line(ICAO_TD3_SPECIMEN[0])
    _, prose_score = score_line("Date of birth / Date de naissance")
    assert zone_score > prose_score


def test_the_bridge_produces_a_structured_document():
    """The intended connection: boxes in, low-level result out, then the public
    parser turns it into the structured per-field document."""
    from app.services.mrz.icao import ICAOMRZParser

    result = detect_and_parse(page_blocks(), ocr_confidence=0.94)
    document = ICAOMRZParser().build_document(result, ocr_confidence=0.94)
    assert document.valid is True
    assert document.value("document_number") == "L898902C3"
    assert document.confidence_of("document_number") > 0.9
