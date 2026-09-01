"""Generic layout analysis: line grouping, reading order, regions."""

from __future__ import annotations

from app.services.layout import (
    confidence_summary,
    deduplicate_blocks,
    full_text,
    group_lines,
    group_regions,
)
from tests.conftest import block


def test_boxes_on_one_baseline_become_one_line():
    lines = group_lines([block("Hello", 40, 40, 80), block("world", 130, 42, 80)])
    assert len(lines) == 1
    assert lines[0].text == "Hello world"


def test_boxes_on_different_baselines_stay_separate():
    lines = group_lines([block("first", 40, 40), block("second", 40, 120)])
    assert [line.text for line in lines] == ["first", "second"]


def test_lines_are_sorted_top_to_bottom():
    lines = group_lines([block("bottom", 40, 300), block("top", 40, 40)])
    assert [line.text for line in lines] == ["top", "bottom"]
    assert [line.reading_order for line in lines] == [0, 1]


def test_left_to_right_ordering_within_a_line():
    lines = group_lines([block("second", 200, 40), block("first", 40, 40)])
    assert lines[0].text == "first second"


def test_right_to_left_lines_read_from_the_right():
    right = block("العالم", 300, 40, 80, 24, 0.9, "arabic")
    left = block("مرحبا", 400, 40, 80, 24, 0.9, "arabic")
    assert group_lines([right, left])[0].text == "مرحبا العالم"


def test_line_confidence_is_length_weighted():
    line = group_lines(
        [
            block("a", 40, 40, 20, confidence=0.10),
            block("bbbbbbbbbbbbbbbbbbbb", 70, 40, 200, confidence=1.0),
        ]
    )[0]
    # The long, confident block dominates rather than being averaged away.
    assert line.confidence > 0.9
    assert line.min_confidence == 0.10


def test_blocks_without_geometry_still_produce_lines():
    from app.services.ocr.base import TextBlock

    lines = group_lines([TextBlock("one", 0.9), TextBlock("two", 0.8)])
    assert [line.text for line in lines] == ["one", "two"]


def test_empty_input_produces_no_lines():
    assert group_lines([]) == []
    assert group_lines([block("   ", 40, 40)]) == []


def test_regions_group_adjacent_lines():
    blocks = [
        block("para one line one", 40, 40, 300),
        block("para one line two", 40, 72, 300),
        block("far away paragraph", 40, 400, 300),
    ]
    regions = group_regions(group_lines(blocks))
    assert [len(r.lines) for r in regions] == [2, 1]
    assert regions[0].text.splitlines() == ["para one line one", "para one line two"]


def test_full_text_joins_lines_with_newlines():
    text = full_text(group_lines([block("one", 40, 40), block("two", 40, 120)]))
    assert text == "one\ntwo"


def test_confidence_summary_reports_the_spread():
    lines = group_lines(
        [block("high", 40, 40, confidence=0.99), block("low", 40, 120, confidence=0.51)]
    )
    summary = confidence_summary(lines)
    assert summary["max"] == 0.99
    assert summary["min"] == 0.51
    assert summary["mean"] == 0.75


def test_confidence_summary_of_nothing_is_zero():
    assert confidence_summary([]) == {"mean": 0.0, "min": 0.0, "max": 0.0}


def test_overlapping_boxes_are_deduplicated_keeping_the_best():
    kept = deduplicate_blocks(
        [
            block("same text", 40, 40, 200, 24, 0.70, "en"),
            block("same text", 42, 41, 198, 24, 0.95, "arabic"),
            block("elsewhere", 600, 400, 200, 24, 0.90, "en"),
        ]
    )
    assert len(kept) == 2
    assert max(b.confidence for b in kept) == 0.95
    assert 0.70 not in [b.confidence for b in kept]


def test_distinct_boxes_are_not_deduplicated():
    assert len(deduplicate_blocks([block("a", 0, 0), block("b", 500, 500)])) == 2
