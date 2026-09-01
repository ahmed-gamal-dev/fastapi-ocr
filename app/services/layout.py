"""Generic layout analysis over raw recognition boxes.

Engine-agnostic and document-agnostic: it only knows about geometry. Boxes are
regrouped into visual lines, lines are ordered into a reading order, and
adjacent lines are grouped into paragraph-like blocks. Nothing here interprets
what the text means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.services.ocr.base import BBox, TextBlock


@dataclass
class TextLine:
    """One visual line of text assembled from recognition boxes."""

    blocks: List[TextBlock] = field(default_factory=list)
    reading_order: int = 0

    @property
    def text(self) -> str:
        return " ".join(b.text.strip() for b in self.blocks if b.text.strip())

    @property
    def confidence(self) -> float:
        # Weight by text length: a long block being right matters more than a
        # two-character block being right.
        scored = [
            (b.confidence, max(len(b.text.strip()), 1))
            for b in self.blocks
            if b.text.strip()
        ]
        if not scored:
            return 0.0
        total_weight = sum(w for _, w in scored)
        return sum(c * w for c, w in scored) / total_weight

    @property
    def min_confidence(self) -> float:
        scored = [b.confidence for b in self.blocks if b.text.strip()]
        return min(scored) if scored else 0.0

    @property
    def bbox(self) -> BBox:
        boxes = [b.bbox for b in self.blocks if b.polygon]
        if not boxes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0

    @property
    def center_y(self) -> float:
        _, y0, _, y1 = self.bbox
        return (y0 + y1) / 2.0

    @property
    def languages(self) -> List[str]:
        return sorted({b.lang for b in self.blocks if b.lang})

    def to_dict(self) -> Dict[str, Any]:
        x0, y0, x1, y1 = self.bbox
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "languages": self.languages,
            "bbox": {
                "x": round(x0, 1),
                "y": round(y0, 1),
                "width": round(x1 - x0, 1),
                "height": round(y1 - y0, 1),
            },
        }


@dataclass
class TextRegion:
    """A paragraph-like grouping of consecutive lines."""

    lines: List[TextLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)

    @property
    def confidence(self) -> float:
        scored = [line.confidence for line in self.lines if line.text]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def bbox(self) -> BBox:
        boxes = [line.bbox for line in self.lines]
        if not boxes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def to_dict(self) -> Dict[str, Any]:
        x0, y0, x1, y1 = self.bbox
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "line_count": len(self.lines),
            "bbox": {
                "x": round(x0, 1),
                "y": round(y0, 1),
                "width": round(x1 - x0, 1),
                "height": round(y1 - y0, 1),
            },
        }


def _is_rtl(text: str) -> bool:
    """True when the line is predominantly right-to-left script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    rtl = sum(1 for c in letters if "֐" <= c <= "ࣿ" or "יִ" <= c <= "﷿")
    return rtl / len(letters) > 0.5


def group_lines(
    blocks: Sequence[TextBlock], tolerance: float = 0.6
) -> List[TextLine]:
    """Group boxes that share a baseline into lines, in reading order.

    ``tolerance`` is a fraction of the box height, so the grouping adapts to the
    text size instead of assuming a fixed pixel threshold.
    """
    usable = [b for b in blocks if b.text.strip()]
    if not usable:
        return []

    positioned = [b for b in usable if b.polygon]
    if not positioned:
        # No geometry available: every box becomes its own line, order preserved.
        return [TextLine([b], i) for i, b in enumerate(usable)]

    positioned.sort(key=lambda b: (b.center[1], b.x_min))
    groups: List[List[TextBlock]] = []
    for block in positioned:
        placed = False
        for group in groups:
            reference_y = sum(b.center[1] for b in group) / len(group)
            reference_h = max(max(b.height for b in group), block.height, 1.0)
            if abs(block.center[1] - reference_y) <= tolerance * reference_h:
                group.append(block)
                placed = True
                break
        if not placed:
            groups.append([block])

    lines: List[TextLine] = []
    for group in groups:
        text_sample = "".join(b.text for b in group)
        # Right-to-left lines read from the rightmost box inwards.
        group.sort(key=lambda b: b.x_min, reverse=_is_rtl(text_sample))
        lines.append(TextLine(list(group)))

    lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    for index, line in enumerate(lines):
        line.reading_order = index
    return lines


def group_regions(lines: Sequence[TextLine], gap_ratio: float = 1.2) -> List[TextRegion]:
    """Group consecutive lines separated by less than ``gap_ratio`` line heights."""
    if not lines:
        return []
    regions: List[TextRegion] = [TextRegion([lines[0]])]
    for line in lines[1:]:
        previous = regions[-1].lines[-1]
        gap = line.bbox[1] - previous.bbox[3]
        reference = max(previous.height, line.height, 1.0)
        horizontal_overlap = min(previous.bbox[2], line.bbox[2]) - max(
            previous.bbox[0], line.bbox[0]
        )
        if gap <= gap_ratio * reference and horizontal_overlap > 0:
            regions[-1].lines.append(line)
        else:
            regions.append(TextRegion([line]))
    return regions


def full_text(lines: Sequence[TextLine]) -> str:
    return "\n".join(line.text for line in lines if line.text)


def deduplicate_blocks(
    blocks: Sequence[TextBlock], iou_threshold: float = 0.6
) -> List[TextBlock]:
    """Drop boxes from different language passes that cover the same text.

    When two passes recognise the same region, the higher-confidence result is
    kept. Boxes without geometry are always kept - they cannot be compared.
    """
    ordered = sorted(blocks, key=lambda b: -b.confidence)
    kept: List[TextBlock] = []
    for block in ordered:
        if not block.polygon:
            kept.append(block)
            continue
        if any(
            other.polygon and _iou(block.bbox, other.bbox) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(block)
    return kept


def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def confidence_summary(lines: Sequence[TextLine]) -> Dict[str, float]:
    scored = [line.confidence for line in lines if line.text]
    if not scored:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(scored) / len(scored), 4),
        "min": round(min(scored), 4),
        "max": round(max(scored), 4),
    }
