"""Locating the machine readable zone band inside a document image.

The MRZ is a dense, high-contrast, very wide block of monospaced glyphs at the
bottom of the document. Blackhat morphology with a wide kernel isolates exactly
that texture. The located strip is cropped, upscaled and contrast-normalised
before it gets its own OCR pass, which is worth far more than trying to read
the MRZ out of a full-page recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.image_processing.preprocess import to_gray, upscale

logger = get_logger(__name__)


@dataclass
class MRZRegion:
    """A candidate MRZ strip, in the coordinate space of the source image."""

    x: int
    y: int
    width: int
    height: int
    score: float
    source: str  # "morphology" | "fallback"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def crop(self, image: Any) -> Any:
        x0, y0, x1, y1 = self.bbox
        return image[max(y0, 0) : y1, max(x0, 0) : x1]

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": round(self.score, 4),
            "source": self.source,
        }


def _expand(
    x: int, y: int, w: int, h: int, shape: Tuple[int, int], pad_x: float, pad_y: float
) -> Tuple[int, int, int, int]:
    height, width = shape
    dx, dy = int(w * pad_x), int(h * pad_y)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(width, x + w + dx)
    y1 = min(height, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def merge_stacked_boxes(
    boxes: List[Tuple[int, int, int, int]],
    max_gap_ratio: float = 2.5,
    min_overlap: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that sit directly above one another into a single band.

    ``max_gap_ratio`` is expressed in multiples of the taller box's height, so
    the tolerance scales with the text size rather than with the image size. It
    is generous because the erosion step shrinks each line box vertically,
    which inflates the apparent gap between two adjacent MRZ lines.
    """
    if len(boxes) < 2:
        return list(boxes)

    ordered = sorted(boxes, key=lambda b: b[1])
    merged: List[List[int]] = [list(ordered[0])]
    for x, y, w, h in ordered[1:]:
        mx, my, mw, mh = merged[-1]
        overlap = min(mx + mw, x + w) - max(mx, x)
        gap = y - (my + mh)
        if (
            overlap > min_overlap * min(mw, w)
            and gap <= max_gap_ratio * max(mh, h)
            and gap > -max(mh, h)
        ):
            nx = min(mx, x)
            ny = min(my, y)
            merged[-1] = [nx, ny, max(mx + mw, x + w) - nx, max(my + mh, y + h) - ny]
            continue
        merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def find_mrz_regions(image: Any, max_regions: int = 3) -> List[MRZRegion]:
    """Return candidate MRZ strips, best first."""
    gray = to_gray(image)
    height, width = gray.shape[:2]
    if height < 60 or width < 120:
        return []

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # A wide, short kernel responds to lines of text, not to individual glyphs.
    kernel_width = max(13, (width // 40) | 1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 5))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    gradient = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gradient = np.absolute(gradient)
    span = gradient.max() - gradient.min()
    if span <= 0:
        return []
    gradient = (255 * (gradient - gradient.min()) / span).astype("uint8")

    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rect_kernel)
    _, threshed = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, square_kernel)
    threshed = cv2.erode(threshed, None, iterations=3)

    # The MRZ runs edge to edge; suppress the outer 4% to drop border artefacts.
    margin = int(width * 0.04)
    threshed[:, :margin] = 0
    threshed[:, width - margin :] = 0

    contours, _ = cv2.findContours(
        threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 8 or w < width * 0.55:
            continue
        # Two MRZ lines are wide and short; a 3-line TD1 zone is squarer.
        if not (3.0 <= w / float(h) <= 60.0):
            continue
        boxes.append((x, y, w, h))

    # Each MRZ line is usually its own contour. They belong to one band, so
    # stack neighbours that overlap horizontally into a single region - the
    # parser needs all the lines of a zone in the same crop.
    boxes = merge_stacked_boxes(boxes)

    regions: List[MRZRegion] = []
    for x, y, w, h in boxes:
        aspect = w / float(h)
        coverage = w / float(width)
        vertical_position = (y + h / 2.0) / height
        score = (
            min(coverage, 1.0) * 0.45
            + min(aspect / 20.0, 1.0) * 0.2
            # The MRZ lives at the bottom of the data page.
            + max(0.0, vertical_position - 0.4) * 0.6
        )
        x, y, w, h = _expand(x, y, w, h, (height, width), 0.02, 0.35)
        regions.append(MRZRegion(x, y, w, h, score, "morphology"))

    regions.sort(key=lambda r: -r.score)
    regions = regions[:max_regions]

    # Always keep a geometric fallback: if morphology missed the band, the
    # bottom strip of the document is still the best place to look.
    fallback_y = int(height * 0.72)
    regions.append(
        MRZRegion(0, fallback_y, width, height - fallback_y, 0.05, "fallback")
    )
    return regions


def prepare_mrz_crop(
    image: Any, region: MRZRegion, target_height: int = 180
) -> Optional[Any]:
    """Crop, upscale and contrast-normalise an MRZ strip for recognition."""
    crop = region.crop(image)
    if crop is None or crop.size == 0:
        return None
    height = crop.shape[0]
    if height < 10:
        return None

    factor = max(settings.MRZ_UPSCALE_FACTOR, target_height / float(height))
    factor = min(factor, 6.0)
    scaled = upscale(crop, factor)

    gray = to_gray(scaled)
    # CLAHE rather than a global stretch: MRZ strips are often unevenly lit.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def region_offset(region: MRZRegion, image: Any, prepared: Any) -> Tuple[float, float, float]:
    """``(dx, dy, scale)`` mapping prepared-crop coordinates back to the source."""
    crop = region.crop(image)
    if crop.size == 0 or prepared is None:
        return (float(region.x), float(region.y), 1.0)
    scale = prepared.shape[0] / float(crop.shape[0]) if crop.shape[0] else 1.0
    return (float(region.x), float(region.y), scale)
