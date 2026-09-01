"""OpenCV preprocessing.

Each step is independent and reports what it did, so the response can explain
which transformations were applied. Nothing here is destructive: the enhanced
image is only ever used as an OCR input, never returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessResult:
    image: Any
    steps: List[str] = field(default_factory=list)
    scale: float = 1.0
    rotation: int = 0
    skew_angle: float = 0.0
    perspective_corrected: bool = False

    def to_dict(self) -> dict:
        return {
            "steps": list(self.steps),
            "scale": round(self.scale, 4),
            "rotation": self.rotation,
            "skew_angle": round(self.skew_angle, 3),
            "perspective_corrected": self.perspective_corrected,
        }


# --------------------------------------------------------------------- basics
def to_gray(image: Any) -> Any:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def downscale(image: Any, max_dimension: Optional[int] = None) -> Tuple[Any, float]:
    """Cap the long side. Returns the image and the scale factor applied."""
    limit = max_dimension or settings.IMAGE_MAX_DIMENSION
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= limit:
        return image, 1.0
    scale = limit / float(longest)
    resized = cv2.resize(
        image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def rotate90(image: Any, turns: int) -> Any:
    """Rotate by a multiple of 90 degrees without interpolation loss."""
    turns %= 4
    if turns == 0:
        return image
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ---------------------------------------------------------------------- skew
def estimate_skew(image: Any, max_angle: float = 15.0) -> float:
    """Estimate the page skew in degrees from the dominant text baseline.

    Returns 0.0 when no confident estimate can be made - guessing an angle is
    worse than leaving the page alone.
    """
    gray = to_gray(image)
    gray, _ = downscale(gray, 1000)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    # Smear characters into lines so the estimate follows the text baseline.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    smeared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        smeared, 1, np.pi / 720, threshold=120, minLineLength=gray.shape[1] // 4,
        maxLineGap=30,
    )
    if lines is None or len(lines) < 3:
        return 0.0

    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if abs(angle) <= max_angle:
            angles.append(angle)
    if len(angles) < 3:
        return 0.0
    median = float(np.median(angles))
    return median if abs(median) >= 0.25 else 0.0


def deskew(image: Any, angle: Optional[float] = None) -> Tuple[Any, float]:
    angle = estimate_skew(image) if angle is None else angle
    if abs(angle) < 0.25:
        return image, 0.0
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


# --------------------------------------------------------------- perspective
def _order_quad(points: Any) -> Any:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def find_document_quad(image: Any, min_area_ratio: float = 0.35) -> Optional[Any]:
    """Find the document outline, if one is clearly separable from the background."""
    gray = to_gray(image)
    small, scale = downscale(gray, 900)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    page_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(contour)
        if area < page_area * min_area_ratio:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Reject a quad that is essentially the whole frame: cropping to the
            # image border achieves nothing and risks shaving off content.
            if area > page_area * 0.985:
                return None
            return (approx.astype("float32") / scale).astype("float32")
    return None


def four_point_transform(image: Any, quad: Any) -> Any:
    ordered = _order_quad(quad)
    (tl, tr, br, bl) = ordered
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return image
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def correct_perspective(image: Any) -> Tuple[Any, bool]:
    quad = find_document_quad(image)
    if quad is None:
        return image, False
    warped = four_point_transform(image, quad)
    height, width = warped.shape[:2]
    # A wildly implausible aspect ratio means the contour was not the document.
    if height == 0 or not (0.3 <= width / float(height) <= 3.5):
        return image, False
    return warped, True


# ---------------------------------------------------------------- enhancement
def enhance(image: Any) -> Any:
    """Even out illumination and sharpen text edges for recognition."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(merged, 5, 40, 40)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    return cv2.addWeighted(denoised, 1.4, blurred, -0.4, 0)


def binarize(image: Any) -> Any:
    """High contrast binarisation, used for the MRZ strip."""
    gray = to_gray(image)
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )


def upscale(image: Any, factor: float) -> Any:
    if factor <= 1.0:
        return image
    height, width = image.shape[:2]
    return cv2.resize(
        image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_CUBIC
    )


# ------------------------------------------------------------------ pipeline
def preprocess(image: Any, rotation: int = 0) -> PreprocessResult:
    """Run the standard preprocessing chain over a decoded image."""
    result = PreprocessResult(image=image)

    if rotation:
        result.image = rotate90(result.image, rotation)
        result.rotation = (rotation * 90) % 360
        result.steps.append("rotate")

    if settings.ENABLE_PERSPECTIVE_CORRECTION:
        corrected, applied = correct_perspective(result.image)
        if applied:
            result.image = corrected
            result.perspective_corrected = True
            result.steps.append("perspective")

    result.image, scale = downscale(result.image)
    result.scale = scale
    if scale != 1.0:
        result.steps.append("downscale")

    if settings.ENABLE_DESKEW:
        result.image, angle = deskew(result.image)
        result.skew_angle = angle
        if angle:
            result.steps.append("deskew")

    result.image = enhance(result.image)
    result.steps.append("enhance")
    return result
