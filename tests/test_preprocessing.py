"""OpenCV preprocessing steps."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.image_processing.preprocess import (
    binarize,
    correct_perspective,
    deskew,
    downscale,
    enhance,
    estimate_skew,
    preprocess,
    rotate90,
    to_gray,
    upscale,
)
from tests.conftest import make_image


def skewed(angle: float) -> np.ndarray:
    image = make_image(1200, 800, [f"LINE {i} OF SYNTHETIC TEXT" for i in range(5)])
    matrix = cv2.getRotationMatrix2D((600, 400), angle, 1.0)
    return cv2.warpAffine(image, matrix, (1200, 800), borderValue=(245, 245, 245))


def test_to_gray_is_idempotent():
    gray = to_gray(make_image())
    assert gray.ndim == 2
    assert to_gray(gray).shape == gray.shape


def test_downscale_caps_the_long_side():
    resized, scale = downscale(make_image(4000, 3000), 1000)
    assert max(resized.shape[:2]) == 1000
    assert scale == pytest.approx(0.25)


def test_downscale_leaves_small_images_alone():
    image = make_image(500, 400)
    resized, scale = downscale(image, 1000)
    assert scale == 1.0
    assert resized.shape == image.shape


def test_rotate90_cycles_back_to_the_original():
    image = make_image(400, 300)
    assert rotate90(image, 1).shape[:2] == (400, 300)
    assert np.array_equal(rotate90(image, 4), image)
    assert np.array_equal(rotate90(rotate90(image, 2), 2), image)


@pytest.mark.parametrize("angle", [-6.0, -2.0, 3.0, 7.0])
def test_deskew_reduces_the_residual_skew(angle):
    image = skewed(angle)
    before = abs(estimate_skew(image))
    corrected, applied = deskew(image)
    assert before > 1.0
    assert abs(estimate_skew(corrected)) < 1.0
    assert applied != 0.0


def test_deskew_leaves_a_straight_page_untouched():
    image = make_image(1200, 800, [f"LINE {i}" for i in range(5)])
    corrected, applied = deskew(image)
    assert applied == 0.0
    assert np.array_equal(corrected, image)


def test_estimate_skew_returns_zero_for_a_blank_page():
    assert estimate_skew(np.full((600, 800, 3), 250, np.uint8)) == 0.0


def test_enhance_preserves_the_shape():
    image = make_image()
    assert enhance(image).shape == image.shape


def test_enhance_accepts_a_grayscale_input():
    assert enhance(to_gray(make_image())).ndim == 3


def test_binarize_produces_two_tone_output():
    values = np.unique(binarize(make_image()))
    assert set(values.tolist()) <= {0, 255}


def test_upscale_multiplies_the_dimensions():
    assert upscale(make_image(100, 80), 2.0).shape[:2] == (160, 200)
    image = make_image(100, 80)
    assert np.array_equal(upscale(image, 1.0), image)


def test_perspective_correction_skips_a_flat_scan():
    """A borderless page has no quad to warp, so it must be left alone."""
    image = make_image()
    _, applied = correct_perspective(image)
    assert applied is False


def test_perspective_correction_straightens_a_photographed_page():
    page = make_image(800, 600, ["ALPHA", "BETA", "GAMMA"])
    canvas = np.full((900, 1100, 3), 30, np.uint8)
    source = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    target = np.float32([[160, 90], [980, 60], [1020, 800], [110, 760]])
    warped = cv2.warpPerspective(page, cv2.getPerspectiveTransform(source, target), (1100, 900))
    canvas = np.where(warped > 0, warped, canvas).astype(np.uint8)

    corrected, applied = correct_perspective(canvas)
    assert applied is True
    # The background border is gone, so the result is smaller than the photo.
    assert corrected.shape[0] < canvas.shape[0]


def test_pipeline_reports_every_step_it_applied():
    result = preprocess(skewed(4.0))
    assert "deskew" in result.steps
    assert "enhance" in result.steps
    assert result.skew_angle != 0.0
    assert result.to_dict()["perspective_corrected"] is False


def test_pipeline_records_rotation():
    result = preprocess(make_image(), rotation=1)
    assert result.rotation == 90
    assert "rotate" in result.steps


def test_pipeline_downscales_oversized_input(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MAX_DIMENSION", 600)
    result = preprocess(make_image(2000, 1500))
    assert max(result.image.shape[:2]) == 600
    assert "downscale" in result.steps
