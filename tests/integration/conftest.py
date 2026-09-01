"""Fixtures for tests that exercise the real PaddleOCR engine.

These tests are skipped automatically when the engine is not installed, so the
unit suite still runs on a bare checkout. Models load once per session.

Every image is generated here at test time from invented strings. There are no
real documents and no personal data: the Arabic samples are country names and
greetings, and the "reference numbers" are keyboard patterns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pytest

# The unit suite's conftest is imported first and already pinned OCR_PROVIDER
# and OCR_API_KEY, so setting them here would be a no-op. These fixtures read
# the effective settings instead, and never mutate them - a session-scoped
# fixture that changed global config would still have it changed while later
# unit tests run.
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")

engine_available = True
try:  # pragma: no cover - depends on the machine
    import paddle  # noqa: F401
    import paddleocr  # noqa: F401
except Exception:  # pragma: no cover
    engine_available = False

pytestmark = pytest.mark.integration

# Whole-package skip: the unit suite must still pass without the engine.
if not engine_available:  # pragma: no cover
    collect_ignore_glob = ["test_*.py"]


BACKGROUND = 245
INK = (15, 15, 15)
#: The loader rejects anything with a short side below 320px, so every
#: generated page clears that by construction.
MIN_SIDE = 360


# --------------------------------------------------------------- rendering
def render_latin(
    lines: Sequence[str],
    width: int = 1200,
    line_height: int = 110,
    scale: float = 1.4,
) -> np.ndarray:
    """Draw Latin text and digits with OpenCV."""
    height = max(MIN_SIDE, 80 + line_height * len(lines))
    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (50, 90 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            INK,
            3,
        )
    return image


#: Fonts that actually contain Arabic glyphs, across macOS and Linux.
ARABIC_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/Amiri-Regular.ttf",
)


def find_arabic_font() -> str:
    for path in ARABIC_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return ""


def _shape(text: str, base_dir: str = "R") -> str:
    """Apply Arabic joining and the bidi algorithm for correct rendering.

    This must run on *every* line containing Arabic, not just right-to-left
    ones: a mixed line like "INVOICE رقم 12345" still needs its Arabic run
    joined and reordered. Skipping it renders isolated glyphs in logical order,
    which no real page looks like - and the engine duly reads such a line
    backwards, which would be the fixture's fault, not the engine's.
    """
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text), base_dir=base_dir)


def render_mixed(
    lines: Sequence[Tuple[str, str]],
    width: int = 1200,
    line_height: int = 115,
    size: int = 56,
) -> np.ndarray:
    """Draw text with PIL, right-aligning lines marked ``rtl``.

    ``lines`` is a sequence of ``(text, "rtl" | "ltr")``.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_arabic_font()
    if not font_path:  # pragma: no cover - depends on the machine
        pytest.skip("no Arabic-capable font available on this machine")

    height = max(MIN_SIDE, 80 + line_height * len(lines))
    image = Image.new("RGB", (width, height), (BACKGROUND,) * 3)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size)

    for index, (text, direction) in enumerate(lines):
        shaped = _shape(text, base_dir="R" if direction == "rtl" else "L")
        text_width = draw.textlength(shaped, font=font)
        x = width - text_width - 60 if direction == "rtl" else 60
        draw.text((x, 40 + line_height * index), shaped, font=font, fill=INK)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok, "failed to encode the generated page"
    return buffer.tobytes()


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, keeping the page background."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderValue=(BACKGROUND,) * 3,
    )


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def arabic_font() -> str:
    path = find_arabic_font()
    if not path:  # pragma: no cover
        pytest.skip("no Arabic-capable font available on this machine")
    try:
        import arabic_reshaper  # noqa: F401
        from bidi.algorithm import get_display  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("arabic-reshaper / python-bidi not installed")
    return path


@pytest.fixture(scope="session")
def real_engine():
    """One engine, models loaded once, shared by every test in the session.

    This is also the assertion that models are reused: the same provider object
    serves every test without reloading.
    """
    import asyncio

    from app.services.ocr.engine import OCREngine
    from app.services.ocr.registry import create_provider

    engine = OCREngine(create_provider("paddle"))
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(engine.startup())
        if not engine.is_ready():  # pragma: no cover
            pytest.skip("the OCR engine failed to load its models")
        yield engine, loop
    finally:
        loop.run_until_complete(engine.shutdown())
        loop.close()


@pytest.fixture
def recognize(real_engine):
    """Run the full pipeline over image bytes and return the result."""
    from app.services.pipeline import PipelineOptions, run_pipeline

    engine, loop = real_engine

    def _run(data: bytes, languages: Sequence[str] = ("en",), **kwargs):
        options = PipelineOptions.build(languages=list(languages), **kwargs)
        return loop.run_until_complete(
            run_pipeline(data, "page.png", options, engine)
        )

    return _run


@pytest.fixture(scope="session")
def real_client(real_engine):
    """A TestClient backed by the real engine, for the HTTP-level tests."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.ocr.engine import get_engine, reset_engine

    engine, _ = real_engine
    # Deliberately no mutation of the global settings here: this fixture is
    # session-scoped, so anything it changed would still be changed while later
    # unit tests run. /api/v1/version already reports the engine that is
    # actually loaded, which is the truthful thing to assert against.
    reset_engine()
    get_engine().set_provider(engine.provider)
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        reset_engine()


@pytest.fixture
def auth() -> dict:
    """Whatever key the running configuration actually accepts."""
    from app.core.config import settings

    if not settings.auth_enabled:
        return {}
    return {"X-API-Key": settings.api_keys[0]}


def texts_of(result) -> List[str]:
    return [line.text for line in result.lines]


def joined(result) -> str:
    return " ".join(texts_of(result))
