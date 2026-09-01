"""Shared pytest fixtures.

Every fixture is synthetic: images are drawn with OpenCV at test time and text
is scripted into the stub OCR provider. No real documents and no personal data
appear anywhere in this suite.
"""

from __future__ import annotations

import os

# Configure the service before anything imports the settings singleton.
#
# Settings also reads a .env file from the working directory, and environment
# variables take precedence over it. That precedence is what makes this block
# work: without it, running the suite in a directory that holds a *deployed*
# .env would test that deployment's configuration instead of a known one.
# `scripts/deploy.sh` runs these tests on the server, next to the production
# .env, so the suite has to be hermetic or the deploy gate is meaningless.
os.environ.setdefault("OCR_PROVIDER", "stub")
os.environ.setdefault("OCR_LANGUAGES", "en")
os.environ.setdefault("OCR_API_KEY", "test-key-primary,test-key-secondary")
os.environ.setdefault("OCR_WARMUP_ON_STARTUP", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENVIRONMENT", "test")

# Forced, not defaulted: these decide whether a request is served at all, so a
# deployment value for any of them would fail the suite for reasons that have
# nothing to do with the code under test.
#
# ALLOWED_HOSTS is the one that bites: a real deployment sets it to its own
# domain, TrustedHostMiddleware then rejects TestClient's "testserver" host,
# and every single HTTP test fails with 400 "Invalid host header".
os.environ["ALLOWED_HOSTS"] = "*"
os.environ["ALLOWED_ORIGINS"] = ""
os.environ["TRUST_PROXY_HEADERS"] = "false"
os.environ["DOCS_ENABLED"] = "true"
os.environ["DEBUG"] = "false"
# Never let a test run write uploads to a deployment's storage directory.
os.environ["STORE_UPLOADS"] = "false"

from collections.abc import Iterator, Sequence  # noqa: E402
from typing import List

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.services.ocr.base import TextBlock  # noqa: E402
from app.services.ocr.engine import get_engine, reset_engine  # noqa: E402
from app.services.ocr.stub import StubOCRProvider  # noqa: E402

API_KEY = "test-key-primary"
AUTH = {"X-API-Key": API_KEY}


# --------------------------------------------------------------- image fixtures
def make_image(
    width: int = 1000,
    height: int = 640,
    lines: Sequence[str] = ("SYNTHETIC TEST PAGE", "SECOND LINE OF TEXT"),
    background: int = 245,
) -> np.ndarray:
    """Draw a synthetic document page."""
    image = np.full((height, width, 3), background, np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (40, 90 + index * 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
        )
    return image


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok, "failed to encode the synthetic fixture"
    return buffer.tobytes()


@pytest.fixture
def image_bytes() -> bytes:
    return encode(make_image())


@pytest.fixture
def jpeg_bytes() -> bytes:
    return encode(make_image(), ".jpg")


def block(
    text: str,
    x: float = 40,
    y: float = 40,
    width: float = 200,
    height: float = 24,
    confidence: float = 0.95,
    lang: str = "en",
) -> TextBlock:
    """Build one recognition box with explicit geometry."""
    return TextBlock(
        text=text,
        confidence=confidence,
        polygon=[(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
        lang=lang,
    )


DEFAULT_BLOCKS: List[TextBlock] = [
    block("SYNTHETIC TEST PAGE", 40, 60, 380, 26, 0.96),
    block("SECOND LINE OF TEXT", 40, 130, 360, 26, 0.91),
    block("REFERENCE 12345", 40, 200, 280, 26, 0.84),
]


# ------------------------------------------------------------------ app fixtures
@pytest.fixture
def stub_provider() -> Iterator[StubOCRProvider]:
    provider = StubOCRProvider()
    provider.set_blocks("en", DEFAULT_BLOCKS)
    yield provider


@pytest.fixture
def client(stub_provider: StubOCRProvider) -> Iterator[TestClient]:
    """A TestClient whose engine is backed by the scripted stub provider."""
    reset_engine()
    engine = get_engine()
    engine.set_provider(stub_provider)
    app = create_app()
    with TestClient(app) as test_client:
        test_client.stub = stub_provider  # type: ignore[attr-defined]
        yield test_client
    reset_engine()


@pytest.fixture
def lenient_client(stub_provider: StubOCRProvider) -> Iterator[TestClient]:
    """A client that lets the application's own 500 handler respond.

    ``TestClient`` re-raises unhandled exceptions by default, which would hide
    the exact behaviour these tests exist to verify.
    """
    reset_engine()
    get_engine().set_provider(stub_provider)
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        test_client.stub = stub_provider  # type: ignore[attr-defined]
        yield test_client
    reset_engine()


@pytest.fixture
def auth_headers() -> dict:
    return dict(AUTH)
