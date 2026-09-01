"""The public endpoint, served by the real engine over HTTP."""

from __future__ import annotations

import pytest

from tests.integration.conftest import encode, render_latin, render_mixed

pytestmark = pytest.mark.integration


def post(client, data, auth, **params):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", data, "image/png")},
        headers=auth,
        params=params or None,
    )


def test_endpoint_returns_recognised_text(real_client, auth):
    data = encode(render_latin(["INVOICE 2026-09-01", "TOTAL 1240.00"]))
    response = post(real_client, data, auth)
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert "1240.00" in body["text"]
    assert "2026-09-01" in body["text"]
    assert body["line_count"] >= 2
    assert body["word_count"] >= 4


def test_endpoint_returns_lines_with_geometry_and_confidence(real_client, auth):
    body = post(real_client, encode(render_latin(["GEOMETRY CHECK"])), auth).json()
    line = body["lines"][0]
    assert line["text"]
    assert 0.0 < line["confidence"] <= 1.0
    assert line["bbox"]["width"] > 0 and line["bbox"]["height"] > 0
    assert line["languages"] == ["en"]


def test_endpoint_returns_raw_blocks_with_polygons(real_client, auth):
    body = post(real_client, encode(render_latin(["POLYGON CHECK"])), auth).json()
    assert body["blocks"]
    assert len(body["blocks"][0]["polygon"]) == 4


def test_endpoint_reports_the_preprocessing_it_applied(real_client, auth):
    body = post(real_client, encode(render_latin(["PREPROCESS CHECK"])), auth).json()
    assert "enhance" in body["preprocessing"]["steps"]
    assert body["image"]["original_width"] == 1200


def test_endpoint_reports_timings(real_client, auth):
    body = post(real_client, encode(render_latin(["TIMING CHECK"])), auth).json()
    assert body["processing_time_ms"] > 0
    assert body["timings_ms"]["ocr_ms"] > 0


def test_endpoint_handles_arabic(real_client, auth, arabic_font):
    data = encode(render_mixed([("جمهورية مصر العربية", "rtl")]))
    body = post(real_client, data, auth, languages="arabic").json()
    assert "جمهورية" in body["text"]
    assert body["languages"] == ["arabic"]


def test_endpoint_handles_mixed_arabic_and_english(real_client, auth, arabic_font):
    data = encode(render_mixed([("INVOICE رقم 12345", "ltr")]))
    body = post(real_client, data, auth, languages="arabic").json()
    assert "12345" in body["text"]
    assert "رقم" in body["text"]


def test_endpoint_accepts_a_jpeg(real_client, auth):
    import cv2

    ok, buffer = cv2.imencode(".jpg", render_latin(["JPEG SAMPLE TEXT"]))
    assert ok
    body = post(real_client, buffer.tobytes(), auth).json()
    assert body["image"]["format"] == "image/jpeg"
    assert "JPEG" in body["text"].upper()


def test_min_confidence_filters_real_results(real_client, auth):
    data = encode(render_latin(["FILTER CHECK LINE"]))
    permissive = post(real_client, data, auth, min_confidence=0.0).json()
    strict = post(real_client, data, auth, min_confidence=0.999).json()
    assert len(strict["lines"]) <= len(permissive["lines"])


def test_authentication_still_applies_with_the_real_engine(real_client):
    response = real_client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", encode(render_latin(["AUTH"])), "image/png")},
    )
    assert response.status_code == 401


def test_readiness_reports_the_engine_is_loaded(real_client):
    body = real_client.get("/ready").json()
    assert body["ocr_ready"] is True
    assert body["status"] == "ready"


def test_version_reports_the_engine_actually_loaded(real_client):
    """/ready echoes the configured provider name; /version reports the object
    that is really serving requests."""
    body = real_client.get("/api/v1/version").json()
    assert body["ocr"]["provider"] == "paddleocr"
    assert body["ocr"]["ready"] is True


def test_version_reports_the_loaded_languages(real_client):
    body = real_client.get("/api/v1/version").json()
    assert body["ocr"]["provider"] == "paddleocr"
    assert body["ocr"]["loaded_languages"]
