"""The generic OCR endpoint contract."""

from __future__ import annotations

import pytest

from tests.conftest import block


def post(client, data, headers, filename="page.png", mime="image/png", **params):
    return client.post(
        "/api/v1/ocr",
        files={"image": (filename, data, mime)},
        headers=headers,
        params=params or None,
    )


def test_successful_response_shape(client, image_bytes, auth_headers):
    response = post(client, image_bytes, auth_headers)
    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    assert body["request_id"]
    assert "SYNTHETIC TEST PAGE" in body["text"]
    assert body["line_count"] == 3
    assert body["word_count"] > 0
    assert body["languages"] == ["en"]
    assert 0.0 < body["confidence"]["mean"] <= 1.0
    assert body["confidence"]["min"] <= body["confidence"]["mean"] <= body["confidence"]["max"]
    assert body["processing_time_ms"] >= 0


def test_lines_carry_geometry_and_confidence(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers).json()
    first = body["lines"][0]
    assert first["text"] == "SYNTHETIC TEST PAGE"
    assert 0.0 <= first["confidence"] <= 1.0
    assert first["languages"] == ["en"]
    for key in ("x", "y", "width", "height"):
        assert key in first["bbox"]
    assert first["bbox"]["width"] > 0


def test_lines_are_returned_in_reading_order(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers).json()
    ys = [line["bbox"]["y"] for line in body["lines"]]
    assert ys == sorted(ys)


def test_image_metadata_is_reported(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers).json()
    assert body["image"]["format"] == "image/png"
    assert body["image"]["original_width"] == 1000
    assert body["image"]["original_height"] == 640
    assert body["image"]["size_bytes"] == len(image_bytes)
    assert body["image"]["processed_width"] > 0


def test_preprocessing_steps_are_reported(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers).json()
    assert "enhance" in body["preprocessing"]["steps"]
    assert body["preprocessing"]["rotation"] in (0, 90, 180, 270)


def test_preprocessing_can_be_disabled(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers, preprocess=False).json()
    assert body["preprocessing"]["steps"] == []


def test_blocks_are_included_by_default(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers).json()
    assert len(body["blocks"]) == 3
    assert body["blocks"][0]["polygon"]


def test_blocks_can_be_omitted(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers, include_blocks=False).json()
    assert "blocks" not in body


def test_regions_are_opt_in(client, image_bytes, auth_headers):
    assert "regions" not in post(client, image_bytes, auth_headers).json()
    body = post(client, image_bytes, auth_headers, include_regions=True).json()
    assert body["regions"] and body["regions"][0]["line_count"] >= 1


def test_min_confidence_filters_blocks(client, image_bytes, auth_headers):
    body = post(client, image_bytes, auth_headers, min_confidence=0.9).json()
    # The 0.84 block is dropped, and the drop is reported rather than hidden.
    assert body["line_count"] == 2
    assert any("min_confidence" in w for w in body["warnings"])


def test_language_selection_is_honoured(client, image_bytes, auth_headers):
    client.stub.set_blocks("arabic", [block("نص عربي", 500, 60, 160, 26, 0.9, "arabic")])
    body = post(client, image_bytes, auth_headers, languages="en,arabic").json()
    assert set(body["languages"]) == {"en", "arabic"}
    assert "نص عربي" in body["text"]


def test_language_aliases_are_normalised(client, image_bytes, auth_headers):
    client.stub.set_blocks("arabic", [block("نص", 500, 60, 80, 26, 0.9, "arabic")])
    body = post(client, image_bytes, auth_headers, languages="ar").json()
    assert body["languages"] == ["arabic"]


def test_empty_recognition_is_reported_not_invented(client, image_bytes, auth_headers):
    client.stub.clear()
    body = post(client, image_bytes, auth_headers).json()
    assert body["success"] is True
    assert body["text"] == ""
    assert body["line_count"] == 0
    assert "no text was recognised in the image" in body["warnings"]


def test_low_confidence_is_flagged(client, image_bytes, auth_headers):
    client.stub.set_blocks("en", [block("faint text", 40, 60, 200, 24, 0.31)])
    body = post(client, image_bytes, auth_headers).json()
    assert any("confidence is low" in w for w in body["warnings"])


def test_jpeg_is_accepted(client, jpeg_bytes, auth_headers):
    response = post(client, jpeg_bytes, auth_headers, filename="page.jpg", mime="image/jpeg")
    assert response.status_code == 200
    assert response.json()["image"]["format"] == "image/jpeg"


def test_response_is_not_cacheable(client, image_bytes, auth_headers):
    response = post(client, image_bytes, auth_headers)
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("value", [-0.5, 1.5])
def test_min_confidence_is_range_checked(client, image_bytes, auth_headers, value):
    response = post(client, image_bytes, auth_headers, min_confidence=value)
    assert response.status_code == 422
