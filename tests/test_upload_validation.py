"""Upload validation: size, type and content checks."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.image_processing.loader import load_image, sniff_mime
from tests.conftest import encode, make_image


def post(client, data, headers, filename="page.png", mime="image/png"):
    return client.post(
        "/api/v1/ocr",
        files={"image": (filename, data, mime)},
        headers=headers,
    )


def test_missing_file_is_rejected(client, auth_headers):
    response = client.post("/api/v1/ocr", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MISSING_FILE"


def test_non_image_payload_is_rejected(client, auth_headers):
    response = post(client, b"%PDF-1.7\n" + b"x" * 4000, auth_headers, "doc.pdf", "application/pdf")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_corrupt_image_is_rejected(client, auth_headers):
    """A valid PNG header followed by garbage must not decode."""
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000
    response = post(client, payload, auth_headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_empty_upload_is_rejected(client, auth_headers):
    response = post(client, b"", auth_headers)
    assert response.status_code in (413, 422)
    assert response.json()["success"] is False


def test_extension_mismatch_is_rejected(client, image_bytes, auth_headers):
    response = post(client, image_bytes, auth_headers, "page.exe", "image/png")
    assert response.status_code == 415


def test_oversized_upload_is_rejected(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE", 2048)
    response = post(client, encode(make_image(1200, 800)), auth_headers)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_tiny_image_is_rejected(client, auth_headers):
    response = post(client, encode(make_image(64, 48, lines=())), auth_headers)
    assert response.status_code in (413, 422)
    assert response.json()["error"]["code"] in ("IMAGE_TOO_SMALL", "IMAGE_TOO_LARGE")


def test_content_type_header_is_not_trusted(client, auth_headers):
    """The declared MIME type is ignored; only the file content decides."""
    response = post(client, encode(make_image()), auth_headers, "page.png", "text/plain")
    assert response.status_code == 200


# ------------------------------------------------------------------ unit level
def test_sniff_mime_identifies_formats():
    assert sniff_mime(encode(make_image(), ".png")) == "image/png"
    assert sniff_mime(encode(make_image(), ".jpg")) == "image/jpeg"
    assert sniff_mime(b"not an image at all") is None


def test_load_image_returns_dimensions():
    loaded = load_image(encode(make_image(800, 600)), "page.png")
    assert (loaded.width, loaded.height) == (800, 600)
    assert loaded.detected_mime == "image/png"
    assert loaded.megapixels == pytest.approx(0.48)


def test_load_image_enforces_the_pixel_ceiling(monkeypatch):
    from app.core.exceptions import ImageTooLargeError

    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ImageTooLargeError):
        load_image(encode(make_image(800, 600)), "page.png")
