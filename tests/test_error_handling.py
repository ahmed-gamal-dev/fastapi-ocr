"""Error envelope, status codes and the no-stack-trace guarantee."""

from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import (
    STATUS_BY_CODE,
    ErrorCode,
    InvalidImageError,
    OCRServiceError,
    OCRTimeoutError,
)


def post(client, image_bytes, headers):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", image_bytes, "image/png")},
        headers=headers,
    )


def test_every_error_code_maps_to_a_status():
    for code in ErrorCode:
        assert STATUS_BY_CODE[code] >= 400


def test_domain_errors_expose_a_stable_payload():
    error = InvalidImageError("bad pixels", details={"hint": "re-scan"})
    assert error.code is ErrorCode.INVALID_IMAGE
    assert error.status_code == 422
    assert error.to_dict() == {
        "code": "INVALID_IMAGE",
        "message": "bad pixels",
        "details": {"hint": "re-scan"},
    }


def test_unknown_code_falls_back_to_500():
    assert OCRServiceError("boom").status_code == 500


def test_error_envelope_shape(client, auth_headers):
    response = client.post(
        "/api/v1/ocr",
        files={"image": ("x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 3000, "image/png")},
        headers=auth_headers,
    )
    body = response.json()
    assert body["success"] is False
    assert set(body["error"]) <= {"code", "message", "details"}
    assert body["request_id"]


def test_unexpected_failures_do_not_leak_internals(
    lenient_client, image_bytes, auth_headers, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail /srv/app/models/weights.bin")

    monkeypatch.setattr("app.api.v1.ocr.run_pipeline", explode)
    monkeypatch.setattr(settings, "DEBUG", False)

    response = post(lenient_client, image_bytes, auth_headers)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "PROCESSING_ERROR"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    assert "weights.bin" not in response.text


def test_debug_mode_surfaces_the_message_for_local_development(
    lenient_client, image_bytes, auth_headers, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("diagnostic detail")

    monkeypatch.setattr("app.api.v1.ocr.run_pipeline", explode)
    monkeypatch.setattr(settings, "DEBUG", True)

    response = post(lenient_client, image_bytes, auth_headers)
    assert "diagnostic detail" in response.json()["error"]["message"]
    # Even in debug mode the traceback stays in the logs.
    assert "Traceback" not in response.text


def test_ocr_timeout_becomes_504(client, image_bytes, auth_headers, monkeypatch):
    async def timeout(*args, **kwargs):
        raise OCRTimeoutError("OCR did not finish within 1s")

    monkeypatch.setattr("app.api.v1.ocr.run_pipeline", timeout)
    response = post(client, image_bytes, auth_headers)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "OCR_TIMEOUT"


def test_engine_failure_reports_ocr_failed(client, image_bytes, auth_headers, monkeypatch):
    from app.core.exceptions import OCRFailedError

    async def fail(*args, **kwargs):
        raise OCRFailedError()

    monkeypatch.setattr("app.api.v1.ocr.run_pipeline", fail)
    response = post(client, image_bytes, auth_headers)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "OCR_FAILED"


def test_unknown_route_returns_the_json_envelope(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["success"] is False


def test_wrong_method_returns_the_json_envelope(client):
    response = client.get("/api/v1/ocr")
    assert response.status_code == 405
    assert response.json()["success"] is False
