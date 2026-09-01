"""API key authentication."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.core.security import verify_api_key


def post(client, image_bytes, headers=None):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", image_bytes, "image/png")},
        headers=headers or {},
    )


def test_request_without_a_key_is_rejected(client, image_bytes):
    response = post(client, image_bytes)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert response.json()["success"] is False


def test_request_with_a_wrong_key_is_rejected(client, image_bytes):
    response = post(client, image_bytes, {"X-API-Key": "not-the-key"})
    assert response.status_code == 401


def test_valid_key_is_accepted(client, image_bytes, auth_headers):
    assert post(client, image_bytes, auth_headers).status_code == 200


def test_any_configured_key_is_accepted(client, image_bytes):
    """Multiple keys are supported so they can be rotated without downtime."""
    response = post(client, image_bytes, {"X-API-Key": "test-key-secondary"})
    assert response.status_code == 200


def test_bearer_token_is_accepted(client, image_bytes):
    response = post(client, image_bytes, {"Authorization": "Bearer test-key-primary"})
    assert response.status_code == 200


def test_unauthorized_response_advertises_the_scheme(client, image_bytes):
    response = post(client, image_bytes)
    assert "ApiKey" in response.headers["WWW-Authenticate"]


def test_the_key_is_never_echoed_back(client, image_bytes):
    response = post(client, image_bytes, {"X-API-Key": "leak-me-please"})
    assert "leak-me-please" not in response.text


@pytest.mark.parametrize("candidate", [None, "", "   ", "wrong"])
def test_verify_api_key_rejects_bad_candidates(candidate):
    assert verify_api_key(candidate) is False


def test_verify_api_key_allows_everything_when_auth_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "OCR_API_KEY", "")
    assert verify_api_key(None) is True
