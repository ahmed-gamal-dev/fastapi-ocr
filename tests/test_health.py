"""Liveness, readiness and version endpoints."""

from __future__ import annotations


def test_health_returns_ok_without_authentication(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_require_an_api_key(client):
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 200


def test_ready_reports_the_loaded_engine(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["ocr_ready"] is True
    assert body["provider"] == "stub"


def test_version_exposes_service_and_limits(client):
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert body["name"]
    assert body["version"]
    assert body["ocr"]["provider"] == "stub"
    assert "stub" in body["ocr"]["available_providers"]
    assert body["limits"]["max_upload_size"] > 0


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_incoming_request_id_is_preserved(client):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_security_headers_are_present(client):
    headers = client.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"
