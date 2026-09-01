"""Logging must never carry document content or credentials."""

from __future__ import annotations

import json
import logging

import pytest

from app.core.config import settings
from app.core.logging import JsonFormatter, scrub


def format_record(**extra) -> dict:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=extra.pop("msg", "event"), args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_sensitive_keys_are_masked():
    payload = format_record(
        document_number="X1234567", surname="A NAME", date_of_birth="1990-01-01"
    )
    assert payload["document_number"] == "[REDACTED]"
    assert payload["surname"] == "[REDACTED]"
    assert payload["date_of_birth"] == "[REDACTED]"


def test_reserved_log_keys_cannot_carry_data_at_all():
    """``name`` and ``filename`` are LogRecord attributes: the logging module
    refuses them in ``extra``, so they can only ever appear nested."""
    with pytest.raises(KeyError):
        logging.getLogger("test").makeRecord(
            "test", logging.INFO, __file__, 1, "m", (), None,
            extra={"filename": "leak.png"},
        )


def test_credentials_are_masked():
    payload = format_record(api_key="super-secret", authorization="Bearer super-secret")
    assert "super-secret" not in json.dumps(payload)


def test_client_filenames_are_masked_when_nested():
    payload = format_record(upload={"filename": "my-document.png", "bytes": 4096})
    assert payload["upload"]["filename"] == "[REDACTED]"
    assert payload["upload"]["bytes"] == 4096


def test_image_bytes_are_never_logged():
    payload = format_record(image=b"\x89PNG" + b"\x00" * 4096)
    assert payload["image"] == "[REDACTED]"


def test_raw_bytes_are_summarised_not_dumped():
    assert scrub({"blob": b"0123456789"})["blob"] == "<10 bytes>"


def test_long_uppercase_runs_are_scrubbed_from_messages():
    """Defence in depth against machine-readable text leaking through a message."""
    payload = format_record(msg="line ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<<<<< seen")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in payload["message"]
    assert "[REDACTED]" in payload["message"]


def test_nested_structures_are_scrubbed():
    scrubbed = scrub({"outer": {"inner": [{"name": "A NAME", "count": 3}]}})
    assert scrubbed["outer"]["inner"][0]["name"] == "[REDACTED]"
    assert scrubbed["outer"]["inner"][0]["count"] == 3


def test_safe_operational_fields_survive():
    payload = format_record(blocks=12, duration_ms=88.5, mean_confidence=0.94)
    assert payload["blocks"] == 12
    assert payload["duration_ms"] == 88.5
    assert payload["mean_confidence"] == 0.94


def test_records_carry_service_metadata():
    payload = format_record()
    assert payload["level"] == "INFO"
    assert payload["service"] == settings.APP_NAME
    assert "timestamp" in payload


def test_debug_mode_can_unmask_but_is_off_by_default(monkeypatch):
    assert settings.LOG_SENSITIVE_DATA is False
    monkeypatch.setattr(settings, "LOG_SENSITIVE_DATA", True)
    assert scrub({"name": "A NAME"})["name"] == "A NAME"


def test_recognised_text_is_not_logged_by_the_pipeline(client, image_bytes, auth_headers, caplog):
    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/api/v1/ocr",
            files={"image": ("page.png", image_bytes, "image/png")},
            headers=auth_headers,
        )
    assert response.status_code == 200
    assert "SYNTHETIC TEST PAGE" in response.json()["text"]
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "SYNTHETIC TEST PAGE" not in logged
    assert "page.png" not in logged
