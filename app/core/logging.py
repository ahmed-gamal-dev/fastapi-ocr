"""Structured JSON logging with hard PII redaction.

Rule enforced here: passport numbers, names, dates of birth, MRZ strings and
image bytes must never reach the log stream unless ``LOG_SENSITIVE_DATA`` is
explicitly enabled (development only).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core.config import settings

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

REDACTED = "[REDACTED]"

# Keys whose values are always masked in structured log extras.
SENSITIVE_KEYS = {
    "passport_number",
    "document_number",
    "name",
    "name_ar",
    "name_en",
    "surname",
    "given_names",
    "date_of_birth",
    "dob",
    "date_of_expiry",
    "date_of_issue",
    "place_of_birth",
    "place_of_issue",
    "mrz",
    "mrz_raw",
    "raw_ocr",
    "personal_number",
    "optional_data",
    "image",
    "image_bytes",
    "filename",
    "api_key",
    "authorization",
    "x-api-key",
}

# Belt and braces: strip anything that looks like an MRZ line out of messages.
_MRZ_PATTERN = re.compile(r"[A-Z0-9<]{25,}")


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return REDACTED
    return REDACTED


def scrub(data: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive keys/values from a log payload."""
    if settings.LOG_SENSITIVE_DATA:
        return data
    if _depth > 6:
        return REDACTED
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for key, value in data.items():
            if str(key).lower() in SENSITIVE_KEYS:
                out[key] = _mask_value(value)
            else:
                out[key] = scrub(value, _depth + 1)
        return out
    if isinstance(data, (list, tuple)):
        return [scrub(item, _depth + 1) for item in data]
    if isinstance(data, (bytes, bytearray, memoryview)):
        return f"<{len(data)} bytes>"
    if isinstance(data, str):
        return _MRZ_PATTERN.sub(REDACTED, data)
    return data


_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage()),
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.ENVIRONMENT,
        }
        rid = request_id_ctx.get()
        if rid:
            payload["request_id"] = rid

        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if extras:
            payload.update(scrub(extras))

        if record.exc_info:
            # Stack traces stay in the logs, never in the HTTP response.
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get()
        prefix = f"[{rid[:8]}] " if rid else ""
        base = (
            f"{datetime.fromtimestamp(record.created).isoformat(timespec='milliseconds')} "
            f"{record.levelname:<8} {record.name}: {prefix}{scrub(record.getMessage())}"
        )
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if extras:
            base += f" | {json.dumps(scrub(extras), ensure_ascii=False, default=str)}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.LOG_FORMAT.lower() == "json" else ConsoleFormatter()
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    # Uvicorn duplicates records through its own handlers otherwise.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Paddle is extremely chatty at INFO.
    for name in ("ppocr", "paddle", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if settings.LOG_SENSITIVE_DATA:
        logging.getLogger(__name__).warning(
            "LOG_SENSITIVE_DATA is enabled - PII will appear in logs. "
            "This must never be used in production."
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex
