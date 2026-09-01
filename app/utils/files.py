"""Temporary / optional persistent file handling.

The pipeline works entirely in memory - the uploaded bytes are decoded straight
into a numpy array and never touch the disk. A temp file is only materialised
when a caller explicitly asks for it, and it is always removed afterwards.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def random_name(suffix: str = ".bin") -> str:
    """A filename that carries nothing from the client."""
    return f"{secrets.token_hex(16)}{suffix}"


def _shred(path: Path) -> None:
    try:
        if path.exists():
            # Overwrite before unlinking so the passport image is not trivially
            # recoverable from the container filesystem.
            size = path.stat().st_size
            if 0 < size <= 32 * 1024 * 1024:
                with open(path, "r+b") as fh:
                    fh.write(b"\0" * size)
                    fh.flush()
                    os.fsync(fh.fileno())
            path.unlink()
    except OSError as exc:  # pragma: no cover - best effort cleanup
        logger.warning("temp_file_cleanup_failed", extra={"error": str(exc)})


@contextmanager
def temporary_file(data: bytes, mime: Optional[str] = None) -> Iterator[Path]:
    """Write ``data`` to a randomly named temp file and delete it on exit."""
    suffix = _EXT_BY_MIME.get(mime or "", ".bin")
    directory = Path(settings.TEMP_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / random_name(suffix)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        yield path
    finally:
        _shred(path)


def store_upload_if_enabled(data: bytes, mime: Optional[str]) -> Optional[str]:
    """Persist the upload only when STORE_UPLOADS is explicitly turned on."""
    if not settings.STORE_UPLOADS:
        return None
    day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    directory = Path(settings.STORE_UPLOADS_DIR) / day
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / random_name(_EXT_BY_MIME.get(mime or "", ".bin"))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    # The reference, not the content, is what gets logged.
    logger.info("upload_stored", extra={"stored_ref": path.name})
    return str(path)


def purge_temp_dir(max_age_seconds: int = 3600) -> int:
    """Sweep orphaned temp files left behind by a crashed worker."""
    directory = Path(settings.TEMP_DIR)
    if not directory.exists():
        return 0
    import time

    removed = 0
    now = time.time()
    for entry in directory.iterdir():
        try:
            if entry.is_file() and now - entry.stat().st_mtime > max_age_seconds:
                _shred(entry)
                removed += 1
        except OSError:  # pragma: no cover
            continue
    return removed
