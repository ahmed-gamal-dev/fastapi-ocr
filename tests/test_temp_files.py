"""Temporary file handling and the opt-in storage switch."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import settings
from app.utils.files import (
    purge_temp_dir,
    random_name,
    store_upload_if_enabled,
    temporary_file,
)


def test_random_names_carry_nothing_from_the_client():
    name = random_name(".jpg")
    assert re.fullmatch(r"[0-9a-f]{32}\.jpg", name)
    assert random_name() != random_name()


def test_temporary_file_is_removed_after_use(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    with temporary_file(b"payload-bytes", "image/png") as path:
        assert path.exists()
        assert path.read_bytes() == b"payload-bytes"
        assert path.suffix == ".png"
        held = path
    assert not held.exists()
    assert list(tmp_path.iterdir()) == []


def test_temporary_file_is_removed_even_when_the_caller_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    held = None
    with pytest.raises(ValueError):
        with temporary_file(b"payload", "image/jpeg") as path:
            held = path
            raise ValueError("processing failed")
    assert held is not None and not held.exists()


def test_temporary_file_is_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    with temporary_file(b"payload", "image/png") as path:
        assert path.stat().st_mode & 0o077 == 0


def test_uploads_are_not_stored_by_default():
    assert settings.STORE_UPLOADS is False
    assert store_upload_if_enabled(b"payload", "image/png") is None


def test_uploads_are_stored_only_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORE_UPLOADS", True)
    monkeypatch.setattr(settings, "STORE_UPLOADS_DIR", str(tmp_path))
    stored = store_upload_if_enabled(b"payload", "image/png")
    assert stored is not None
    path = Path(stored)
    assert path.read_bytes() == b"payload"
    # The stored name is random: the client's filename is never reused.
    assert re.fullmatch(r"[0-9a-f]{32}\.png", path.name)


def test_purge_removes_stale_files_only(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    stale = tmp_path / "stale.bin"
    fresh = tmp_path / "fresh.bin"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    old_time = time.time() - 7200
    os.utime(stale, (old_time, old_time))

    assert purge_temp_dir(max_age_seconds=3600) == 1
    assert not stale.exists()
    assert fresh.exists()


def test_the_endpoint_leaves_no_temp_files_behind(
    client, image_bytes, auth_headers, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    response = client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", image_bytes, "image/png")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []
