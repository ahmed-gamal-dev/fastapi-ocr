"""Upload validation and decoding.

Bytes go straight from the request into a numpy array; nothing is written to
disk. Validation happens before decoding wherever possible so a malformed or
oversized payload is rejected without allocating for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    ImageTooLargeError,
    ImageTooSmallError,
    InvalidImageError,
    UnsupportedFormatError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# Magic numbers, checked instead of trusting the client's Content-Type.
_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


@dataclass
class LoadedImage:
    image: Any  # np.ndarray, BGR
    detected_mime: str
    width: int
    height: int
    size_bytes: int

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def sniff_mime(data: bytes) -> Optional[str]:
    """Identify the image type from its content."""
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_extension(filename: Optional[str]) -> None:
    """Reject obviously wrong extensions.

    The filename is never used to name anything on disk - this check exists
    purely to fail fast with a clear message.
    """
    if not filename:
        return
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Extension '{ext}' is not supported. Allowed: "
            f"{', '.join(settings.ALLOWED_EXTENSIONS)}"
        )


def validate_size(data: bytes) -> None:
    if len(data) > settings.MAX_UPLOAD_SIZE:
        raise ImageTooLargeError(
            f"Image is {len(data)} bytes; the limit is {settings.MAX_UPLOAD_SIZE} bytes"
        )
    if len(data) < settings.MIN_UPLOAD_SIZE:
        raise ImageTooSmallError(
            f"Image is only {len(data)} bytes and cannot be a usable document scan"
        )


def load_image(data: bytes, filename: Optional[str] = None) -> LoadedImage:
    """Validate and decode an uploaded image into a BGR array."""
    if not data:
        raise InvalidImageError("The uploaded file is empty")

    validate_size(data)
    validate_extension(filename)

    mime = sniff_mime(data)
    if mime is None:
        raise UnsupportedFormatError(
            "The uploaded file is not a recognised image (expected JPEG, PNG, "
            "WebP, BMP or TIFF)"
        )
    if mime not in settings.ALLOWED_MIME_TYPES:
        raise UnsupportedFormatError(f"Image type '{mime}' is not allowed")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The image could not be decoded")

    height, width = image.shape[:2]
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(
            f"Image has {width * height} pixels; the limit is "
            f"{settings.MAX_IMAGE_PIXELS}"
        )
    if min(width, height) < settings.IMAGE_MIN_DIMENSION:
        raise ImageTooSmallError(
            f"Image is {width}x{height}; the shorter side must be at least "
            f"{settings.IMAGE_MIN_DIMENSION}px for reliable recognition"
        )

    # Log shape only - never the bytes, never the client filename.
    logger.debug(
        "image_loaded",
        extra={"mime": mime, "width": width, "height": height, "bytes": len(data)},
    )
    return LoadedImage(image, mime, width, height, len(data))
