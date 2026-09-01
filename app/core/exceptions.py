"""Domain error codes and the exception hierarchy exposed through the API.

Error codes are part of the public contract consumed by the Laravel client:
never rename one without versioning the API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    INVALID_IMAGE = "INVALID_IMAGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MISSING_FILE = "MISSING_FILE"
    OCR_FAILED = "OCR_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"
    MRZ_NOT_FOUND = "MRZ_NOT_FOUND"
    MRZ_INVALID = "MRZ_INVALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_DATA_EXTRACTED = "NO_DATA_EXTRACTED"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


# HTTP status per error code. Anything not listed falls back to 500.
STATUS_BY_CODE: Dict[ErrorCode, int] = {
    ErrorCode.INVALID_IMAGE: 422,
    ErrorCode.IMAGE_TOO_LARGE: 413,
    ErrorCode.IMAGE_TOO_SMALL: 422,
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.MISSING_FILE: 422,
    ErrorCode.OCR_FAILED: 500,
    ErrorCode.OCR_TIMEOUT: 504,
    ErrorCode.MRZ_NOT_FOUND: 422,
    ErrorCode.MRZ_INVALID: 422,
    ErrorCode.LOW_CONFIDENCE: 422,
    ErrorCode.NO_DATA_EXTRACTED: 422,
    ErrorCode.PROCESSING_ERROR: 500,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
}


class OCRServiceError(Exception):
    """Base class for every error that is safe to surface to the caller."""

    code: ErrorCode = ErrorCode.PROCESSING_ERROR
    message: str = "Processing failed"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.message
        if code is not None:
            self.code = code
        self.details = details or {}
        super().__init__(self.message)

    @property
    def status_code(self) -> int:
        return STATUS_BY_CODE.get(self.code, 500)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class InvalidImageError(OCRServiceError):
    code = ErrorCode.INVALID_IMAGE
    message = "The uploaded file could not be decoded as an image"


class ImageTooLargeError(OCRServiceError):
    code = ErrorCode.IMAGE_TOO_LARGE
    message = "The uploaded image exceeds the maximum allowed size"


class ImageTooSmallError(OCRServiceError):
    code = ErrorCode.IMAGE_TOO_SMALL
    message = "The uploaded image is too small to be processed reliably"


class UnsupportedFormatError(OCRServiceError):
    code = ErrorCode.UNSUPPORTED_FORMAT
    message = "Unsupported image format"


class MissingFileError(OCRServiceError):
    code = ErrorCode.MISSING_FILE
    message = "No image file was provided"


class OCRFailedError(OCRServiceError):
    code = ErrorCode.OCR_FAILED
    message = "The OCR engine failed to process the image"


class OCRTimeoutError(OCRServiceError):
    code = ErrorCode.OCR_TIMEOUT
    message = "OCR processing timed out"


class MRZNotFoundError(OCRServiceError):
    code = ErrorCode.MRZ_NOT_FOUND
    message = "Could not detect passport MRZ"


class MRZInvalidError(OCRServiceError):
    code = ErrorCode.MRZ_INVALID
    message = "The detected MRZ failed structural validation"


class NoDataExtractedError(OCRServiceError):
    code = ErrorCode.NO_DATA_EXTRACTED
    message = "No passport data could be extracted from the image"


class UnauthorizedError(OCRServiceError):
    code = ErrorCode.UNAUTHORIZED
    message = "Invalid or missing API key"


class RateLimitedError(OCRServiceError):
    code = ErrorCode.RATE_LIMITED
    message = "Rate limit exceeded"


class ServiceUnavailableError(OCRServiceError):
    code = ErrorCode.SERVICE_UNAVAILABLE
    message = "OCR engine is not ready"
