"""Pydantic request/response models."""

from __future__ import annotations

from app.schemas.common import (
    BoundingBox,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    VersionResponse,
)
from app.schemas.ocr import (
    ConfidenceSummary,
    ImageInfo,
    MRZFieldModel,
    MRZModel,
    OCRResponse,
    PreprocessingInfo,
    TextBlockModel,
    TextLineModel,
    TextRegionModel,
    build_response,
)

__all__ = [
    "BoundingBox",
    "ConfidenceSummary",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ImageInfo",
    "MRZFieldModel",
    "MRZModel",
    "OCRResponse",
    "PreprocessingInfo",
    "ReadinessResponse",
    "TextBlockModel",
    "TextLineModel",
    "TextRegionModel",
    "VersionResponse",
    "build_response",
]
