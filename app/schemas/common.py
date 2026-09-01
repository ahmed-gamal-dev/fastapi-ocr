"""Response envelopes shared by every endpoint."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="Left edge, in processed-image pixels")
    y: float = Field(..., description="Top edge, in processed-image pixels")
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["INVALID_IMAGE"])
    message: str = Field(..., examples=["The uploaded file could not be decoded"])
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """The only shape an error ever takes. Never contains a stack trace."""

    success: bool = False
    error: ErrorDetail
    request_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": {
                    "code": "UNSUPPORTED_FORMAT",
                    "message": "Image type 'application/pdf' is not allowed",
                },
                "request_id": "0f9c1f7a1b6e4d1c9a2b3c4d5e6f7a8b",
            }
        }
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class ReadinessResponse(BaseModel):
    status: str = Field(..., examples=["ready"])
    ocr_ready: bool
    provider: str
    languages: list[str] = Field(default_factory=list)


class VersionResponse(BaseModel):
    name: str
    version: str
    api_version: str
    environment: str
    ocr: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)
