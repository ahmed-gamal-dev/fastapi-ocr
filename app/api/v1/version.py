"""Service version and effective limits."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.common import VersionResponse
from app.services.ocr.engine import get_engine
from app.services.ocr.registry import available_providers

router = APIRouter(tags=["system"])

API_VERSION = "v1"


@router.get("/version", response_model=VersionResponse, summary="Service version")
async def version() -> VersionResponse:
    engine = get_engine()
    info = engine.info()
    return VersionResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        api_version=API_VERSION,
        environment=settings.ENVIRONMENT,
        ocr={
            "provider": info.get("provider", settings.OCR_PROVIDER),
            "ready": info.get("ready", False),
            "languages": list(settings.OCR_LANGUAGES),
            "loaded_languages": info.get("loaded_languages", []),
            "available_providers": list(available_providers()),
            "concurrency": info.get("concurrency", 1),
        },
        limits={
            "max_upload_size": settings.MAX_UPLOAD_SIZE,
            "max_image_pixels": settings.MAX_IMAGE_PIXELS,
            "allowed_mime_types": list(settings.ALLOWED_MIME_TYPES),
            "rate_limit_requests": settings.RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": settings.RATE_LIMIT_WINDOW_SECONDS,
            "ocr_timeout_seconds": settings.OCR_TIMEOUT_SECONDS,
        },
    )
