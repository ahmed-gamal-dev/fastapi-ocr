"""Health, readiness and version endpoints. No authentication required."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.schemas.common import HealthResponse, ReadinessResponse
from app.services.ocr.engine import get_engine

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Liveness only: the process is up and serving.

    Deliberately does not touch the OCR engine, so a slow model load cannot make
    a healthy process look dead.
    """
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
)
async def ready(response: Response) -> ReadinessResponse:
    """Readiness: the engine is loaded and able to accept work."""
    engine = get_engine()
    is_ready = engine.is_ready()
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if is_ready else "loading",
        ocr_ready=is_ready,
        provider=settings.OCR_PROVIDER,
        languages=list(settings.OCR_LANGUAGES),
    )
