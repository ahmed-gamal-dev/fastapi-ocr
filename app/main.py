"""FastAPI application factory and entrypoint.

Run locally with::

    ./scripts/run.sh
    # or
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import (
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.api.v1 import system
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.ratelimit import close_rate_limiter
from app.services.ocr.engine import get_engine
from app.utils.files import purge_temp_dir

logger = get_logger(__name__)

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OCR models once, here - never per request."""
    configure_logging()
    logger.info(
        "service_starting",
        extra={
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "provider": settings.OCR_PROVIDER,
            "auth_enabled": settings.auth_enabled,
        },
    )
    if not settings.auth_enabled:
        logger.warning(
            "OCR_API_KEY is not set - the API is running unauthenticated. "
            "Set OCR_API_KEY before exposing this service to anything."
        )

    purged = purge_temp_dir()
    if purged:
        logger.info("temp_files_purged", extra={"count": purged})

    engine = get_engine()
    await engine.startup()
    try:
        yield
    finally:
        logger.info("service_stopping")
        await engine.shutdown()
        await close_rate_limiter()
        purge_temp_dir(max_age_seconds=0)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document OCR Service",
        description=(
            "A local, open-source OCR microservice. Upload an image, get back "
            "structured text with per-line geometry and confidence scores."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        root_path=settings.ROOT_PATH,
        docs_url="/docs" if settings.DOCS_ENABLED else None,
        redoc_url="/redoc" if settings.DOCS_ENABLED else None,
        openapi_url="/openapi.json" if settings.DOCS_ENABLED else None,
    )

    # Middleware runs bottom-up: the request context is outermost so every
    # other layer, including error handlers, can see the request id.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MaxBodySizeMiddleware)
    app.add_middleware(RequestContextMiddleware)

    if settings.ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", settings.API_KEY_HEADER, "Authorization"],
            max_age=600,
        )
    if settings.ALLOWED_HOSTS and settings.ALLOWED_HOSTS != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    register_exception_handlers(app)

    # /health and /ready sit at the root so probes do not need the API prefix.
    app.include_router(system.router)
    app.include_router(api_router, prefix=API_V1_PREFIX)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if settings.WORKERS > 1 else None,
        log_config=None,  # the app configures logging itself
    )
