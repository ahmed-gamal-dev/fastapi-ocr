"""Request-scoped middleware: correlation id, timing and an upload size guard."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.core.exceptions import ErrorCode
from app.core.logging import get_logger, new_request_id, request_id_ctx

logger = get_logger(__name__)

# Multipart framing adds a few hundred bytes around the file itself.
_MULTIPART_OVERHEAD = 8192


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id to the log context and to the response headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("x-request-id")
        request_id = incoming if incoming and len(incoming) <= 64 else new_request_id()
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration = (time.perf_counter() - started) * 1000
            # Path and status only. Query strings and bodies are never logged.
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration, 1),
                },
            )
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized uploads from the Content-Length header.

    This is the cheap first line of defence; the endpoint still counts bytes as
    it streams, because Content-Length can lie or be absent.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            limit = settings.MAX_UPLOAD_SIZE + _MULTIPART_OVERHEAD
            if int(content_length) > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "error": {
                            "code": ErrorCode.IMAGE_TOO_LARGE.value,
                            "message": (
                                "Request body exceeds the maximum upload size of "
                                f"{settings.MAX_UPLOAD_SIZE} bytes"
                            ),
                        },
                        "request_id": request_id_ctx.get(),
                    },
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        # Responses can contain recognised document text: never let a shared
        # cache keep a copy.
        response.headers.setdefault("Cache-Control", "no-store")
        return response
