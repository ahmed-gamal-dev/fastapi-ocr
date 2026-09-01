"""Exception handlers.

Every failure leaves through here so the client always receives the same JSON
envelope and never receives a stack trace, an internal path or a raw exception
message from an unexpected error.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.exceptions import ErrorCode, OCRServiceError
from app.core.logging import get_logger, request_id_ctx
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)


def error_payload(code: str, message: str, details=None) -> dict:
    body = ErrorResponse(
        error={"code": code, "message": message, "details": details},
        request_id=request_id_ctx.get(),
    )
    return body.model_dump(exclude_none=True)


def _json(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code, content=error_payload(code, message, details)
    )
    if code == ErrorCode.RATE_LIMITED.value and details and details.get("retry_after"):
        response.headers["Retry-After"] = str(details["retry_after"])
    if code == ErrorCode.UNAUTHORIZED.value:
        response.headers["WWW-Authenticate"] = f'ApiKey header="{settings.API_KEY_HEADER}"'
    return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(OCRServiceError)
    async def _domain_error(request: Request, exc: OCRServiceError) -> JSONResponse:
        logger.info(
            "request_rejected",
            extra={
                "code": exc.code.value,
                "path": request.url.path,
                "status": exc.status_code,
            },
        )
        return _json(exc.status_code, exc.code.value, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Report which field failed, never the value that was submitted.
        fields = [".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()]
        missing_file = any("image" in field for field in fields)
        code = (
            ErrorCode.MISSING_FILE.value if missing_file else ErrorCode.INVALID_IMAGE.value
        )
        return _json(
            422,
            code,
            "Request validation failed",
            {"fields": fields} if fields else None,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: ErrorCode.UNAUTHORIZED.value,
            413: ErrorCode.IMAGE_TOO_LARGE.value,
            415: ErrorCode.UNSUPPORTED_FORMAT.value,
            429: ErrorCode.RATE_LIMITED.value,
            503: ErrorCode.SERVICE_UNAVAILABLE.value,
        }.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return _json(exc.status_code, code, detail)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the logs; the caller gets a generic message.
        logger.error(
            "unhandled_exception",
            extra={"path": request.url.path, "type": type(exc).__name__},
            exc_info=exc,
        )
        message = (
            f"{type(exc).__name__}: {exc}"
            if settings.DEBUG
            else "An internal error occurred while processing the request"
        )
        return _json(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.PROCESSING_ERROR.value,
            message,
        )
