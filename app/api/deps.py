"""Shared FastAPI dependencies: authentication and rate limiting."""

from __future__ import annotations

import hashlib
from typing import Optional

from fastapi import Depends, Header, Request

from app.core.config import settings
from app.core.exceptions import RateLimitedError, UnauthorizedError
from app.core.logging import get_logger
from app.core.ratelimit import get_rate_limiter
from app.core.security import verify_api_key

logger = get_logger(__name__)


def client_ip(request: Request) -> str:
    """Best-effort client address.

    ``X-Forwarded-For`` is only trusted when the deployment says it is behind a
    proxy, otherwise any caller could spoof its way around the rate limiter.
    """
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Validate the API key. Also accepts ``Authorization: Bearer <key>``."""
    candidate = x_api_key
    if not candidate and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            candidate = token.strip()

    if not verify_api_key(candidate):
        # Log the fact, never the presented key.
        logger.warning(
            "auth_failed",
            extra={"client": client_ip(request), "path": request.url.path},
        )
        raise UnauthorizedError()

    if not candidate:
        return "anonymous"
    # An identifier for rate limiting that is not the secret itself.
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]


async def enforce_rate_limit(
    request: Request, api_key_id: str = Depends(require_api_key)
) -> str:
    """Rate limit per API key, falling back to the client address."""
    if not settings.RATE_LIMIT_ENABLED:
        return api_key_id

    key = api_key_id if api_key_id != "anonymous" else f"ip:{client_ip(request)}"
    allowed, retry_after = await get_rate_limiter().hit(key)
    if not allowed:
        logger.warning("rate_limited", extra={"path": request.url.path})
        raise RateLimitedError(
            f"Rate limit of {settings.RATE_LIMIT_REQUESTS} requests per "
            f"{settings.RATE_LIMIT_WINDOW_SECONDS}s exceeded",
            details={"retry_after": retry_after},
        )
    return api_key_id
