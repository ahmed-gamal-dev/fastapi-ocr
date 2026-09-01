"""Fixed-window rate limiting.

Two backends:
  * in-process (default) - correct for a single worker, approximate across many
  * redis - shared across every worker/container, used when REDIS_URL is set
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    async def hit(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        raise NotImplementedError

    async def close(self) -> None:
        return None


class InMemoryRateLimiter(RateLimiter):
    def __init__(self, limit: int, window: int) -> None:
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        # A plain threading lock, not an asyncio one: the critical section never
        # awaits, and this stays correct no matter which event loop (or thread)
        # the limiter was constructed on.
        self._lock = threading.Lock()

    async def hit(self, key: str) -> Tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry = int(self.window - (now - bucket[0])) + 1
                return False, max(retry, 1)
            bucket.append(now)
            # Opportunistic cleanup so idle keys do not leak memory.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return True, 0


class RedisRateLimiter(RateLimiter):
    def __init__(self, url: str, limit: int, window: int) -> None:
        import redis.asyncio as aioredis  # imported lazily: optional dependency

        self.redis = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
        self.limit = limit
        self.window = window
        self._fallback = InMemoryRateLimiter(limit, window)

    async def hit(self, key: str) -> Tuple[bool, int]:
        bucket = f"rl:{key}:{int(time.time() // self.window)}"
        try:
            pipe = self.redis.pipeline()
            pipe.incr(bucket)
            pipe.expire(bucket, self.window + 1)
            count, _ = await pipe.execute()
        except Exception as exc:  # pragma: no cover - infra failure path
            # Never let a Redis outage take the OCR service down with it.
            logger.warning("rate_limit_redis_unavailable", extra={"error": str(exc)})
            return await self._fallback.hit(key)
        if int(count) > self.limit:
            return False, self.window
        return True, 0

    async def close(self) -> None:
        try:
            await self.redis.aclose()
        except Exception:  # pragma: no cover
            pass


_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        if settings.REDIS_URL:
            try:
                _limiter = RedisRateLimiter(
                    settings.REDIS_URL,
                    settings.RATE_LIMIT_REQUESTS,
                    settings.RATE_LIMIT_WINDOW_SECONDS,
                )
                logger.info("rate_limiter_backend", extra={"backend": "redis"})
            except Exception as exc:
                logger.warning(
                    "rate_limiter_redis_init_failed", extra={"error": str(exc)}
                )
                _limiter = InMemoryRateLimiter(
                    settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS
                )
        else:
            _limiter = InMemoryRateLimiter(
                settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS
            )
    return _limiter


async def close_rate_limiter() -> None:
    global _limiter
    if _limiter is not None:
        await _limiter.close()
        _limiter = None
