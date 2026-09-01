"""Rate limiting behaviour."""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.ratelimit import InMemoryRateLimiter


def post(client, image_bytes, headers):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", image_bytes, "image/png")},
        headers=headers,
    )


def test_requests_are_limited_per_key(client, image_bytes, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
    # Rebuild the limiter so it picks up the patched limits.
    import app.core.ratelimit as ratelimit

    monkeypatch.setattr(ratelimit, "_limiter", InMemoryRateLimiter(3, 60))

    statuses = [post(client, image_bytes, auth_headers).status_code for _ in range(4)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_rate_limited_response_has_retry_after(client, image_bytes, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    import app.core.ratelimit as ratelimit

    monkeypatch.setattr(ratelimit, "_limiter", InMemoryRateLimiter(1, 60))

    post(client, image_bytes, auth_headers)
    response = post(client, image_bytes, auth_headers)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert int(response.headers["Retry-After"]) > 0


def test_limiter_allows_up_to_the_limit():
    async def run():
        limiter = InMemoryRateLimiter(limit=2, window=60)
        return [await limiter.hit("k") for _ in range(3)]

    results = asyncio.run(run())
    assert [allowed for allowed, _ in results] == [True, True, False]


def test_limiter_keys_are_independent():
    async def run():
        limiter = InMemoryRateLimiter(limit=1, window=60)
        return await limiter.hit("a"), await limiter.hit("b")

    first, second = asyncio.run(run())
    assert first[0] is True and second[0] is True


def test_limiter_window_expires():
    async def run():
        limiter = InMemoryRateLimiter(limit=1, window=0)
        await limiter.hit("k")
        return await limiter.hit("k")

    allowed, _ = asyncio.run(run())
    assert allowed is True
