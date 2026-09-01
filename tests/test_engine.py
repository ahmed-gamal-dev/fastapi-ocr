"""Engine lifecycle: models load once, concurrency is bounded, timeouts surface."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.config import settings
from app.core.exceptions import OCRTimeoutError, ServiceUnavailableError
from app.services.ocr.base import OCRProvider, OCRResult, TextBlock
from app.services.ocr.engine import OCREngine, get_engine, reset_engine


class CountingProvider(OCRProvider):
    """Records how often it is constructed, warmed up and called."""

    name = "counting"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.warmups = 0
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._ready = False

    def supported_languages(self):
        return ["en"]

    def warmup(self, languages=None):
        self.warmups += 1
        self._ready = True

    def is_ready(self):
        return self._ready

    def recognize(self, image, lang="en"):
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls += 1
            return OCRResult([TextBlock("ok", 1.0, [], lang)], lang, 0.0, self.name)
        finally:
            self._active -= 1


def test_models_are_loaded_once_at_startup_not_per_request():
    provider = CountingProvider()

    async def run():
        engine = OCREngine(provider)
        await engine.startup()
        for _ in range(5):
            await engine.recognize(None, "en")
        await engine.shutdown()

    asyncio.run(run())
    assert provider.warmups == 1  # not once per request
    assert provider.calls == 5


def test_recognition_before_startup_is_refused():
    async def run():
        engine = OCREngine(CountingProvider())
        with pytest.raises(ServiceUnavailableError):
            await engine.recognize(None, "en")

    asyncio.run(run())


def test_concurrency_is_capped_by_the_configured_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "OCR_MAX_CONCURRENCY", 2)
    provider = CountingProvider(delay=0.05)

    async def run():
        engine = OCREngine(provider)
        await engine.startup()
        await asyncio.gather(*(engine.recognize(None, "en") for _ in range(6)))
        await engine.shutdown()

    asyncio.run(run())
    assert provider.calls == 6
    assert provider.max_concurrent <= 2


def test_a_slow_inference_times_out_as_a_domain_error(monkeypatch):
    provider = CountingProvider(delay=0.5)

    async def run():
        engine = OCREngine(provider)
        await engine.startup()
        try:
            with pytest.raises(OCRTimeoutError):
                await engine.recognize(None, "en", timeout=0.05)
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_recognize_many_runs_each_language():
    provider = CountingProvider()

    async def run():
        engine = OCREngine(provider)
        await engine.startup()
        results = await engine.recognize_many(None, ["en", "arabic"])
        await engine.shutdown()
        return results

    results = asyncio.run(run())
    assert [r.lang for r in results] == ["en", "arabic"]


def test_a_failing_language_does_not_abort_the_others():
    class FlakyProvider(CountingProvider):
        def recognize(self, image, lang="en"):
            if lang == "arabic":
                raise RuntimeError("model unavailable")
            return super().recognize(image, lang)

    async def run():
        engine = OCREngine(FlakyProvider())
        await engine.startup()
        results = await engine.recognize_many(None, ["en", "arabic"])
        await engine.shutdown()
        return results

    results = asyncio.run(run())
    assert [r.lang for r in results] == ["en"]


def test_engine_reports_statistics():
    provider = CountingProvider()

    async def run():
        engine = OCREngine(provider)
        await engine.startup()
        await engine.recognize(None, "en")
        info = engine.info()
        await engine.shutdown()
        return info

    info = asyncio.run(run())
    assert info["stats"]["requests"] == 1
    assert info["stats"]["failures"] == 0
    assert info["provider"] == "counting"


def test_engine_singleton_can_be_reset():
    reset_engine()
    first = get_engine()
    assert get_engine() is first
    reset_engine()
    assert get_engine() is not first
