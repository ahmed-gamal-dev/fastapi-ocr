"""Process-wide OCR engine manager.

Models are loaded once at application startup and reused for every request.
Inference itself is blocking C++ work, so it runs in a dedicated thread pool
that is deliberately small: each concurrent inference multiplies peak RSS and
oversubscribes the CPU, which makes every request slower rather than faster.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.exceptions import OCRTimeoutError, ServiceUnavailableError
from app.core.logging import get_logger
from app.services.ocr.base import OCRProvider, OCRResult
from app.services.ocr.registry import create_provider

logger = get_logger(__name__)


class OCREngine:
    """Owns the provider, the worker pool and the concurrency ceiling."""

    def __init__(self, provider: Optional[OCRProvider] = None) -> None:
        self._provider = provider
        self._executor: Optional[ThreadPoolExecutor] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = threading.Lock()
        self._started_at: Optional[float] = None
        self._stats: Dict[str, float] = {"requests": 0, "failures": 0, "total_ms": 0.0}

    # ------------------------------------------------------------- lifecycle
    @property
    def provider(self) -> OCRProvider:
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    self._provider = create_provider()
        return self._provider

    def set_provider(self, provider: OCRProvider) -> None:
        """Swap the engine implementation (used by tests and by any future
        provider hot-swap). Existing in-flight work keeps the old object."""
        with self._lock:
            self._provider = provider

    async def startup(self) -> None:
        concurrency = max(1, settings.OCR_MAX_CONCURRENCY)
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="ocr"
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._started_at = time.time()

        if settings.OCR_WARMUP_ON_STARTUP:
            loop = asyncio.get_running_loop()
            started = time.perf_counter()
            await loop.run_in_executor(
                self._executor, self.provider.warmup, list(settings.OCR_LANGUAGES)
            )
            logger.info(
                "ocr_engine_ready",
                extra={
                    "provider": self.provider.name,
                    "languages": list(settings.OCR_LANGUAGES),
                    "warmup_ms": round((time.perf_counter() - started) * 1000, 1),
                    "concurrency": concurrency,
                },
            )
        else:
            logger.info(
                "ocr_engine_lazy",
                extra={"provider": self.provider.name, "concurrency": concurrency},
            )

    async def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._provider is not None:
            self._provider.close()
        self._semaphore = None

    # -------------------------------------------------------------- querying
    def is_ready(self) -> bool:
        if self._executor is None:
            return False
        if not settings.OCR_WARMUP_ON_STARTUP:
            return True
        try:
            return self.provider.is_ready()
        except Exception:  # pragma: no cover - provider blew up while loading
            return False

    def info(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"concurrency": max(1, settings.OCR_MAX_CONCURRENCY)}
        try:
            data.update(self.provider.info())
        except Exception as exc:  # pragma: no cover
            data.update({"provider": settings.OCR_PROVIDER, "error": str(exc)})
        requests = self._stats["requests"] or 1
        data["stats"] = {
            "requests": int(self._stats["requests"]),
            "failures": int(self._stats["failures"]),
            "avg_ms": round(self._stats["total_ms"] / requests, 1),
            "uptime_seconds": (
                round(time.time() - self._started_at, 1) if self._started_at else 0.0
            ),
        }
        return data

    # ------------------------------------------------------------- inference
    async def recognize(
        self, image: Any, lang: str = "en", timeout: Optional[float] = None
    ) -> OCRResult:
        if self._executor is None or self._semaphore is None:
            raise ServiceUnavailableError("OCR engine has not been started")

        timeout = timeout or settings.OCR_TIMEOUT_SECONDS
        loop = asyncio.get_running_loop()
        started = time.perf_counter()

        # The semaphore bounds queueing; the executor bounds parallelism.
        async with self._semaphore:
            future = loop.run_in_executor(
                self._executor, self.provider.recognize, image, lang
            )
            try:
                result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            except asyncio.TimeoutError as exc:
                self._stats["failures"] += 1
                # The thread cannot be killed; let it finish and drop the result.
                future.add_done_callback(lambda f: f.exception())
                raise OCRTimeoutError(
                    f"OCR did not finish within {timeout:.0f}s"
                ) from exc
            except Exception:
                self._stats["failures"] += 1
                raise

        self._stats["requests"] += 1
        self._stats["total_ms"] += (time.perf_counter() - started) * 1000
        return result

    async def recognize_many(
        self, image: Any, languages: Sequence[str], timeout: Optional[float] = None
    ) -> List[OCRResult]:
        """Run several language models over the same image, sequentially.

        Sequential on purpose: the models share the CPU, and running them in
        parallel inside one worker only increases latency and peak memory.
        """
        results: List[OCRResult] = []
        for lang in languages:
            try:
                results.append(await self.recognize(image, lang, timeout=timeout))
            except OCRTimeoutError:
                raise
            except Exception as exc:
                logger.warning(
                    "ocr_language_failed", extra={"lang": lang, "error": str(exc)}
                )
        return results


_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine


def reset_engine() -> None:
    """Drop the singleton. Used between test cases."""
    global _engine
    _engine = None
