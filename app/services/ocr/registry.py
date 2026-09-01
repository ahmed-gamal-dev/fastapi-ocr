"""Provider registry - the single place where an engine name becomes an object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Dict

from app.core.config import settings
from app.core.exceptions import OCRFailedError
from app.services.ocr.base import OCRProvider

ProviderFactory = Callable[[], OCRProvider]

_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name.strip().lower()] = factory


def available_providers() -> Sequence[str]:
    return sorted(_REGISTRY)


def create_provider(name: str = "") -> OCRProvider:
    key = (name or settings.OCR_PROVIDER).strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise OCRFailedError(
            f"Unknown OCR provider '{key}'. Available: {', '.join(available_providers())}"
        )
    return factory()


def _paddle_factory() -> OCRProvider:
    from app.services.ocr.paddle import PaddleOCRProvider

    return PaddleOCRProvider(
        languages=settings.OCR_LANGUAGES,
        use_gpu=settings.OCR_USE_GPU,
        det_limit_side_len=settings.OCR_DET_LIMIT_SIDE_LEN,
        drop_score=settings.OCR_DROP_SCORE,
        model_dir=settings.OCR_MODEL_DIR,
        cpu_threads=settings.OCR_CPU_THREADS,
        det_model_name=settings.OCR_DET_MODEL_NAME,
        rec_model_name=settings.OCR_REC_MODEL_NAME,
    )


def _stub_factory() -> OCRProvider:
    from app.services.ocr.stub import StubOCRProvider

    return StubOCRProvider()


register_provider("paddle", _paddle_factory)
register_provider("paddleocr", _paddle_factory)
register_provider("stub", _stub_factory)
