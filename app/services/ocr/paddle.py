"""PaddleOCR implementation of :class:`OCRProvider`.

PaddleOCR's constructor and call signatures changed between 2.x and 3.x, so the
adapter introspects what the installed version actually accepts instead of
hard-coding one dialect. Models are created once per language and reused for
the lifetime of the process.
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from app.core.exceptions import OCRFailedError
from app.core.logging import get_logger
from app.services.ocr.base import OCRProvider, OCRResult, TextBlock

logger = get_logger(__name__)

# Friendly code -> PaddleOCR model language.
LANG_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "latin": "en",
    "ar": "arabic",
    "ara": "arabic",
    "arabic": "arabic",
    "fr": "fr",
    "de": "german",
    "es": "es",
    "ru": "ru",
    "ch": "ch",
    "chinese": "ch",
    "japan": "japan",
    "korean": "korean",
}


# The engine-facing code for a language changed between major versions:
# 2.x names the recognition model ("arabic"), 3.x takes an ISO code ("ar").
# The service-level name stays stable either way - only this mapping moves.
ENGINE_LANG_V3 = {
    "arabic": "ar",
    "en": "en",
    "fr": "fr",
    "german": "de",
    "es": "es",
    "ru": "ru",
    "ch": "ch",
    "japan": "japan",
    "korean": "korean",
}


def resolve_lang(lang: str) -> str:
    """Normalise a caller's language code to the service-level name."""
    return LANG_ALIASES.get((lang or "").strip().lower(), (lang or "en").strip().lower())


def resolve_model_override(value: Optional[str], lang: str) -> Optional[str]:
    """Pick the model override that applies to ``lang``.

    Accepts either a bare model name, which applies to every language, or a
    per-language mapping ``en:PP-OCRv5_mobile_rec,arabic:arabic_PP-OCRv5_mobile_rec``.

    The mapping form matters: recognition models are script-specific, so a bare
    Latin model name in a multi-language deployment would silently read Arabic
    pages with the English model and return transliterated nonsense at low
    confidence.
    """
    if not value:
        return None
    value = value.strip()
    if ":" not in value:
        return value
    for pair in value.split(","):
        name, _, model = pair.partition(":")
        if resolve_lang(name.strip()) == resolve_lang(lang):
            return model.strip() or None
    return None


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        use_gpu: bool = False,
        det_limit_side_len: int = 1600,
        drop_score: float = 0.35,
        model_dir: Optional[str] = None,
        cpu_threads: int = 1,
        det_model_name: Optional[str] = None,
        rec_model_name: Optional[str] = None,
    ) -> None:
        self._languages = [resolve_lang(x) for x in (languages or ["en"])]
        self._use_gpu = use_gpu
        self._det_limit_side_len = det_limit_side_len
        self._drop_score = drop_score
        self._model_dir = model_dir
        self._cpu_threads = max(1, cpu_threads)
        self._det_model_name = det_model_name
        self._rec_model_name = rec_model_name
        if (
            det_model_name or rec_model_name
        ) and ":" not in f"{det_model_name or ''}{rec_model_name or ''}" and len(
            set(self._languages)
        ) > 1:
            logger.warning(
                "ocr_model_override_is_global",
                extra={
                    "languages": sorted(set(self._languages)),
                    "hint": "recognition models are script-specific; use "
                    "'en:MODEL,arabic:MODEL' to override per language",
                },
            )
        self._engines: Dict[str, Any] = {}
        # PaddleOCR predictors are not thread safe: one lock per language.
        self._locks: Dict[str, threading.Lock] = {}
        self._init_lock = threading.Lock()
        self._version: Optional[str] = None

    # ------------------------------------------------------------- lifecycle
    def _build_kwargs(self, signature: inspect.Signature, lang: str) -> Dict[str, Any]:
        """Build the constructor kwargs in the dialect this version speaks.

        PaddleOCR 2.x takes ``**kwargs`` and merges them over its own argparse
        defaults, so introspection cannot tell us which names it honours - an
        intersection with the signature would leave nothing but ``lang`` and
        silently fall back to every default. 3.x declares its parameters
        explicitly under different names. So: detect the dialect, then send that
        dialect's names.
        """
        legacy: Dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": True,
            "show_log": False,
            "use_gpu": self._use_gpu,
            "det_limit_side_len": self._det_limit_side_len,
            "det_limit_type": "max",
            "drop_score": self._drop_score,
            # One inference thread per predictor: the engine already runs inside
            # a bounded pool, and letting it fan out over every core makes
            # concurrent requests contend instead of finishing.
            "cpu_threads": self._cpu_threads,
        }
        modern: Dict[str, Any] = {
            "lang": ENGINE_LANG_V3.get(lang, lang),
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "text_det_limit_side_len": self._det_limit_side_len,
            # Without this the side length is applied as a *minimum*: PP-OCRv6
            # upscales a 1100x300 scan to 4693x1600, which costs seconds of
            # inference and gigabytes of RSS for no accuracy gain.
            "text_det_limit_type": "max",
            "text_rec_score_thresh": self._drop_score,
        }

        accepted = set(signature.parameters)
        has_varkw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )

        # 3.x names its parameters explicitly; 2.x hides everything behind
        # **kwargs. Either way, prefer the dialect whose names are recognised.
        if accepted & set(modern) - {"lang"}:
            wanted = modern
        elif has_varkw:
            wanted = legacy
        else:
            wanted = {k: v for k, v in legacy.items() if k in accepted}

        if self._model_dir:
            wanted = {**wanted, "det_model_dir": self._model_dir}
        # 3.x only: choose a specific model instead of the language default.
        if wanted is modern:
            det = resolve_model_override(self._det_model_name, lang)
            rec = resolve_model_override(self._rec_model_name, lang)
            if det:
                wanted = {**wanted, "text_detection_model_name": det}
            if rec:
                wanted = {**wanted, "text_recognition_model_name": rec}
        if has_varkw:
            return dict(wanted)
        return {k: v for k, v in wanted.items() if k in accepted}

    def _create_engine(self, lang: str) -> Any:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise OCRFailedError(
                "PaddleOCR is not installed. Install the project requirements "
                "into your virtualenv first."
            ) from exc

        try:
            import paddleocr  # type: ignore

            self._version = getattr(paddleocr, "__version__", None)
        except Exception:  # pragma: no cover
            self._version = None

        kwargs = self._build_kwargs(inspect.signature(PaddleOCR.__init__), lang)
        started = time.perf_counter()
        try:
            engine = PaddleOCR(**kwargs)
        except TypeError:
            # Last resort: the minimal signature every version supports.
            engine = PaddleOCR(lang=lang)
        logger.info(
            "ocr_model_loaded",
            extra={
                "lang": lang,
                "paddleocr_version": self._version,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return engine

    def _get_engine(self, lang: str) -> Any:
        lang = resolve_lang(lang)
        engine = self._engines.get(lang)
        if engine is not None:
            return engine
        with self._init_lock:
            engine = self._engines.get(lang)
            if engine is None:
                engine = self._create_engine(lang)
                self._engines[lang] = engine
                self._locks[lang] = threading.Lock()
        return engine

    def warmup(self, languages: Optional[Sequence[str]] = None) -> None:
        import numpy as np

        targets = [resolve_lang(x) for x in (languages or self._languages)]
        blank = np.full((64, 320, 3), 255, dtype=np.uint8)
        for lang in targets:
            try:
                self._get_engine(lang)
                # A real inference pass allocates the kernels too, so the first
                # request does not pay for it.
                self._invoke(lang, blank)
            except Exception as exc:
                logger.error(
                    "ocr_warmup_failed", extra={"lang": lang, "error": str(exc)}
                )

    def is_ready(self) -> bool:
        return bool(self._engines)

    def supported_languages(self) -> Sequence[str]:
        return sorted(set(LANG_ALIASES.values()))

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data.update(
            {
                "loaded_languages": sorted(self._engines),
                "paddleocr_version": self._version,
                "gpu": self._use_gpu,
                "cpu_threads": self._cpu_threads,
                "det_model": self._det_model_name,
                "rec_model": self._rec_model_name,
            }
        )
        return data

    def close(self) -> None:
        self._engines.clear()
        self._locks.clear()

    # ------------------------------------------------------------ inference
    def _invoke(self, lang: str, image: Any) -> Any:
        engine = self._get_engine(lang)
        lock = self._locks.setdefault(lang, threading.Lock())
        with lock:
            if hasattr(engine, "predict"):
                try:
                    return engine.predict(image)
                except TypeError:  # pragma: no cover - signature drift
                    pass
            try:
                return engine.ocr(image, cls=True)
            except TypeError:
                return engine.ocr(image)

    def recognize(self, image: Any, lang: str = "en") -> OCRResult:
        lang = resolve_lang(lang)
        started = time.perf_counter()
        try:
            raw = self._invoke(lang, image)
        except Exception as exc:
            logger.error("ocr_inference_failed", extra={"lang": lang, "error": str(exc)})
            raise OCRFailedError("OCR inference failed") from exc

        blocks = _parse_output(raw, lang)
        duration = (time.perf_counter() - started) * 1000
        logger.debug(
            "ocr_completed",
            extra={"lang": lang, "blocks": len(blocks), "duration_ms": round(duration, 1)},
        )
        return OCRResult(
            blocks=blocks, lang=lang, duration_ms=duration, provider=self.name
        )


# --------------------------------------------------------------- output parsing
def _to_points(poly: Any) -> List[Any]:
    try:
        return [(float(p[0]), float(p[1])) for p in poly]
    except (TypeError, ValueError, IndexError):
        return []


def _parse_output(raw: Any, lang: str) -> List[TextBlock]:
    """Normalise every known PaddleOCR return shape into ``TextBlock`` list."""
    blocks: List[TextBlock] = []
    if raw is None:
        return blocks

    # PaddleOCR 3.x: list of result objects exposing rec_texts / rec_scores.
    for item in raw if isinstance(raw, (list, tuple)) else [raw]:
        if item is None:
            continue
        data = item
        if not isinstance(item, dict):
            for attr in ("json", "res", "_json"):
                candidate = getattr(item, attr, None)
                if isinstance(candidate, dict):
                    data = candidate
                    break
        if isinstance(data, dict):
            if "res" in data and isinstance(data["res"], dict):
                data = data["res"]
            texts = data.get("rec_texts")
            if texts is not None:
                scores = data.get("rec_scores") or []
                polys = data.get("rec_polys")
                if polys is None:
                    polys = data.get("dt_polys") or data.get("rec_boxes") or []
                for idx, text in enumerate(texts):
                    poly = _to_points(polys[idx]) if idx < len(polys) else []
                    score = float(scores[idx]) if idx < len(scores) else 0.0
                    if str(text).strip():
                        blocks.append(
                            TextBlock(str(text).strip(), score, poly, lang)
                        )
                continue

        # PaddleOCR 2.x: [[ [poly, (text, score)], ... ]]
        if isinstance(item, (list, tuple)):
            for entry in item:
                parsed = _parse_legacy_entry(entry, lang)
                if parsed is not None:
                    blocks.append(parsed)
    return blocks


def _parse_legacy_entry(entry: Any, lang: str) -> Optional[TextBlock]:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    poly, payload = entry[0], entry[1]
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        text, score = payload[0], payload[1]
    elif isinstance(payload, str):
        text, score = payload, 0.0
    else:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return TextBlock(text, score, _to_points(poly), lang)
