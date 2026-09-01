"""Generic OCR pipeline.

Upload bytes in, structured text out:

    validate -> decode -> preprocess -> recognise (per language)
             -> merge -> layout analysis -> confidence summary

The pipeline is document-agnostic. It reports what it did to the image and how
confident the engine was, and it never guesses at text it could not read.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.services.image_processing.loader import LoadedImage, load_image
from app.services.image_processing.preprocess import (
    PreprocessResult,
    preprocess,
    rotate90,
)
from app.services.layout import (
    TextLine,
    TextRegion,
    confidence_summary,
    deduplicate_blocks,
    full_text,
    group_lines,
    group_regions,
)
from app.services.ocr.base import OCRResult, TextBlock
from app.services.ocr.engine import OCREngine, get_engine

logger = get_logger(__name__)

# An orientation retry is only worth its cost when the first pass went badly.
ORIENTATION_RETRY_SCORE = 20.0


@dataclass
class PipelineOptions:
    languages: Sequence[str] = ()
    preprocess_enabled: bool = True
    detect_orientation: bool = True
    include_blocks: bool = True
    include_regions: bool = False
    min_confidence: float = 0.0

    @classmethod
    def build(
        cls,
        languages: Optional[Sequence[str]] = None,
        preprocess_enabled: Optional[bool] = None,
        detect_orientation: Optional[bool] = None,
        include_blocks: Optional[bool] = None,
        include_regions: bool = False,
        min_confidence: float = 0.0,
    ) -> PipelineOptions:
        return cls(
            languages=list(languages) if languages else list(settings.OCR_LANGUAGES),
            preprocess_enabled=(
                True if preprocess_enabled is None else preprocess_enabled
            ),
            detect_orientation=(
                settings.ENABLE_ORIENTATION_CORRECTION
                if detect_orientation is None
                else detect_orientation
            ),
            include_blocks=(
                settings.INCLUDE_RAW_BLOCKS_DEFAULT
                if include_blocks is None
                else include_blocks
            ),
            include_regions=include_regions,
            min_confidence=max(0.0, min(1.0, min_confidence)),
        )


@dataclass
class PipelineResult:
    text: str = ""
    lines: List[TextLine] = field(default_factory=list)
    regions: List[TextRegion] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)
    image: Optional[LoadedImage] = None
    processed_size: Tuple[int, int] = (0, 0)
    preprocessing: Optional[PreprocessResult] = None
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len([w for w in self.text.split() if w])


def _score(blocks: Sequence[TextBlock]) -> float:
    """How much readable text a pass produced. Drives orientation selection."""
    return sum(len(b.text.strip()) * b.confidence for b in blocks)


async def _recognize_all(
    engine: OCREngine, image: Any, languages: Sequence[str]
) -> Tuple[List[TextBlock], List[str], List[str]]:
    """Run every requested language over one image."""
    blocks: List[TextBlock] = []
    succeeded: List[str] = []
    warnings: List[str] = []
    results: List[OCRResult] = await engine.recognize_many(image, languages)
    for result in results:
        blocks.extend(result.blocks)
        succeeded.append(result.lang)
    for lang in languages:
        if lang not in succeeded:
            warnings.append(f"language '{lang}' could not be processed")
    return blocks, succeeded, warnings


async def run_pipeline(
    data: bytes,
    filename: Optional[str] = None,
    options: Optional[PipelineOptions] = None,
    engine: Optional[OCREngine] = None,
) -> PipelineResult:
    """Execute the full OCR pipeline over uploaded image bytes."""
    options = options or PipelineOptions.build()
    engine = engine or get_engine()
    timings: Dict[str, float] = {}
    warnings: List[str] = []

    started = time.perf_counter()
    loaded = load_image(data, filename)
    timings["decode_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # ---------------------------------------------------------- preprocessing
    started = time.perf_counter()
    if options.preprocess_enabled:
        prepared = preprocess(loaded.image)
    else:
        prepared = PreprocessResult(image=loaded.image, steps=[])
    timings["preprocess_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # ------------------------------------------------------------ recognition
    started = time.perf_counter()
    blocks, languages, lang_warnings = await _recognize_all(
        engine, prepared.image, options.languages
    )
    warnings.extend(lang_warnings)

    # Retry at other orientations only when the upright pass produced almost
    # nothing - a full four-way sweep on every request would quadruple latency.
    if options.detect_orientation and _score(blocks) < ORIENTATION_RETRY_SCORE:
        best_score = _score(blocks)
        for turns in (1, 2, 3):
            rotated = rotate90(prepared.image, turns)
            candidate, candidate_langs, _ = await _recognize_all(
                engine, rotated, options.languages
            )
            score = _score(candidate)
            if score > best_score:
                best_score = score
                blocks = candidate
                languages = candidate_langs
                prepared.image = rotated
                prepared.rotation = (turns * 90) % 360
                if "rotate" not in prepared.steps:
                    prepared.steps.append("rotate")
        if prepared.rotation:
            logger.debug("orientation_corrected", extra={"rotation": prepared.rotation})
    timings["ocr_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # ------------------------------------------------------------------ layout
    started = time.perf_counter()
    if options.min_confidence > 0:
        dropped = len(blocks)
        blocks = [b for b in blocks if b.confidence >= options.min_confidence]
        dropped -= len(blocks)
        if dropped:
            warnings.append(f"{dropped} block(s) dropped below min_confidence")

    if len(options.languages) > 1:
        blocks = deduplicate_blocks(blocks)

    lines = group_lines(blocks)
    regions = group_regions(lines) if options.include_regions else []
    timings["layout_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if not blocks:
        warnings.append("no text was recognised in the image")

    summary = confidence_summary(lines)
    if summary["mean"] and summary["mean"] < settings.MIN_OVERALL_CONFIDENCE:
        warnings.append("overall recognition confidence is low")

    height, width = prepared.image.shape[:2]
    result = PipelineResult(
        text=full_text(lines),
        lines=lines,
        regions=regions,
        blocks=blocks if options.include_blocks else [],
        languages=languages,
        confidence=summary,
        image=loaded,
        processed_size=(width, height),
        preprocessing=prepared,
        timings=timings,
        warnings=warnings,
    )

    # Counts and timings only: no recognised text ever reaches the log stream.
    logger.info(
        "ocr_pipeline_completed",
        extra={
            "languages": languages,
            "blocks": len(blocks),
            "lines": len(lines),
            "words": result.word_count,
            "mean_confidence": summary["mean"],
            "timings": timings,
            "steps": prepared.steps,
        },
    )
    return result
