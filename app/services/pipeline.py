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
    parse_mrz: bool = False
    parse_viz: bool = False

    @classmethod
    def build(
        cls,
        languages: Optional[Sequence[str]] = None,
        preprocess_enabled: Optional[bool] = None,
        detect_orientation: Optional[bool] = None,
        include_blocks: Optional[bool] = None,
        include_regions: bool = False,
        min_confidence: float = 0.0,
        parse_mrz: Optional[bool] = None,
        parse_viz: Optional[bool] = None,
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
            parse_mrz=(settings.ENABLE_MRZ if parse_mrz is None else parse_mrz),
            parse_viz=(settings.ENABLE_VIZ if parse_viz is None else parse_viz),
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
    #: Parsed machine-readable zone, or None when none was found or requested.
    mrz: Optional[Any] = None
    #: Visual-zone fields found by label, keyed by field name. Empty when none
    #: were found or none were requested. Unverified: no check digits exist.
    viz: Dict[str, Any] = field(default_factory=dict)

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

    # Deduplication picks one box per region across languages, which is right
    # for the page text and wrong for field extraction: a bilingual value like
    # "DAMMAM الدمام" loses one of its two scripts, and with it one of the two
    # fields it carries. Field extraction gets the boxes as recognised.
    every_block = list(blocks)
    if len(options.languages) > 1:
        blocks = deduplicate_blocks(blocks)

    lines = group_lines(blocks)
    regions = group_regions(lines) if options.include_regions else []
    timings["layout_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if not blocks:
        warnings.append("no text was recognised in the image")

    # ---------------------------------------------------------------- MRZ
    # Optional and additive: callers that do not ask for it see exactly the
    # response they saw before, and a document without a zone yields None
    # rather than an error. Parsing is text-only, so the cost is negligible
    # next to recognition.
    mrz_document = None
    if options.parse_mrz and blocks:
        started_mrz = time.perf_counter()
        try:
            mrz_document = _parse_mrz(blocks, summary_mean=None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("mrz_parse_failed", extra={"error": str(exc)})
            warnings.append("machine-readable zone parsing failed")
        timings["mrz_ms"] = round((time.perf_counter() - started_mrz) * 1000, 1)

        # A full-page pass optimises for the page, not for a dense band of
        # monospaced glyphs at its foot. When it misses the zone entirely -
        # routine on faint or older document designs - locate the band and
        # recognise it on its own terms before concluding there is none.
        #
        # A zone that parsed is not necessarily a zone that was read correctly:
        # the name carries no check digit, so a misread there validates like
        # anything else. _mrz_looks_truncated spots the shape that leaves, and
        # the band pass gets its chance on those too.
        if settings.MRZ_BAND_FALLBACK and (
            mrz_document is None or _mrz_looks_truncated(mrz_document)
        ):
            started_band = time.perf_counter()
            try:
                from_band = await _parse_mrz_from_band(
                    engine, prepared.image, options.languages
                )
                mrz_document = _better_mrz(mrz_document, from_band)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("mrz_band_pass_failed", extra={"error": str(exc)})
            timings["mrz_band_ms"] = round(
                (time.perf_counter() - started_band) * 1000, 1
            )

    # ---------------------------------------------------------------- VIZ
    # The fields printed beside a label and absent from the machine-readable
    # zone. Additive and opt-in, like the MRZ block above.
    viz_fields: Dict[str, Any] = {}
    if options.parse_viz and blocks:
        started_viz = time.perf_counter()
        try:
            viz_fields = _parse_viz(every_block)
            if settings.VIZ_FALLBACK:
                viz_fields = await _fill_viz_gaps(
                    engine,
                    prepared.image,
                    options.languages,
                    viz_fields,
                    is_identity_document=mrz_document is not None,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("viz_parse_failed", extra={"error": str(exc)})
            warnings.append("visual-zone field extraction failed")
        timings["viz_ms"] = round((time.perf_counter() - started_viz) * 1000, 1)

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
        mrz=mrz_document,
        viz=viz_fields,
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
            # Whether a zone was found and whether it validated - never its
            # contents, which are personal data.
            "mrz_found": mrz_document is not None,
            "mrz_valid": bool(mrz_document and mrz_document.valid),
            # Which fields were found, never what they said.
            "viz_fields": sorted(viz_fields),
        },
    )
    return result


def _mrz_looks_truncated(document: Any) -> bool:
    """Did the name field lose its separator?

    ``SURNAME<<GIVEN<NAMES`` becomes one long surname the moment a ``<<`` is
    read as anything else, and no check digit covers the name, so the result
    validates cleanly while being wrong. An empty given-names field on a
    document that has a surname is the shape that leaves behind.

    A holder with only one name produces the same shape legitimately, which is
    why this only earns a second look - never a rejection.
    """
    try:
        surname = document.value("surname")
        given_names = document.value("given_names")
    except Exception:  # pragma: no cover - defensive
        return False
    return bool(surname) and not given_names


def _better_mrz(first: Any, second: Any) -> Any:
    """Pick between a page-pass parse and a band-pass parse of the same zone.

    Completeness decides: a parse that recovered the given names read a
    separator the other one missed. Nothing else about the two is comparable -
    both satisfy every check digit, which is exactly the problem.
    """
    if second is None:
        return first
    if first is None:
        return second
    if _mrz_looks_truncated(first) and not _mrz_looks_truncated(second):
        logger.debug("mrz_name_recovered_from_band")
        return second
    return first


def _parse_viz(blocks: Sequence[TextBlock]) -> Dict[str, Any]:
    """Extract the labelled visual-zone fields from recognised boxes."""
    from app.services import viz

    return viz.extract(blocks)


async def _fill_viz_gaps(
    engine: OCREngine,
    image: Any,
    languages: Sequence[str],
    found: Dict[str, Any],
    is_identity_document: bool = False,
) -> Dict[str, Any]:
    """Re-read the data region enlarged, and fill in only what is still missing.

    The harsher processing that rescues faint print degrades print that was
    already legible, so what the first pass read is kept as it read it. This
    pass exists to turn an absent field into a present one, never to revise a
    present one.

    It costs a full extra recognition, so it needs a reason to believe the
    fields are there to be found: a field already read, or a machine-readable
    zone. An invoice has neither, and re-reading one enlarged would cost every
    caller a second pass to discover again that it holds no passport fields.
    """
    from app.services import viz
    from app.services.image_processing.preprocess import enhance_faint_text

    if all(spec.name in found for spec in viz.FIELD_SPECS):
        return found
    if not found and not is_identity_document:
        return found

    height = image.shape[0]
    # The zone lives above the machine-readable band; including it wastes the
    # enlargement on glyphs that are already read elsewhere.
    region = image[0 : max(1, int(height * 0.80)), :]
    enhanced = enhance_faint_text(region, settings.VIZ_UPSCALE_FACTOR)

    blocks: List[TextBlock] = []
    for language in _viz_languages(languages):
        recognised, _, _ = await _recognize_all(engine, enhanced, (language,))
        blocks.extend(recognised)
    if not blocks:
        return found

    filled = dict(found)
    for name, candidate in viz.extract(blocks).items():
        if name in filled:
            continue
        if candidate.confidence < settings.VIZ_FALLBACK_MIN_CONFIDENCE:
            continue
        filled[name] = candidate
        logger.debug("viz_recovered_from_region", extra={"field": name})
    return filled


def _viz_languages(languages: Sequence[str]) -> List[str]:
    """Both scripts are needed: the fields come in an Arabic and a latin form."""
    ordered = [lang for lang in languages]
    for required in ("arabic", "en"):
        if required not in ordered:
            ordered.append(required)
    return ordered


async def _parse_mrz_from_band(
    engine: OCREngine, image: Any, languages: Sequence[str]
):
    """Locate the MRZ band, recognise it alone, and parse the result.

    The zone is Latin OCR-B, so it is read with the latin model regardless of
    the languages the caller asked for; an arabic-only request still gets its
    passport parsed. Candidates are tried best-scoring first and the first
    structurally valid parse wins - nothing is returned when none of them
    parses, exactly as when no band was found at all.
    """
    from app.services.image_processing.mrz_locator import (
        find_mrz_regions,
        prepare_mrz_crop,
    )

    regions = find_mrz_regions(image, max_regions=settings.MRZ_BAND_MAX_REGIONS)
    if not regions:
        return None

    # Latin first. The caller's own languages are the fallback for deployments
    # that ship no latin model at all.
    latin_first: List[Sequence[str]] = [("en",)]
    if languages and list(languages) != ["en"]:
        latin_first.append(tuple(languages))

    for region in regions:
        crop = prepare_mrz_crop(image, region)
        if crop is None:
            continue
        for attempt in latin_first:
            blocks, _, _ = await _recognize_all(engine, crop, attempt)
            if not blocks:
                continue
            document = _parse_mrz(blocks)
            if document is not None:
                logger.debug(
                    "mrz_recovered_from_band",
                    extra={"source": region.source, "score": round(region.score, 3)},
                )
                return document
            # The latin model read something that did not parse; a second
            # model on the same crop will not read it better.
            break
    return None


def _parse_mrz(blocks: Sequence[TextBlock], summary_mean: Optional[float] = None):
    """Find and parse a machine-readable zone in the recognised boxes.

    Uses the geometry-aware detector: a zone line is frequently split across
    several recognition boxes, and regrouping them by baseline recovers the
    line that a plain text join would mangle.
    """
    from app.services.mrz.detector import detect_and_parse
    from app.services.mrz.icao import ICAOMRZParser

    result = detect_and_parse(blocks)
    if result is None or not result.structure_valid:
        return None
    return ICAOMRZParser().build_document(result, ocr_confidence=result.ocr_confidence)
