"""``POST /api/v1/ocr`` - recognise text in an uploaded image."""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.api.deps import enforce_rate_limit
from app.core.config import normalise_lang, settings
from app.core.exceptions import ImageTooLargeError, MissingFileError
from app.core.logging import get_logger, request_id_ctx
from app.schemas.common import ErrorResponse
from app.schemas.ocr import OCRResponse, build_response
from app.services.pipeline import PipelineOptions, run_pipeline
from app.utils.files import store_upload_if_enabled

logger = get_logger(__name__)

router = APIRouter(tags=["ocr"])

_READ_CHUNK = 64 * 1024

_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported image type"},
    422: {"model": ErrorResponse, "description": "Undecodable or unusable image"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    504: {"model": ErrorResponse, "description": "OCR timed out"},
}


async def _read_upload(upload: UploadFile) -> bytes:
    """Read the upload, enforcing the size limit as the bytes arrive.

    Streaming with a running total means a client that lies about (or omits)
    Content-Length still cannot make the process allocate past the limit.
    """
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.MAX_UPLOAD_SIZE:
            raise ImageTooLargeError(
                f"Upload exceeds the maximum size of {settings.MAX_UPLOAD_SIZE} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_languages(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    langs = [normalise_lang(part) for part in raw.split(",") if part.strip()]
    return langs or None


@router.post(
    "/ocr",
    response_model=OCRResponse,
    response_model_exclude_none=True,
    responses=_RESPONSES,
    summary="Recognise text in an image",
)
async def ocr(
    request: Request,
    image: UploadFile = File(..., description="Image file to recognise"),
    languages: Optional[str] = Query(
        default=None,
        description=(
            "Comma separated language codes, e.g. `en` or `en,arabic`. "
            "Defaults to the OCR_LANGUAGES setting."
        ),
        examples=["en,arabic"],
    ),
    preprocess: bool = Query(
        default=True, description="Run the OpenCV preprocessing chain"
    ),
    detect_orientation: Optional[bool] = Query(
        default=None,
        description="Retry other page orientations when the first pass finds little text",
    ),
    include_blocks: Optional[bool] = Query(
        default=None, description="Include every raw recognition box"
    ),
    include_regions: bool = Query(
        default=False, description="Include paragraph-like line groupings"
    ),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Drop boxes below this confidence"
    ),
    mrz: Optional[bool] = Query(
        default=None,
        description=(
            "Parse a machine-readable zone (ICAO 9303 TD1/TD2/TD3/MRV) if the "
            "document has one. Defaults to the ENABLE_MRZ setting. The `mrz` "
            "block is absent when no zone was found."
        ),
    ),
    include_mrz_raw: bool = Query(
        default=False,
        description=(
            "Include the zone exactly as read. It reproduces every field in one "
            "string, so it is opt-in."
        ),
    ),
    crop: bool = Query(
        default=False,
        description=(
            "Cut the document out of the frame and enlarge it before "
            "recognition. For a page photographed in the hand, where the "
            "printed text is small because the page is small in the frame. "
            "Costs roughly twice the recognition time, so it is opt-in: send "
            "the ordinary request first and retry with this when a field you "
            "expected comes back empty. No effect on an image the document "
            "already fills."
        ),
    ),
    _: str = Depends(enforce_rate_limit),
) -> OCRResponse:
    """Run OCR over one image and return the recognised text with geometry.

    The file is validated by content, held in memory for the duration of the
    request and then dropped. Nothing is written to disk unless ``STORE_UPLOADS``
    is explicitly enabled.
    """
    started = time.perf_counter()

    if image is None or not image.filename and not image.size:
        raise MissingFileError()

    data = await _read_upload(image)
    try:
        # The client's filename is used for nothing except this extension check.
        options = PipelineOptions.build(
            languages=_parse_languages(languages),
            preprocess_enabled=preprocess,
            detect_orientation=detect_orientation,
            include_blocks=include_blocks,
            include_regions=include_regions,
            min_confidence=min_confidence,
            parse_mrz=mrz,
            crop_to_document=crop,
        )
        result = await run_pipeline(data, image.filename, options)
        store_upload_if_enabled(data, result.image.detected_mime if result.image else None)
    finally:
        # Drop the reference promptly rather than waiting for the request scope.
        del data
        await image.close()

    elapsed = (time.perf_counter() - started) * 1000
    return build_response(
        result, request_id_ctx.get(), elapsed, include_mrz_raw=include_mrz_raw
    )
