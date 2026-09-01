"""Request options and response models for the OCR endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import BoundingBox


class TextBlockModel(BaseModel):
    """A single recognition box, as returned by the engine."""

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    lang: Optional[str] = None
    bbox: BoundingBox
    polygon: List[List[float]] = Field(
        default_factory=list,
        description="Four corner points, clockwise from the top-left",
    )


class TextLineModel(BaseModel):
    """Recognition boxes assembled into one visual line, in reading order."""

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    languages: List[str] = Field(default_factory=list)
    bbox: BoundingBox


class TextRegionModel(BaseModel):
    """Consecutive lines grouped into a paragraph-like region."""

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    line_count: int
    bbox: BoundingBox


class ConfidenceSummary(BaseModel):
    mean: float = Field(..., ge=0.0, le=1.0)
    min: float = Field(..., ge=0.0, le=1.0)
    max: float = Field(..., ge=0.0, le=1.0)


class ImageInfo(BaseModel):
    format: str = Field(..., description="MIME type detected from the file content")
    size_bytes: int
    original_width: int
    original_height: int
    processed_width: int = Field(
        ..., description="Width of the image the engine actually saw"
    )
    processed_height: int


class PreprocessingInfo(BaseModel):
    steps: List[str] = Field(default_factory=list)
    scale: float = 1.0
    rotation: int = 0
    skew_angle: float = 0.0
    perspective_corrected: bool = False


class OCRResponse(BaseModel):
    """Successful result of ``POST /api/v1/ocr``."""

    success: bool = True
    request_id: Optional[str] = None
    text: str = Field(..., description="All recognised text, one line per row")
    languages: List[str] = Field(
        default_factory=list,
        description=(
            "Language models that were run over the image. A language is listed "
            "even if it contributed no text."
        ),
    )
    confidence: ConfidenceSummary
    line_count: int
    word_count: int
    lines: List[TextLineModel] = Field(default_factory=list)
    regions: Optional[List[TextRegionModel]] = None
    blocks: Optional[List[TextBlockModel]] = None
    image: ImageInfo
    preprocessing: PreprocessingInfo
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: float
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "request_id": "0f9c1f7a1b6e4d1c9a2b3c4d5e6f7a8b",
                "text": "INVOICE 2026-09\nTOTAL 1240.00",
                "languages": ["en"],
                "confidence": {"mean": 0.94, "min": 0.88, "max": 0.99},
                "line_count": 2,
                "word_count": 4,
                "lines": [
                    {
                        "text": "INVOICE 2026-09",
                        "confidence": 0.97,
                        "min_confidence": 0.95,
                        "languages": ["en"],
                        "bbox": {"x": 40, "y": 40, "width": 250, "height": 22},
                    }
                ],
                "image": {
                    "format": "image/png",
                    "size_bytes": 84213,
                    "original_width": 1100,
                    "original_height": 700,
                    "processed_width": 1100,
                    "processed_height": 700,
                },
                "preprocessing": {
                    "steps": ["deskew", "enhance"],
                    "scale": 1.0,
                    "rotation": 0,
                    "skew_angle": -0.8,
                    "perspective_corrected": False,
                },
                "timings_ms": {"decode_ms": 12.1, "ocr_ms": 812.4},
                "processing_time_ms": 861.7,
                "warnings": [],
            }
        }
    )


def _bbox(data: Dict[str, Any]) -> BoundingBox:
    return BoundingBox(**data)


def build_response(result: Any, request_id: Optional[str], elapsed_ms: float) -> OCRResponse:
    """Map a :class:`~app.services.pipeline.PipelineResult` onto the wire format."""
    lines = [
        TextLineModel(**{**data, "bbox": _bbox(data["bbox"])})
        for data in (line.to_dict() for line in result.lines)
    ]
    blocks = None
    if result.blocks:
        blocks = [
            TextBlockModel(**{**data, "bbox": _bbox(data["bbox"])})
            for data in (block.to_dict() for block in result.blocks)
        ]
    regions = None
    if result.regions:
        regions = [
            TextRegionModel(**{**data, "bbox": _bbox(data["bbox"])})
            for data in (region.to_dict() for region in result.regions)
        ]

    preprocessing = (
        result.preprocessing.to_dict()
        if result.preprocessing is not None
        else {"steps": []}
    )
    return OCRResponse(
        request_id=request_id,
        text=result.text,
        languages=result.languages,
        confidence=ConfidenceSummary(**result.confidence),
        line_count=len(result.lines),
        word_count=result.word_count,
        lines=lines,
        regions=regions,
        blocks=blocks,
        image=ImageInfo(
            format=result.image.detected_mime,
            size_bytes=result.image.size_bytes,
            original_width=result.image.width,
            original_height=result.image.height,
            processed_width=result.processed_size[0],
            processed_height=result.processed_size[1],
        ),
        preprocessing=PreprocessingInfo(**preprocessing),
        timings_ms=result.timings,
        processing_time_ms=round(elapsed_ms, 1),
        warnings=result.warnings,
    )
