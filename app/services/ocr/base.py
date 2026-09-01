"""OCR engine abstraction.

The rest of the application depends only on the types in this module, never on
PaddleOCR directly. Swapping in another engine means implementing
:class:`OCRProvider` and registering it - no API or pipeline changes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max


@dataclass
class TextBlock:
    """One recognised piece of text with its geometry and confidence."""

    text: str
    confidence: float
    polygon: List[Point] = field(default_factory=list)
    lang: str = ""

    # ------------------------------------------------------------- geometry
    @property
    def bbox(self) -> BBox:
        if not self.polygon:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def x_min(self) -> float:
        return self.bbox[0]

    @property
    def y_min(self) -> float:
        return self.bbox[1]

    @property
    def x_max(self) -> float:
        return self.bbox[2]

    @property
    def y_max(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bbox
        return x1 - x0

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0

    @property
    def center(self) -> Point:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def angle(self) -> float:
        """Baseline angle in degrees, derived from the top edge of the box."""
        if len(self.polygon) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.polygon[0], self.polygon[1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    def translated(self, dx: float, dy: float, scale: float = 1.0) -> TextBlock:
        """Map this block back into the coordinate space of another image."""
        return TextBlock(
            text=self.text,
            confidence=self.confidence,
            polygon=[(x / scale + dx, y / scale + dy) for x, y in self.polygon],
            lang=self.lang,
        )

    def to_dict(self) -> Dict[str, Any]:
        x0, y0, x1, y1 = self.bbox
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "lang": self.lang or None,
            "bbox": {
                "x": round(x0, 1),
                "y": round(y0, 1),
                "width": round(x1 - x0, 1),
                "height": round(y1 - y0, 1),
            },
            "polygon": [[round(x, 1), round(y, 1)] for x, y in self.polygon],
        }


@dataclass
class OCRResult:
    """Everything one engine invocation produced."""

    blocks: List[TextBlock] = field(default_factory=list)
    lang: str = ""
    duration_ms: float = 0.0
    provider: str = ""

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)

    @property
    def mean_confidence(self) -> float:
        scored = [b.confidence for b in self.blocks if b.text.strip()]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)

    def merged_with(self, other: OCRResult) -> OCRResult:
        return OCRResult(
            blocks=self.blocks + other.blocks,
            lang=",".join(sorted({x for x in (self.lang, other.lang) if x})),
            duration_ms=self.duration_ms + other.duration_ms,
            provider=self.provider or other.provider,
        )


class OCRProvider(ABC):
    """Contract every OCR engine implementation must satisfy."""

    #: Stable identifier used in configuration and in the version endpoint.
    name: str = "base"

    @abstractmethod
    def supported_languages(self) -> Sequence[str]:
        """Language codes this provider can be asked for."""

    @abstractmethod
    def warmup(self, languages: Optional[Sequence[str]] = None) -> None:
        """Load models ahead of the first request.

        Called once during application startup. Implementations must make this
        idempotent - it is also used by the readiness probe.
        """

    @abstractmethod
    def recognize(self, image: Any, lang: str) -> OCRResult:
        """Run detection + recognition over a BGR ``numpy`` image."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True once at least one model is resident in memory."""

    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "ready": self.is_ready(),
            "languages": list(self.supported_languages()),
        }

    def close(self) -> None:
        """Release engine resources. Optional."""
        return None
