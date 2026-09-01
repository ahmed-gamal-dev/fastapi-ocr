"""A deterministic in-memory provider used by the test-suite.

It exists so the API, pipeline and MRZ layers can be exercised without the
PaddleOCR models being installed. It never invents text: it returns exactly the
blocks it was seeded with, and an empty result when it was seeded with nothing.
Selected with ``OCR_PROVIDER=stub`` - never use it to serve real traffic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from app.services.ocr.base import OCRProvider, OCRResult, TextBlock


class StubOCRProvider(OCRProvider):
    name = "stub"

    def __init__(self, scripted: Optional[Dict[str, List[TextBlock]]] = None) -> None:
        self._scripted: Dict[str, List[TextBlock]] = scripted or {}
        self._ready = False

    def set_blocks(self, lang: str, blocks: Sequence[TextBlock]) -> None:
        self._scripted[lang] = list(blocks)

    def clear(self) -> None:
        self._scripted.clear()

    def supported_languages(self) -> Sequence[str]:
        return ["en", "arabic"]

    def warmup(self, languages: Optional[Sequence[str]] = None) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def recognize(self, image: Any, lang: str = "en") -> OCRResult:
        blocks = [
            TextBlock(b.text, b.confidence, list(b.polygon), lang)
            for b in self._scripted.get(lang, [])
        ]
        return OCRResult(blocks=blocks, lang=lang, duration_ms=0.0, provider=self.name)
