"""The parser interface.

Callers depend on :class:`MRZParser` and :class:`MRZDocument`, never on a
specific standard's implementation. Adding support for another machine-readable
text standard means implementing this interface and registering it - no caller
changes.

The interface takes **text lines that have already been recognised**. It does no
image handling, no OCR and no HTTP, and imports nothing from those layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

from app.services.mrz.document import MRZDocument


class MRZParser(ABC):
    """Contract for a machine-readable text parser."""

    #: Stable identifier used in configuration and in :meth:`info`.
    name: str = "base"

    @abstractmethod
    def supported_formats(self) -> Sequence[str]:
        """Format identifiers this parser understands."""

    @abstractmethod
    def detect_format(self, lines: Sequence[str]) -> Optional[str]:
        """The format these lines look like, or ``None`` if none applies."""

    @abstractmethod
    def parse_lines(
        self,
        lines: Sequence[str],
        mrz_type: Optional[str] = None,
        ocr_confidence: float = 0.0,
    ) -> Optional[MRZDocument]:
        """Parse lines that are believed to be a machine-readable zone.

        Returns ``None`` when the lines do not form a parsable zone. It must
        never return a partially invented document.
        """

    @abstractmethod
    def parse_text(
        self, text: str, ocr_confidence: float = 0.0
    ) -> Optional[MRZDocument]:
        """Find a zone inside a larger block of text and parse it."""

    def can_parse(self, lines: Sequence[str]) -> bool:
        """Cheap check for whether :meth:`parse_lines` is worth calling."""
        return self.detect_format(lines) is not None

    def info(self) -> Dict[str, object]:
        return {"parser": self.name, "formats": list(self.supported_formats())}


# --------------------------------------------------------------- registry
ParserFactory = Callable[[], MRZParser]

_REGISTRY: Dict[str, ParserFactory] = {}


def register_parser(name: str, factory: ParserFactory) -> None:
    _REGISTRY[name.strip().lower()] = factory


def available_parsers() -> List[str]:
    return sorted(_REGISTRY)


def create_parser(name: str = "icao9303") -> MRZParser:
    key = (name or "icao9303").strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown MRZ parser '{key}'. Available: {', '.join(available_parsers())}"
        )
    return factory()
