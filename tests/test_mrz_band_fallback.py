"""The targeted MRZ band pass.

A full-page recognition optimises for the page, not for a dense band of
monospaced glyphs at its foot, and on faint or older document designs it misses
the zone completely. When that happens the pipeline locates the band, crops and
contrast-normalises it, and recognises it on its own terms.

The zone lines used here are the worked specimen published in ICAO Doc 9303
itself (`UTO` is the reserved code for the fictional state "Utopia"). No real
document data appears in this suite.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Sequence

import numpy as np
import pytest

from app.core.config import settings
from app.services.ocr.base import OCRResult, TextBlock
from app.services.ocr.engine import get_engine, reset_engine
from app.services.ocr.stub import StubOCRProvider
from app.services.pipeline import PipelineOptions, run_pipeline
from tests.conftest import block, encode, make_image

SPECIMEN = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]

PAGE_BLOCKS = [
    block("REPUBLIC OF UTOPIA", 40, 60, 300, 26, 0.97),
    block("PASSPORT", 40, 130, 180, 26, 0.95),
]


def zone_blocks(lines: Sequence[str] = SPECIMEN, y: int = 40) -> List[TextBlock]:
    """Zone lines as recognition boxes.

    ``y`` matters: grouped into lines by baseline, a zone sharing a band with
    the page header would be merged into it and stop being a zone at all. On a
    page the zone sits at the foot; in a band crop it is all there is.
    """
    return [
        block(line, 20, y + index * 60, 880, 40, 0.96)
        for index, line in enumerate(lines)
    ]


class BandAwareProvider(StubOCRProvider):
    """A stub that answers differently for the page and for a band crop.

    This is the situation the fallback exists for: the zone is unreadable in
    the full-page pass and readable once the band has been isolated and
    normalised. Crops are told apart by shape - ``prepare_mrz_crop`` returns a
    wide, short strip, never a whole page.
    """

    def __init__(
        self,
        page: Sequence[TextBlock],
        band: Sequence[TextBlock],
        band_langs: Sequence[str] = ("en",),
    ) -> None:
        super().__init__()
        self._page = list(page)
        self._band = list(band)
        self._band_langs = tuple(band_langs)
        self.calls: List[tuple] = []

    def supported_languages(self) -> Sequence[str]:
        return ["en", "arabic"]

    def recognize(self, image: Any, lang: str = "en") -> OCRResult:
        height, width = image.shape[:2]
        is_band = width / float(height) > 4.0
        self.calls.append(("band" if is_band else "page", lang))
        source = self._page
        if is_band:
            source = list(self._band) if lang in self._band_langs else []
        blocks = [
            TextBlock(b.text, b.confidence, list(b.polygon), lang) for b in source
        ]
        return OCRResult(blocks=blocks, lang=lang, duration_ms=0.0, provider=self.name)

    @property
    def band_calls(self) -> List[tuple]:
        return [c for c in self.calls if c[0] == "band"]


def run(
    provider: StubOCRProvider,
    languages: Sequence[str] = ("en",),
    image: Optional[np.ndarray] = None,
    parse_mrz: bool = True,
):
    data = encode(image if image is not None else make_image())
    options = PipelineOptions.build(
        languages=list(languages), parse_mrz=parse_mrz, include_blocks=False
    )

    async def go():
        reset_engine()
        engine = get_engine()
        engine.set_provider(provider)
        await engine.startup()
        try:
            return await run_pipeline(data, "page.png", options)
        finally:
            await engine.shutdown()

    try:
        return asyncio.run(go())
    finally:
        reset_engine()


@pytest.fixture
def band_fallback_on(monkeypatch):
    monkeypatch.setattr(settings, "MRZ_BAND_FALLBACK", True)


# ------------------------------------------------------------------- recovery
def test_a_zone_the_page_pass_missed_is_recovered_from_the_band(band_fallback_on):
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks())

    result = run(provider)

    assert result.mrz is not None, "the band pass should have recovered the zone"
    assert result.mrz.mrz_type == "TD3"
    assert result.mrz.valid is True
    assert result.mrz.value("document_number") == "L898902C3"
    assert result.mrz.value("birth_date") == "1974-08-12"
    assert provider.band_calls, "the band should have been recognised separately"


def test_the_recovered_zone_is_timed_separately(band_fallback_on):
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks())

    result = run(provider)

    assert "mrz_band_ms" in result.timings
    assert result.timings["mrz_band_ms"] >= 0.0


def test_page_text_is_not_rewritten_by_the_band_pass(band_fallback_on):
    """The band pass answers the MRZ question, and leaves the page text alone."""
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks())

    result = run(provider)

    assert "REPUBLIC OF UTOPIA" in result.text
    for line in SPECIMEN:
        assert line not in result.text


# ------------------------------------------------------------------ fast path
def test_the_band_pass_does_not_run_when_the_page_pass_found_a_zone(band_fallback_on):
    """A document that reads cleanly must not pay for the second pass."""
    provider = BandAwareProvider(page=PAGE_BLOCKS + zone_blocks(y=500), band=[])

    result = run(provider)

    assert result.mrz is not None
    assert result.mrz.valid is True
    assert provider.band_calls == []
    assert "mrz_band_ms" not in result.timings


def test_the_band_pass_does_not_run_when_mrz_was_not_requested(band_fallback_on):
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks())

    result = run(provider, parse_mrz=False)

    assert result.mrz is None
    assert provider.band_calls == []


# -------------------------------------------------------------------- opt out
def test_the_fallback_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "MRZ_BAND_FALLBACK", False)
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks())

    result = run(provider)

    assert result.mrz is None
    assert provider.band_calls == []
    assert "mrz_band_ms" not in result.timings


# --------------------------------------------------------------- never invent
def test_an_unreadable_band_yields_no_document(band_fallback_on):
    """A band that will not parse returns nothing - never a partial guess."""
    noise = [block("....-----....", 20, 40, 880, 40, 0.42)]
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=noise)

    result = run(provider)

    assert result.mrz is None
    assert provider.band_calls, "the band pass should still have been attempted"


def test_a_corrupt_zone_is_never_reported_as_valid(band_fallback_on):
    """Structure alone is not enough; the digits have to agree.

    The document number's check digit is corrupted rather than the composite
    one, because the composite digit does not exist in every 2x44 format - a
    zone with only that digit wrong is a legitimately valid MRV-A.
    """
    broken = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C31UTO7408122F1204159ZE184226B<<<<<10",
    ]
    provider = BandAwareProvider(page=PAGE_BLOCKS, band=zone_blocks(broken))

    result = run(provider)

    assert result.mrz is None or result.mrz.valid is False


# --------------------------------------------------------------- latin model
def test_an_arabic_only_request_still_reads_the_latin_zone(band_fallback_on):
    """The zone is Latin OCR-B whatever the caller asked the page to be read as."""
    provider = BandAwareProvider(
        page=PAGE_BLOCKS, band=zone_blocks(), band_langs=("en",)
    )

    result = run(provider, languages=("arabic",))

    assert result.mrz is not None
    assert result.mrz.valid is True
    assert ("band", "en") in provider.calls


def test_the_callers_language_is_tried_when_no_latin_model_answers(band_fallback_on):
    """Deployments without a latin model still get their zone parsed."""
    provider = BandAwareProvider(
        page=PAGE_BLOCKS, band=zone_blocks(), band_langs=("arabic",)
    )

    result = run(provider, languages=("arabic",))

    assert result.mrz is not None
    assert result.mrz.valid is True
    assert ("band", "arabic") in provider.calls
