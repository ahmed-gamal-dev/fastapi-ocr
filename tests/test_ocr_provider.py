"""The OCRProvider abstraction, the registry and the PaddleOCR adapter."""

from __future__ import annotations

import inspect

import pytest

from app.core.exceptions import OCRFailedError
from app.services.ocr.base import OCRProvider, OCRResult, TextBlock
from app.services.ocr.paddle import PaddleOCRProvider, _parse_output, resolve_lang
from app.services.ocr.registry import (
    available_providers,
    create_provider,
    register_provider,
)
from app.services.ocr.stub import StubOCRProvider
from tests.conftest import block


# ------------------------------------------------------------------- contract
def test_provider_is_abstract():
    with pytest.raises(TypeError):
        OCRProvider()  # type: ignore[abstract]


def test_every_registered_provider_implements_the_contract():
    for name in available_providers():
        if name in ("paddle", "paddleocr"):
            continue  # requires the models to be installed
        provider = create_provider(name)
        assert isinstance(provider, OCRProvider)
        for method in ("recognize", "warmup", "is_ready", "supported_languages"):
            assert callable(getattr(provider, method))


def test_unknown_provider_is_rejected():
    with pytest.raises(OCRFailedError):
        create_provider("does-not-exist")


def test_a_new_provider_can_be_registered():
    """The whole point of the abstraction: swap the engine without touching the API."""

    class FakeProvider(OCRProvider):
        name = "fake"

        def supported_languages(self):
            return ["en"]

        def warmup(self, languages=None):
            return None

        def is_ready(self):
            return True

        def recognize(self, image, lang="en"):
            return OCRResult([TextBlock("fake", 1.0, [], lang)], lang, 0.0, self.name)

    register_provider("fake", FakeProvider)
    provider = create_provider("fake")
    assert provider.recognize(None, "en").text == "fake"
    assert "fake" in available_providers()


# ----------------------------------------------------------------------- stub
def test_stub_returns_only_what_it_was_given():
    provider = StubOCRProvider()
    provider.set_blocks("en", [block("scripted", 10, 10)])
    assert provider.recognize(None, "en").text == "scripted"
    # It never invents text for a language it was not seeded with.
    assert provider.recognize(None, "arabic").blocks == []


def test_stub_reports_readiness_after_warmup():
    provider = StubOCRProvider()
    assert provider.is_ready() is False
    provider.warmup()
    assert provider.is_ready() is True


# ---------------------------------------------------------------- result types
def test_ocr_result_aggregates_text_and_confidence():
    result = OCRResult([block("one", 0, 0, confidence=1.0), block("two", 0, 40, confidence=0.5)])
    assert result.text == "one\ntwo"
    assert result.mean_confidence == 0.75


def test_results_can_be_merged_across_languages():
    merged = OCRResult([block("en text")], "en").merged_with(
        OCRResult([block("نص", lang="arabic")], "arabic")
    )
    assert len(merged.blocks) == 2
    assert merged.lang == "arabic,en"


def test_text_block_geometry():
    item = TextBlock("x", 0.9, [(10, 20), (110, 20), (110, 50), (10, 50)])
    assert item.bbox == (10, 20, 110, 50)
    assert item.center == (60.0, 35.0)
    assert (item.width, item.height) == (100, 30)
    assert item.area == 3000
    assert item.angle == 0.0


def test_text_block_without_geometry_is_safe():
    item = TextBlock("x", 0.9)
    assert item.bbox == (0.0, 0.0, 0.0, 0.0)
    assert item.width == 0.0
    assert item.to_dict()["polygon"] == []


def test_text_block_can_be_mapped_back_to_source_coordinates():
    item = TextBlock("x", 0.9, [(10, 10), (20, 10), (20, 20), (10, 20)])
    moved = item.translated(100, 200, scale=2.0)
    assert moved.polygon[0] == (105.0, 205.0)


# --------------------------------------------------------------- paddle adapter
@pytest.mark.parametrize(
    "given,expected",
    [("ar", "arabic"), ("ARABIC", "arabic"), ("en", "en"), ("english", "en"), ("fr", "fr")],
)
def test_language_aliases_resolve(given, expected):
    assert resolve_lang(given) == expected


def test_parse_output_handles_the_2x_shape():
    raw = [
        [
            [[[10, 10], [110, 10], [110, 40], [10, 40]], ("HELLO", 0.97)],
            [[[10, 50], [90, 50], [90, 80], [10, 80]], ("WORLD", 0.88)],
        ]
    ]
    blocks = _parse_output(raw, "en")
    assert [b.text for b in blocks] == ["HELLO", "WORLD"]
    assert blocks[0].confidence == 0.97
    assert blocks[0].bbox == (10, 10, 110, 40)


def test_parse_output_handles_the_3x_shape():
    raw = [
        {
            "rec_texts": ["HELLO", "WORLD"],
            "rec_scores": [0.97, 0.88],
            "rec_polys": [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                [[10, 50], [90, 50], [90, 80], [10, 80]],
            ],
        }
    ]
    blocks = _parse_output(raw, "en")
    assert [b.text for b in blocks] == ["HELLO", "WORLD"]
    assert blocks[1].confidence == 0.88


def test_parse_output_handles_a_nested_res_payload():
    raw = [{"res": {"rec_texts": ["A"], "rec_scores": [0.5], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}}]
    assert _parse_output(raw, "en")[0].text == "A"


@pytest.mark.parametrize("raw", [None, [], [None], [[]], ["nonsense"], [{"rec_texts": []}]])
def test_parse_output_survives_empty_and_malformed_payloads(raw):
    assert _parse_output(raw, "en") == []


def test_parse_output_skips_blank_text():
    raw = [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("   ", 0.9)]]]
    assert _parse_output(raw, "en") == []


def test_build_kwargs_only_passes_supported_arguments():
    """The adapter must not hand PaddleOCR 3.x arguments only 2.x understands."""
    provider = PaddleOCRProvider(languages=["en"])

    def legacy(self, lang=None, use_angle_cls=None, show_log=None, drop_score=None):
        ...

    kwargs = provider._build_kwargs(inspect.signature(legacy), "en")
    assert set(kwargs) <= {"lang", "use_angle_cls", "show_log", "drop_score"}
    assert kwargs["lang"] == "en"

    def modern(self, lang=None, use_textline_orientation=None, text_rec_score_thresh=None):
        ...

    kwargs = provider._build_kwargs(inspect.signature(modern), "arabic")
    assert "use_angle_cls" not in kwargs
    assert kwargs["use_textline_orientation"] is True


def test_provider_reports_itself_as_not_ready_before_loading():
    provider = PaddleOCRProvider(languages=["en"])
    assert provider.is_ready() is False
    assert provider.info()["provider"] == "paddleocr"


def test_missing_paddleocr_raises_a_domain_error(monkeypatch):
    """A missing engine must surface as OCR_FAILED, not as an ImportError."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("no module named paddleocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(OCRFailedError):
        PaddleOCRProvider(languages=["en"])._create_engine("en")


# ------------------------------------------------- model override resolution
@pytest.mark.parametrize(
    "value,lang,expected",
    [
        (None, "en", None),
        ("", "en", None),
        # A bare name applies to every language.
        ("PP-OCRv5_mobile_rec", "en", "PP-OCRv5_mobile_rec"),
        ("PP-OCRv5_mobile_rec", "arabic", "PP-OCRv5_mobile_rec"),
        # The mapping form picks per language.
        ("en:LATIN_REC,arabic:ARABIC_REC", "en", "LATIN_REC"),
        ("en:LATIN_REC,arabic:ARABIC_REC", "arabic", "ARABIC_REC"),
        # Aliases resolve before matching.
        ("ar:ARABIC_REC", "arabic", "ARABIC_REC"),
        ("english:LATIN_REC", "en", "LATIN_REC"),
        # A language with no entry keeps the engine default.
        ("en:LATIN_REC", "arabic", None),
    ],
)
def test_model_override_resolution(value, lang, expected):
    from app.services.ocr.paddle import resolve_model_override

    assert resolve_model_override(value, lang) == expected


def test_a_per_language_override_reaches_the_constructor_kwargs():
    """Recognition models are script-specific: a Latin model must not be sent
    for the Arabic pipeline."""
    provider = PaddleOCRProvider(
        languages=["en", "arabic"],
        rec_model_name="en:LATIN_REC,arabic:ARABIC_REC",
    )

    def modern(self, lang=None, use_textline_orientation=None,
               text_recognition_model_name=None):
        ...

    signature = inspect.signature(modern)
    assert provider._build_kwargs(signature, "en")["text_recognition_model_name"] == "LATIN_REC"
    assert (
        provider._build_kwargs(signature, "arabic")["text_recognition_model_name"]
        == "ARABIC_REC"
    )


def test_legacy_dialect_ignores_model_name_overrides():
    """The 3.x model-name arguments do not exist in 2.x and must not be sent."""
    provider = PaddleOCRProvider(languages=["en"], rec_model_name="LATIN_REC")

    def legacy(self, lang=None, use_angle_cls=None, drop_score=None):
        ...

    kwargs = provider._build_kwargs(inspect.signature(legacy), "en")
    assert "text_recognition_model_name" not in kwargs


def test_the_engine_facing_language_code_matches_the_dialect():
    """3.x takes an ISO code; the service-level name stays 'arabic' either way."""
    provider = PaddleOCRProvider(languages=["arabic"])

    def modern(self, lang=None, use_textline_orientation=None):
        ...

    def legacy(self, lang=None, use_angle_cls=None):
        ...

    assert provider._build_kwargs(inspect.signature(modern), "arabic")["lang"] == "ar"
    assert provider._build_kwargs(inspect.signature(legacy), "arabic")["lang"] == "arabic"


def test_det_limit_type_is_pinned_to_max():
    """Left unset, the side length is applied as a minimum and small scans get
    upscaled enormously - seconds of inference for no accuracy gain."""
    provider = PaddleOCRProvider(languages=["en"])

    def modern(self, lang=None, text_det_limit_side_len=None, text_det_limit_type=None):
        ...

    kwargs = provider._build_kwargs(inspect.signature(modern), "en")
    assert kwargs["text_det_limit_type"] == "max"
