"""The parser interface, the registry, and its independence from other layers."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional, Sequence

import pytest

from app.services.mrz import MRV_A, MRV_B, TD1, TD2, TD3
from app.services.mrz.base import (
    MRZParser,
    available_parsers,
    create_parser,
    register_parser,
)
from app.services.mrz.document import MRZDocument
from app.services.mrz.icao import ICAOMRZParser
from tests.mrz.conftest import ICAO_TD3_SPECIMEN, make_lines


# ------------------------------------------------------------------ contract
def test_the_interface_is_abstract():
    with pytest.raises(TypeError):
        MRZParser()  # type: ignore[abstract]


def test_the_icao_parser_implements_the_interface(parser):
    assert isinstance(parser, MRZParser)
    for method in ("parse_lines", "parse_text", "detect_format", "supported_formats"):
        assert callable(getattr(parser, method))


def test_supported_formats(parser):
    assert set(parser.supported_formats()) == {TD1, TD2, TD3, MRV_A, MRV_B}


def test_info_describes_the_parser(parser):
    info = parser.info()
    assert info["parser"] == "icao9303"
    assert TD3 in info["formats"]


def test_can_parse_is_a_cheap_precheck(parser, td3_lines):
    assert parser.can_parse(td3_lines) is True
    assert parser.can_parse(["not a zone at all"]) is False


def test_detect_format_identifies_the_layout(parser, td3_lines):
    assert parser.detect_format(td3_lines) == TD3
    assert parser.detect_format([]) is None


# ------------------------------------------------------------------ registry
def test_the_default_parser_is_registered():
    assert "icao9303" in available_parsers()
    assert isinstance(create_parser("icao9303"), ICAOMRZParser)


def test_the_registry_accepts_an_alias():
    assert isinstance(create_parser("icao"), ICAOMRZParser)


def test_create_parser_defaults_to_the_icao_implementation():
    assert isinstance(create_parser(), ICAOMRZParser)


def test_an_unknown_parser_is_rejected():
    with pytest.raises(ValueError, match="Unknown MRZ parser"):
        create_parser("does-not-exist")


def test_another_standard_can_be_plugged_in():
    """The point of the interface: a different machine-readable text standard
    can be added without touching any caller."""

    class FakeParser(MRZParser):
        name = "fake"

        def supported_formats(self) -> Sequence[str]:
            return ["FAKE"]

        def detect_format(self, lines) -> Optional[str]:
            return "FAKE" if lines else None

        def parse_lines(self, lines, mrz_type=None, ocr_confidence=0.0):
            return MRZDocument(mrz_type="FAKE", parser=self.name)

        def parse_text(self, text, ocr_confidence=0.0):
            return self.parse_lines(text.splitlines())

    register_parser("fake", FakeParser)
    parser = create_parser("fake")
    assert parser.parse_lines(["anything"]).parser == "fake"
    assert "fake" in available_parsers()


# -------------------------------------------------------------- independence
def test_the_parser_imports_nothing_from_the_api_or_ocr_layers():
    """Requirement: the parser is a standalone library. Importing it must not
    drag in FastAPI, the OCR engines or the image pipeline."""
    code = (
        "import sys\n"
        "import app.services.mrz as m\n"
        "m.create_parser().parse_lines(["
        "'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<',"
        "'L898902C36UTO7408122F1204159ZE184226B<<<<<10'])\n"
        "banned = [n for n in sys.modules if n.startswith(("
        "'fastapi', 'starlette', 'app.api', 'app.main', 'app.schemas',"
        "'app.services.ocr', 'app.services.pipeline', 'app.services.layout',"
        "'app.services.image_processing'))]\n"
        "print(','.join(sorted(banned)))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert output.stdout.strip() == "", f"parser pulled in: {output.stdout.strip()}"


def test_the_parser_needs_no_image_libraries():
    """It takes text lines, so nothing here should require OpenCV or numpy."""
    code = (
        "import sys\n"
        "import app.services.mrz as m\n"
        "print('cv2' in sys.modules or 'numpy' in sys.modules)\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert output.stdout.strip() == "False"


# --------------------------------------------------------------- entry points
def test_parse_lines_accepts_already_recognised_text(parser):
    document = parser.parse_lines(ICAO_TD3_SPECIMEN)
    assert document.valid is True
    assert document.value("document_number") == "L898902C3"


def test_parse_lines_returns_none_for_non_zone_input(parser):
    assert parser.parse_lines(["Dear Sir or Madam,", "Please find enclosed."]) is None


def test_parse_lines_returns_none_for_empty_input(parser):
    assert parser.parse_lines([]) is None
    assert parser.parse_lines(["", "  "]) is None


def test_parse_text_finds_a_zone_inside_surrounding_text(parser):
    blob = "\n".join(
        ["REPUBLIC OF UTOPIA", "Given names / Prenoms", *ICAO_TD3_SPECIMEN]
    )
    document = parser.parse_text(blob)
    assert document.valid is True
    assert document.value("surname") == "ERIKSSON"


def test_parse_text_returns_none_when_there_is_no_zone(parser):
    assert parser.parse_text("An ordinary page of prose.\nSecond line here.") is None


def test_an_explicit_format_can_be_forced(parser, td3_lines):
    assert parser.parse_lines(td3_lines, mrz_type=TD3).mrz_type == TD3


def test_the_reference_date_is_injectable():
    """Century resolution must not depend on the day the tests run."""
    from datetime import date

    lines = make_lines(birth_date="050101")
    early = ICAOMRZParser(reference_date=date(2026, 9, 1))
    assert early.parse_lines(lines).value("birth_date") == "2005-01-01"
