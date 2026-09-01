"""Standards-based machine-readable text parsing (ICAO Doc 9303).

A self-contained library. It takes **already-recognised text lines** and returns
a structured, per-field result. It imports nothing from the API layer, the OCR
engines or the image pipeline, and can be used on its own::

    from app.services.mrz import create_parser

    parser = create_parser("icao9303")
    document = parser.parse_lines([line_one, line_two], ocr_confidence=0.94)

    if document and document.valid:
        number = document.value("document_number")
        score = document.confidence_of("document_number")

Layers, bottom up:

``charset``    the zone alphabet and conservative glyph normalisation
``checkdigit`` the 7-3-1 check digit computation
``corrector``  type coercion and check-digit-guided repair
``parser``     the ICAO field layouts for TD1, TD2, TD3 and MRV
``textscan``   finding zone lines inside plain text
``icao``       the public parser, producing structured per-field results

``detector`` is the optional bridge that finds a zone in positioned OCR boxes.
It is the only module here that imports anything from the OCR layer, and
nothing else in this package depends on it.
"""

from __future__ import annotations

from app.services.mrz.base import (
    MRZParser,
    available_parsers,
    create_parser,
    register_parser,
)
from app.services.mrz.checkdigit import compute_check_digit, verify_check_digit
from app.services.mrz.document import MRZDocument
from app.services.mrz.fields import (
    BIRTH_DATE,
    CHECK_PROTECTED,
    DOCUMENT_CATEGORY,
    DOCUMENT_CODE,
    DOCUMENT_NUMBER,
    EXPIRY_DATE,
    FIELD_NAMES,
    GIVEN_NAMES,
    ISSUING_STATE,
    NATIONALITY,
    OPTIONAL_DATA,
    PERSONAL_NUMBER,
    SEX,
    SURNAME,
    ParsedField,
)

# Importing the implementation registers it with the factory.
from app.services.mrz.icao import ICAOMRZParser
from app.services.mrz.models import (
    FORMAT_SHAPES,
    MRV_A,
    MRV_B,
    TD1,
    TD2,
    TD3,
    CheckResult,
    MRZResult,
)
from app.services.mrz.parser import detect_format, parse
from app.services.mrz.synthetic import build_mrz
from app.services.mrz.textscan import looks_like_mrz_line, parse_text

__all__ = [
    # interface
    "MRZParser",
    "MRZDocument",
    "ParsedField",
    "ICAOMRZParser",
    "create_parser",
    "register_parser",
    "available_parsers",
    # field names
    "FIELD_NAMES",
    "CHECK_PROTECTED",
    "DOCUMENT_CODE",
    "DOCUMENT_CATEGORY",
    "ISSUING_STATE",
    "DOCUMENT_NUMBER",
    "NATIONALITY",
    "BIRTH_DATE",
    "SEX",
    "EXPIRY_DATE",
    "OPTIONAL_DATA",
    "PERSONAL_NUMBER",
    "SURNAME",
    "GIVEN_NAMES",
    # formats
    "TD1",
    "TD2",
    "TD3",
    "MRV_A",
    "MRV_B",
    "FORMAT_SHAPES",
    # lower level
    "MRZResult",
    "CheckResult",
    "parse",
    "parse_text",
    "detect_format",
    "looks_like_mrz_line",
    "compute_check_digit",
    "verify_check_digit",
    "build_mrz",
]
