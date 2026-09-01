"""ICAO Doc 9303 MRZ parsers.

Supported formats:

===========  ==========  =============================================
Format       Shape       Used by
===========  ==========  =============================================
TD3          2 x 44      Passports
TD2          2 x 36      Older travel documents / official documents
TD1          3 x 30      Identity cards
MRV-A/MRV-B  2 x 44/36   Visas (no composite check digit)
===========  ==========  =============================================

Parsing is strictly structural: the layout tables below are transcriptions of
the standard, and every field is validated against its check digit where the
standard defines one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

from app.services.mrz.charset import (
    FILLER,
    normalize_line,
    pad_or_trim,
    strip_fillers,
)
from app.services.mrz.checkdigit import compute_check_digit, verify_check_digit
from app.services.mrz.corrector import (
    coerce_sex,
    correct_line,
    is_plausible_mrz_date,
    repair_date,
    repair_with_check_digit,
)
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

# ``(start, end, kind)`` slices used for type coercion, per line.
_LINE1_TRAVEL: Sequence[Tuple[int, int, str]] = ((0, 2, "doc_code"), (2, 5, "alpha"))
_LINE2_SPEC_HEAD: Sequence[Tuple[int, int, str]] = (
    (9, 10, "num"),
    (10, 13, "alpha"),
    (13, 19, "date"),
    (19, 20, "num"),
    (20, 21, "sex"),
    (21, 27, "date"),
    (27, 28, "num"),
)

DOCUMENT_CATEGORIES = {
    "P": "passport",
    "V": "visa",
    "I": "identity_card",
    "A": "identity_card",
    "C": "identity_card",
    "D": "identity_card",
}


def detect_format(lines: Sequence[str]) -> Optional[str]:
    """Infer the MRZ format from the line count and length."""
    lines = [ln for ln in lines if ln]
    if len(lines) == 3 and all(abs(len(ln) - 30) <= 4 for ln in lines):
        return TD1
    if len(lines) == 2:
        width = max(len(ln) for ln in lines)
        is_visa = lines[0][:1] == "V"
        if width >= 40:
            return MRV_A if is_visa else TD3
        if width >= 30:
            return MRV_B if is_visa else TD2
    return None


def parse(
    lines: Sequence[str],
    mrz_type: Optional[str] = None,
    ocr_confidence: float = 0.0,
) -> Optional[MRZResult]:
    """Parse candidate MRZ lines. Returns ``None`` if the shape is unusable."""
    cleaned = [normalize_line(ln) for ln in lines if normalize_line(ln)]
    if not cleaned:
        return None

    mrz_type = mrz_type or detect_format(cleaned)
    if mrz_type not in FORMAT_SHAPES:
        return None

    expected_lines, width = FORMAT_SHAPES[mrz_type]
    if len(cleaned) < expected_lines:
        return None
    cleaned = [pad_or_trim(ln, width) for ln in cleaned[:expected_lines]]

    if mrz_type == TD1:
        return _parse_td1(cleaned, ocr_confidence)
    return _parse_two_line(cleaned, mrz_type, ocr_confidence)


# --------------------------------------------------------------------- TD3/TD2
def _optional_layout(mrz_type: str) -> Tuple[int, int, bool, bool]:
    """``(optional_start, optional_end, has_optional_check, has_composite)``."""
    return {
        TD3: (28, 42, True, True),
        TD2: (28, 35, False, True),
        MRV_A: (28, 44, False, False),
        MRV_B: (28, 36, False, False),
    }[mrz_type]


def _parse_two_line(lines: List[str], mrz_type: str, ocr_confidence: float) -> MRZResult:
    width = FORMAT_SHAPES[mrz_type][1]
    opt_start, opt_end, has_opt_check, has_composite = _optional_layout(mrz_type)

    line1, c1 = correct_line(lines[0], _LINE1_TRAVEL)
    line2, c2 = correct_line(lines[1], _LINE2_SPEC_HEAD)
    corrections = ["line1:" + c for c in c1] + ["line2:" + c for c in c2]

    result = MRZResult(mrz_type=mrz_type, ocr_confidence=ocr_confidence)

    # ---- line 1
    document_code = line1[0:2].replace(FILLER, "")
    issuing_state = line1[2:5]
    surname, given_names = _parse_name(line1[5:width])

    # ---- line 2
    document_number = line2[0:9]
    document_number_cd = line2[9]
    nationality = line2[10:13]
    birth_date = line2[13:19]
    birth_cd = line2[19]
    sex = coerce_sex(line2[20])
    expiry_date = line2[21:27]
    expiry_cd = line2[27]
    optional_data = line2[opt_start:opt_end]
    optional_cd = line2[42] if has_opt_check else None

    # Document numbers longer than 9 characters overflow into the optional data
    # field (ICAO 9303 part 4, 4.2.2): the in-place check digit becomes a filler
    # and the full number plus its own check digit is stored in optional data.
    extended_number = False
    optional_value = optional_data
    if document_number_cd == FILLER and optional_data.strip(FILLER):
        recovered = _recover_long_document_number(document_number, optional_data)
        if recovered is not None:
            document_number, document_number_cd, optional_value = recovered
            extended_number = True
            corrections.append("document_number:extended_field")

    if not extended_number:
        document_number, fixed, ambiguous = repair_with_check_digit(
            document_number, document_number_cd
        )
        if fixed:
            corrections.append("document_number:check_digit_repair")
        if ambiguous:
            result.warnings.append(
                "document_number could not be corrected unambiguously"
            )

    birth_date, fixed, ambiguous = repair_date(birth_date, birth_cd)
    if fixed:
        corrections.append("birth_date:check_digit_repair")
    if ambiguous:
        result.warnings.append("birth_date could not be corrected unambiguously")

    expiry_date, fixed, ambiguous = repair_date(expiry_date, expiry_cd)
    if fixed:
        corrections.append("expiry_date:check_digit_repair")
    if ambiguous:
        result.warnings.append("expiry_date could not be corrected unambiguously")

    # Write the repaired values back before the checks are evaluated. The
    # composite digit is untouched OCR output, so recomputing it over the
    # repaired line is an independent confirmation of those repairs rather
    # than a circular one.
    line2 = _rebuild_line2(
        line2,
        None if extended_number else document_number,
        document_number_cd if not extended_number else None,
        birth_date,
        sex,
        expiry_date,
        width,
    )

    checks = [
        CheckResult(
            "document_number",
            document_number,
            document_number_cd,
            compute_check_digit(document_number),
            verify_check_digit(document_number, document_number_cd),
        ),
        CheckResult(
            "birth_date",
            line2[13:19],
            line2[19],
            compute_check_digit(line2[13:19]),
            verify_check_digit(line2[13:19], line2[19]),
        ),
        CheckResult(
            "expiry_date",
            line2[21:27],
            line2[27],
            compute_check_digit(line2[21:27]),
            verify_check_digit(line2[21:27], line2[27]),
        ),
    ]

    if has_opt_check:
        # The digit protects the optional field exactly as printed, which for an
        # extended document number still contains the number's overflow.
        printed_optional = line2[opt_start:opt_end]
        checks.append(
            CheckResult(
                "optional_data",
                printed_optional,
                optional_cd,
                compute_check_digit(printed_optional),
                verify_check_digit(printed_optional, optional_cd),
            )
        )

    if has_composite:
        composite_field = _composite_field(line2, mrz_type)
        composite_cd = line2[width - 1]
        checks.append(
            CheckResult(
                "composite",
                composite_field,
                composite_cd,
                compute_check_digit(composite_field),
                verify_check_digit(composite_field, composite_cd),
            )
        )

    result.lines = [line1, line2]
    result.document_code = document_code
    result.document_category = DOCUMENT_CATEGORIES.get(
        document_code[:1], "other" if document_code else "unknown"
    )
    result.issuing_state = strip_fillers(issuing_state)
    result.document_number = strip_fillers(document_number)
    result.nationality = strip_fillers(nationality)
    result.birth_date = birth_date
    result.sex = sex if sex in ("M", "F", "X") else None
    result.expiry_date = expiry_date
    result.optional_data = strip_fillers(optional_value)
    result.personal_number = result.optional_data if mrz_type == TD3 else ""
    result.surname = surname
    result.given_names = given_names
    result.checks = checks
    result.corrections = corrections
    result.structure_valid = _structure_ok(result)
    result.warnings.extend(_structure_warnings(result))
    return result


def _composite_field(line2: str, mrz_type: str) -> str:
    if mrz_type == TD3:
        return line2[0:10] + line2[13:20] + line2[21:43]
    # TD2
    return line2[0:10] + line2[13:20] + line2[21:35]


def _rebuild_line2(
    line2: str,
    document_number: Optional[str],
    document_number_cd: Optional[str],
    birth_date: str,
    sex: str,
    expiry_date: str,
    width: int,
) -> str:
    """Write repaired values back into the line, preserving everything else."""
    chars = list(pad_or_trim(line2, width))
    # ``None`` means the number lives in the extended field and its printed
    # positions must not be rewritten.
    if document_number is not None:
        chars[0:9] = list(pad_or_trim(document_number, 9))
    if document_number_cd is not None:
        chars[9] = document_number_cd or FILLER
    chars[13:19] = list(pad_or_trim(birth_date, 6))
    chars[20] = sex or FILLER
    chars[21:27] = list(pad_or_trim(expiry_date, 6))
    return "".join(chars)


def _recover_long_document_number(
    document_number: str, optional_data: str
) -> Optional[Tuple[str, str, str]]:
    """Rebuild a >9 character document number from the optional data field."""
    payload = optional_data.split(FILLER)[0]
    if len(payload) < 2:
        return None
    candidate = document_number.rstrip(FILLER) + payload[:-1]
    check = payload[-1]
    if not check.isdigit():
        return None
    if compute_check_digit(candidate) != check:
        return None
    remainder = optional_data[len(payload) :].lstrip(FILLER)
    return candidate, check, remainder


# ------------------------------------------------------------------------- TD1
_TD1_LINE1_SPEC: Sequence[Tuple[int, int, str]] = (
    (0, 2, "doc_code"),
    (2, 5, "alpha"),
    (14, 15, "num"),
)
_TD1_LINE2_SPEC: Sequence[Tuple[int, int, str]] = (
    (0, 6, "date"),
    (6, 7, "num"),
    (7, 8, "sex"),
    (8, 14, "date"),
    (14, 15, "num"),
    (15, 18, "alpha"),
    (29, 30, "num"),
)


def _parse_td1(lines: List[str], ocr_confidence: float) -> MRZResult:
    line1, c1 = correct_line(lines[0], _TD1_LINE1_SPEC)
    line2, c2 = correct_line(lines[1], _TD1_LINE2_SPEC)
    line3 = lines[2]
    corrections = ["line1:" + c for c in c1] + ["line2:" + c for c in c2]

    result = MRZResult(mrz_type=TD1, ocr_confidence=ocr_confidence)

    document_code = line1[0:2].replace(FILLER, "")
    issuing_state = line1[2:5]
    document_number = line1[5:14]
    document_number_cd = line1[14]
    optional_1 = line1[15:30]

    birth_date = line2[0:6]
    birth_cd = line2[6]
    sex = coerce_sex(line2[7])
    expiry_date = line2[8:14]
    expiry_cd = line2[14]
    nationality = line2[15:18]
    optional_2 = line2[18:29]
    composite_cd = line2[29]

    if document_number_cd == FILLER and optional_1.strip(FILLER):
        recovered = _recover_long_document_number(document_number, optional_1)
        if recovered is not None:
            document_number, document_number_cd, optional_1 = recovered
            corrections.append("document_number:extended_field")

    document_number, fixed, ambiguous = repair_with_check_digit(
        document_number, document_number_cd
    )
    if fixed:
        corrections.append("document_number:check_digit_repair")
    if ambiguous:
        result.warnings.append("document_number could not be corrected unambiguously")

    birth_date, fixed, _ = repair_date(birth_date, birth_cd)
    if fixed:
        corrections.append("birth_date:check_digit_repair")
    expiry_date, fixed, _ = repair_date(expiry_date, expiry_cd)
    if fixed:
        corrections.append("expiry_date:check_digit_repair")

    composite_field = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    checks = [
        CheckResult(
            "document_number",
            document_number,
            document_number_cd,
            compute_check_digit(document_number),
            verify_check_digit(document_number, document_number_cd),
        ),
        CheckResult(
            "birth_date",
            birth_date,
            birth_cd,
            compute_check_digit(birth_date),
            verify_check_digit(birth_date, birth_cd),
        ),
        CheckResult(
            "expiry_date",
            expiry_date,
            expiry_cd,
            compute_check_digit(expiry_date),
            verify_check_digit(expiry_date, expiry_cd),
        ),
        CheckResult(
            "composite",
            composite_field,
            composite_cd,
            compute_check_digit(composite_field),
            verify_check_digit(composite_field, composite_cd),
        ),
    ]

    surname, given_names = _parse_name(line3)

    result.lines = [line1, line2, line3]
    result.document_code = document_code
    result.document_category = DOCUMENT_CATEGORIES.get(
        document_code[:1], "other" if document_code else "unknown"
    )
    result.issuing_state = strip_fillers(issuing_state)
    result.document_number = strip_fillers(document_number)
    result.nationality = strip_fillers(nationality)
    result.birth_date = birth_date
    result.sex = sex if sex in ("M", "F", "X") else None
    result.expiry_date = expiry_date
    result.optional_data = " ".join(
        p for p in (strip_fillers(optional_1), strip_fillers(optional_2)) if p
    )
    result.surname = surname
    result.given_names = given_names
    result.checks = checks
    result.corrections = corrections
    result.structure_valid = _structure_ok(result)
    result.warnings.extend(_structure_warnings(result))
    return result


# ------------------------------------------------------------------- shared
def _parse_name(field: str) -> Tuple[str, str]:
    """Split an MRZ name field into ``(surname, given_names)``.

    The primary and secondary identifiers are separated by ``<<``; single
    fillers separate words. A field with no ``<<`` is treated as a surname
    only, which is what the standard prescribes for single-identifier names.
    """
    raw = (field or "").rstrip(FILLER)
    if not raw:
        return "", ""
    if "<<" in raw:
        primary, _, secondary = raw.partition("<<")
    else:
        primary, secondary = raw, ""
    surname = " ".join(p for p in primary.split(FILLER) if p)
    given = " ".join(p for p in secondary.split(FILLER) if p)
    return surname.strip(), given.strip()


def _structure_ok(result: MRZResult) -> bool:
    """Structural sanity, independent of the check digits."""
    if not result.document_code or not result.document_code[0].isalpha():
        return False
    if len(result.issuing_state) != 3 or not result.issuing_state.isalpha():
        return False
    if not result.document_number:
        return False
    if not is_plausible_mrz_date(result.birth_date):
        return False
    if not is_plausible_mrz_date(result.expiry_date):
        return False
    return True


def _structure_warnings(result: MRZResult) -> List[str]:
    warnings: List[str] = []
    if len(result.nationality) != 3 or not result.nationality.isalpha():
        warnings.append("nationality is not a 3-letter code")
    if result.sex is None:
        warnings.append("sex is unspecified or unreadable")
    if not result.surname and not result.given_names:
        warnings.append("name field is empty")
    for check in result.checks:
        if check.valid is False:
            warnings.append(f"{check.name} check digit mismatch")
    return warnings


def build_line(fields: Dict[str, str], mrz_type: str = TD3) -> List[str]:
    """Build a syntactically valid MRZ from plain field values.

    Used by the test-suite to generate synthetic fixtures - it is the inverse of
    :func:`parse` and never touches real documents.
    """
    from app.services.mrz.charset import name_to_mrz

    width = FORMAT_SHAPES[mrz_type][1]
    document_code = pad_or_trim(fields.get("document_code", "P"), 2)
    issuing_state = pad_or_trim(fields.get("issuing_state", "UTO"), 3)
    name = name_to_mrz(fields.get("surname", "")) + "<<" + name_to_mrz(
        fields.get("given_names", "")
    )
    line1 = pad_or_trim(document_code + issuing_state + name, width)

    document_number = pad_or_trim(fields.get("document_number", ""), 9)
    birth_date = pad_or_trim(fields.get("birth_date", ""), 6)
    expiry_date = pad_or_trim(fields.get("expiry_date", ""), 6)
    nationality = pad_or_trim(fields.get("nationality", "UTO"), 3)
    sex = (fields.get("sex") or FILLER)[:1]
    opt_start, opt_end, has_opt_check, has_composite = _optional_layout(mrz_type)
    optional = pad_or_trim(fields.get("optional_data", ""), opt_end - opt_start)

    line2 = (
        document_number
        + compute_check_digit(document_number)
        + nationality
        + birth_date
        + compute_check_digit(birth_date)
        + sex
        + expiry_date
        + compute_check_digit(expiry_date)
        + optional
    )
    if has_opt_check:
        line2 += compute_check_digit(optional)
    if has_composite:
        line2 = pad_or_trim(line2, width - 1)
        line2 += compute_check_digit(_composite_field(pad_or_trim(line2, width), mrz_type))
    return [line1, pad_or_trim(line2, width)]
