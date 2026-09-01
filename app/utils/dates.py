"""Date parsing, century resolution and plausibility validation.

Everything the service emits is ISO ``YYYY-MM-DD``.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import List, Optional, Tuple

from app.utils.text import normalise_digits

ISO_FORMAT = "%Y-%m-%d"

# A passport is valid for at most 10 years anywhere in the world; allow slack.
MAX_PASSPORT_VALIDITY_YEARS = 15
MAX_HUMAN_AGE_YEARS = 125

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Arabic Gregorian month names as printed on Egyptian / Saudi passports.
_ARABIC_MONTHS = {
    "يناير": 1, "كانون الثاني": 1,
    "فبراير": 2, "شباط": 2,
    "مارس": 3, "اذار": 3, "آذار": 3,
    "ابريل": 4, "أبريل": 4, "نيسان": 4,
    "مايو": 5, "ايار": 5, "أيار": 5,
    "يونيو": 6, "يونية": 6, "حزيران": 6,
    "يوليو": 7, "يولية": 7, "تموز": 7,
    "اغسطس": 8, "أغسطس": 8, "اب": 8, "آب": 8,
    "سبتمبر": 9, "ايلول": 9, "أيلول": 9,
    "اكتوبر": 10, "أكتوبر": 10, "تشرين الاول": 10, "تشرين الأول": 10,
    "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الاول": 12, "كانون الأول": 12,
}

_HIJRI_MONTHS = {
    "محرم": 1, "صفر": 2, "ربيع الاول": 3, "ربيع الأول": 3, "ربيع الاخر": 4,
    "ربيع الآخر": 4, "ربيع الثاني": 4, "جمادى الاولى": 5, "جمادى الأولى": 5,
    "جمادى الاخرة": 6, "جمادى الآخرة": 6, "جمادى الثانية": 6, "رجب": 7,
    "شعبان": 8, "رمضان": 9, "شوال": 10, "ذو القعدة": 11, "ذوالقعدة": 11,
    "ذو الحجة": 12, "ذوالحجة": 12,
}

_NUMERIC_DATE = re.compile(
    r"(?<!\d)(\d{1,4})\s*[/\-.٫، ]\s*(\d{1,2})\s*[/\-.٫، ]\s*(\d{1,4})(?!\d)"
)
_DMY_TEXT = re.compile(r"(?<!\d)(\d{1,2})\s*[/\- ]?\s*([A-Z]{3,4})\s*[/\- ]?\s*(\d{2,4})(?!\d)")
_COMPACT = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")


def today() -> date:
    return date.today()


def to_iso(value: Optional[date]) -> Optional[str]:
    return value.strftime(ISO_FORMAT) if value else None


def safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


# --------------------------------------------------------------------- MRZ
def parse_mrz_date(
    raw: Optional[str],
    kind: str = "birth",
    reference: Optional[date] = None,
) -> Optional[date]:
    """Parse a 6 digit ``YYMMDD`` MRZ date and resolve its century.

    ``kind`` is one of ``birth``, ``expiry`` or ``issue`` and drives the
    plausibility window used to pick between 19xx and 20xx.
    """
    if not raw:
        return None
    digits = re.sub(r"\D", "", normalise_digits(raw))
    if len(digits) != 6:
        return None

    yy, mm, dd = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    ref = reference or today()
    candidates: List[date] = []
    for century in (1900, 2000):
        candidate = safe_date(century + yy, mm, dd)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    def plausible(value: date) -> bool:
        if kind == "birth":
            return value <= ref and (ref - value).days <= MAX_HUMAN_AGE_YEARS * 366
        if kind == "expiry":
            # Expired passports are a legitimate input, so allow the past too.
            return (
                value >= ref - timedelta(days=MAX_PASSPORT_VALIDITY_YEARS * 366)
                and value <= ref + timedelta(days=(MAX_PASSPORT_VALIDITY_YEARS + 1) * 366)
            )
        # issue
        return (
            value <= ref + timedelta(days=2)
            and value >= ref - timedelta(days=(MAX_PASSPORT_VALIDITY_YEARS + 5) * 366)
        )

    viable = [c for c in candidates if plausible(c)]
    if len(viable) == 1:
        return viable[0]
    if len(viable) > 1:
        # Ambiguous window: prefer the one closest to the reference date.
        return min(viable, key=lambda d: abs((d - ref).days))
    # Nothing plausible: fall back to the ICAO convention rather than失败.
    if kind == "birth":
        return safe_date(1900 + yy, mm, dd) or safe_date(2000 + yy, mm, dd)
    return safe_date(2000 + yy, mm, dd) or safe_date(1900 + yy, mm, dd)


def to_mrz_date(value: date) -> str:
    return value.strftime("%y%m%d")


# --------------------------------------------------------------------- OCR
def _resolve_two_digit_year(yy: int, kind: str, ref: date) -> int:
    if kind == "birth":
        return 1900 + yy if 1900 + yy <= ref.year else 2000 + yy
    return 2000 + yy if 2000 + yy >= ref.year - MAX_PASSPORT_VALIDITY_YEARS else 1900 + yy


def parse_free_date(
    text: Optional[str],
    kind: str = "birth",
    reference: Optional[date] = None,
) -> Optional[date]:
    """Parse a human readable date found by OCR on the passport page.

    Handles ``DD/MM/YYYY``, ``YYYY-MM-DD``, ``DD MON YYYY`` (English and
    Arabic month names), Arabic-Indic digits and ``DDMMYYYY``.
    """
    if not text:
        return None
    ref = reference or today()
    raw = normalise_digits(str(text)).strip().upper()

    # 1) Textual month, English.
    m = _DMY_TEXT.search(raw)
    if m:
        day, mon, year = int(m.group(1)), _MONTHS.get(m.group(2)), int(m.group(3))
        if mon:
            if year < 100:
                year = _resolve_two_digit_year(year, kind, ref)
            got = safe_date(year, mon, day)
            if got:
                return got

    # 2) Textual month, Arabic.
    lowered = str(text)
    for name, mon in _ARABIC_MONTHS.items():
        if name in lowered:
            nums = re.findall(r"\d{1,4}", normalise_digits(lowered))
            if len(nums) >= 2:
                day = int(nums[0])
                year = int(nums[-1])
                if year < 100:
                    year = _resolve_two_digit_year(year, kind, ref)
                got = safe_date(year, mon, day)
                if got:
                    return got

    # 3) Numeric separated.
    for m in _NUMERIC_DATE.finditer(raw):
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        got = _from_numeric_triplet(a, b, c, kind, ref)
        if got:
            return got

    # 4) Compact DDMMYYYY.
    m = _COMPACT.search(raw)
    if m:
        got = safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if got:
            return got

    return None


def _from_numeric_triplet(a: int, b: int, c: int, kind: str, ref: date) -> Optional[date]:
    # YYYY-MM-DD
    if a > 31:
        return safe_date(a, b, c)
    # DD/MM/YYYY - the only ordering used on Egyptian and Saudi passports.
    if c > 31:
        return safe_date(c, b, a)
    # All three are two digit: assume DD/MM/YY.
    year = _resolve_two_digit_year(c, kind, ref)
    return safe_date(year, b, a)


# ------------------------------------------------------------------- Hijri
def hijri_to_gregorian(hy: int, hm: int, hd: int) -> Optional[date]:
    """Tabular (Kuwaiti algorithm) Hijri -> Gregorian conversion.

    Accurate to about +/- 1 day against the Umm al-Qura calendar printed on
    Saudi passports. It is used only to cross-check a Gregorian date that was
    read elsewhere on the page - never as the value returned to the caller.
    """
    if not (1 <= hm <= 12 and 1 <= hd <= 30 and 1000 <= hy <= 1600):
        return None
    jd = (
        int((11 * hy + 3) / 30)
        + 354 * hy
        + 30 * hm
        - int((hm - 1) / 2)
        + hd
        + 1948440
        - 385
    )
    if jd > 2299160:
        ll = jd + 68569
        n = int((4 * ll) / 146097)
        ll = ll - int((146097 * n + 3) / 4)
        i = int((4000 * (ll + 1)) / 1461001)
        ll = ll - int((1461 * i) / 4) + 31
        j = int((80 * ll) / 2447)
        day = ll - int((2447 * j) / 80)
        ll = int(j / 11)
        month = j + 2 - 12 * ll
        year = 100 * (n - 49) + i + ll
    else:  # pragma: no cover - predates any passport
        ll = jd + 1402
        n = int((ll - 1) / 1461)
        i = ll - 1461 * n
        j = int((i - 1) / 365) - int(i / 1461)
        ll = i - 365 * j + 30
        i = int((80 * ll) / 2447)
        day = ll - int((2447 * i) / 80)
        ll = int(i / 11)
        month = i + 2 - 12 * ll
        year = 4 * n + j + ll - 4716
    return safe_date(year, month, day)


def looks_hijri(year: int) -> bool:
    return 1300 <= year <= 1550


def parse_hijri_free_date(text: Optional[str]) -> Optional[date]:
    """Parse a Hijri date string and return its Gregorian equivalent."""
    if not text:
        return None
    raw = normalise_digits(str(text))
    hm: Optional[int] = None
    for name, idx in _HIJRI_MONTHS.items():
        if name in raw:
            hm = idx
            break
    nums = [int(n) for n in re.findall(r"\d{1,4}", raw)]
    if hm is not None and len(nums) >= 2:
        return hijri_to_gregorian(nums[-1], hm, nums[0])
    if len(nums) >= 3:
        a, b, c = nums[0], nums[1], nums[2]
        if looks_hijri(c):
            return hijri_to_gregorian(c, b, a)
        if looks_hijri(a):
            return hijri_to_gregorian(a, b, c)
    return None


# --------------------------------------------------------------- validation
def validate_date_set(
    date_of_birth: Optional[date],
    date_of_expiry: Optional[date],
    date_of_issue: Optional[date],
    reference: Optional[date] = None,
) -> Tuple[List[str], List[str]]:
    """Cross-validate the date triplet.

    Returns ``(warnings, invalid_fields)``. A single implausible date never
    invalidates the others.
    """
    ref = reference or today()
    warnings: List[str] = []
    invalid: List[str] = []

    if date_of_birth:
        if date_of_birth > ref:
            warnings.append("date_of_birth is in the future")
            invalid.append("date_of_birth")
        elif (ref - date_of_birth).days > MAX_HUMAN_AGE_YEARS * 366:
            warnings.append("date_of_birth implies an implausible age")
            invalid.append("date_of_birth")

    if date_of_expiry and date_of_birth and date_of_expiry <= date_of_birth:
        warnings.append("date_of_expiry is not after date_of_birth")
        invalid.append("date_of_expiry")

    if date_of_issue:
        if date_of_issue > ref + timedelta(days=2):
            warnings.append("date_of_issue is in the future")
            invalid.append("date_of_issue")
        if date_of_expiry and date_of_issue >= date_of_expiry:
            warnings.append("date_of_issue is not before date_of_expiry")
            invalid.append("date_of_issue")
        if date_of_birth and date_of_issue < date_of_birth:
            warnings.append("date_of_issue precedes date_of_birth")
            invalid.append("date_of_issue")
        if (
            date_of_expiry
            and date_of_issue < date_of_expiry
            and (date_of_expiry - date_of_issue).days > MAX_PASSPORT_VALIDITY_YEARS * 366
        ):
            warnings.append("validity period exceeds 15 years")

    if date_of_expiry and date_of_expiry < ref:
        warnings.append("passport is expired")

    return warnings, sorted(set(invalid))
