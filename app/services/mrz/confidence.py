"""Confidence scoring for machine-readable zone fields.

Confidence here means one thing only: how much the *standard's own redundancy*
supports the value that was read, adjusted for how well the text was recognised
in the first place. It is not a probability, and it is never used to decide
whether to emit a value - an unreadable field is dropped regardless of score.

The scale:

===================================  =======  ==============================
Situation                            Score    Reasoning
===================================  =======  ==============================
Own check digit agrees               0.98     Independently confirmed
Own check digit disagrees            0.20     Actively contradicted
No check digit, composite agrees     0.85     Covered by the composite digit
No check digit, composite disagrees  0.55     Something in the zone is wrong
No check digit, no composite         0.70     Structurally sound, unverified
===================================  =======  ==============================
"""

from __future__ import annotations

from typing import List, Optional

CHECK_PASSED = 0.98
CHECK_FAILED = 0.20
UNPROTECTED_COMPOSITE_OK = 0.85
UNPROTECTED_COMPOSITE_FAILED = 0.55
UNPROTECTED_NO_COMPOSITE = 0.70

#: Deduction for a field the corrector had to repair. The check digit still had
#: to agree afterwards, so the penalty is modest rather than disqualifying.
CORRECTION_PENALTY = 0.08

#: How much recognition quality can pull a score down. At OCR confidence 0 the
#: score keeps 75% of its structural value; at 1.0 it keeps all of it.
OCR_FLOOR = 0.75


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def blend_ocr(score: float, ocr_confidence: float) -> float:
    """Temper a structural score with how well the text was recognised."""
    if ocr_confidence <= 0.0:
        # No recognition confidence was supplied; report the structural score.
        return clamp(score)
    return clamp(score * (OCR_FLOOR + (1.0 - OCR_FLOOR) * clamp(ocr_confidence)))


def field_confidence(
    *,
    present: bool,
    check_valid: Optional[bool],
    composite_valid: Optional[bool],
    corrected: bool = False,
    ocr_confidence: float = 0.0,
) -> float:
    """Score one field. An absent field scores zero, never a default."""
    if not present:
        return 0.0

    if check_valid is True:
        score = CHECK_PASSED
    elif check_valid is False:
        score = CHECK_FAILED
    elif composite_valid is True:
        score = UNPROTECTED_COMPOSITE_OK
    elif composite_valid is False:
        score = UNPROTECTED_COMPOSITE_FAILED
    else:
        score = UNPROTECTED_NO_COMPOSITE

    if corrected:
        score -= CORRECTION_PENALTY
    return blend_ocr(score, ocr_confidence)


def document_confidence(
    *,
    structure_valid: bool,
    check_digit_score: float,
    corrections: Optional[List[str]] = None,
    ocr_confidence: float = 0.0,
) -> float:
    """Score the zone as a whole.

    A zone that fails structural validation is capped low no matter how many
    individual digits happen to agree.
    """
    check_digit_score = clamp(check_digit_score)
    if not structure_valid:
        return clamp(min(0.25, 0.25 * check_digit_score))

    score = 0.45 + 0.55 * check_digit_score
    if corrections:
        score -= 0.05 * min(len(corrections), 3)
    return blend_ocr(score, ocr_confidence)
