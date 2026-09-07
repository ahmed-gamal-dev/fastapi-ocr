"""The optional visual-zone block on the OCR endpoint.

Every value here is invented; no real document data appears in this suite.
"""

from __future__ import annotations

from tests.conftest import block

LABEL_AR = "الاسم"
NAME_AR = "سالم بن محمد الحارثي"
PLACE_AR = "الدمام"
PLACE_EN = "DAMMAM"


def seed_page(client):
    client.stub.set_blocks(
        "en",
        [
            block("KINGDOM", 40, 40, 200, 26, 0.97),
            block(LABEL_AR, 900, 100, 80, 30, 0.95),
            block(NAME_AR, 560, 98, 330, 30, 0.97),
            block("مكان الإصدار", 900, 400, 180, 30, 0.92),
            block(f"{PLACE_EN}{PLACE_AR}", 905, 440, 170, 30, 0.94),
        ],
    )


def post(client, data, headers, **params):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", data, "image/png")},
        headers=headers,
        params=params or None,
    )


def test_fields_are_returned_when_requested(client, image_bytes, auth_headers):
    seed_page(client)

    body = post(client, image_bytes, auth_headers, viz="true").json()

    assert body["success"] is True
    viz = body["viz"]
    assert viz["name_ar"]["value"] == NAME_AR
    assert viz["issuing_authority"]["value"] == PLACE_EN
    assert viz["issuing_authority_ar"]["value"] == PLACE_AR
    assert 0.0 < viz["name_ar"]["confidence"] <= 1.0
    assert viz["name_ar"]["source"] in {"merged", "adjacent"}


def test_the_block_is_absent_unless_asked_for(client, image_bytes, auth_headers):
    seed_page(client)

    body = post(client, image_bytes, auth_headers).json()

    assert body["success"] is True
    assert body.get("viz") is None


def test_the_block_is_absent_when_no_field_was_found(client, image_bytes, auth_headers):
    """A page with no labels is a normal result, not an error or an empty shape."""
    client.stub.set_blocks("en", [block("JUST SOME TEXT", 40, 40, 300, 26, 0.96)])

    body = post(client, image_bytes, auth_headers, viz="true").json()

    assert body["success"] is True
    assert body.get("viz") is None


def test_page_text_is_unaffected_by_the_extraction(client, image_bytes, auth_headers):
    seed_page(client)

    body = post(client, image_bytes, auth_headers, viz="true").json()

    assert "KINGDOM" in body["text"]


def test_a_page_with_no_fields_costs_no_second_pass(client, image_bytes, auth_headers):
    """An invoice must not pay for a re-read that can only find nothing again.

    The stub answers identically whatever it is given, so the cost shows up as
    the number of recognitions rather than as a different result.
    """
    # Enough text that the orientation sweep stays out of the count, and no
    # zone requested, so the only pass that could be added is the viz re-read.
    client.stub.set_blocks(
        "en",
        [
            block("INVOICE NUMBER 4471 ISSUED THIS MONTH", 40, 40, 500, 26, 0.96),
            block("TOTAL DUE ON RECEIPT 1240.00", 40, 90, 460, 26, 0.95),
        ],
    )
    calls = {"n": 0}
    original = client.stub.recognize

    def counting(image, lang="en"):
        calls["n"] += 1
        return original(image, lang)

    client.stub.recognize = counting  # type: ignore[method-assign]
    post(client, image_bytes, auth_headers, viz="true", mrz="false")

    assert calls["n"] == 1, "no field and no zone means nothing to look harder for"


def test_values_never_reach_the_logs(client, image_bytes, auth_headers, caplog):
    """Only which fields were found is logged, never what they said."""
    seed_page(client)

    with caplog.at_level("INFO"):
        post(client, image_bytes, auth_headers, viz="true")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert NAME_AR not in logged
    assert PLACE_AR not in logged
