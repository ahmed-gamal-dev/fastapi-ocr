"""The optional MRZ block on the OCR endpoint.

The zone lines used here are the worked specimen published in ICAO Doc 9303
itself (`UTO` is the reserved code for the fictional state "Utopia"). No real
document data appears in this suite.
"""

from __future__ import annotations

import logging

from tests.conftest import block

SPECIMEN = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]


def seed_zone(client, lines=SPECIMEN, header: bool = True):
    """Script a page whose lower half holds a machine-readable zone."""
    blocks = []
    if header:
        blocks.append(block("REPUBLIC OF UTOPIA", 40, 60, 300, 26, 0.97))
    for index, line in enumerate(lines):
        blocks.append(block(line, 20, 500 + index * 30, 880, 22, 0.96))
    client.stub.set_blocks("en", blocks)


def post(client, data, headers, **params):
    return client.post(
        "/api/v1/ocr",
        files={"image": ("page.png", data, "image/png")},
        headers=headers,
        params=params or None,
    )


# ------------------------------------------------------------------ presence
def test_a_zone_is_parsed_when_requested(client, image_bytes, auth_headers):
    seed_zone(client)
    body = post(client, image_bytes, auth_headers, mrz="true").json()

    assert body["success"] is True
    mrz = body["mrz"]
    assert mrz["type"] == "TD3"
    assert mrz["valid"] is True
    assert mrz["check_digits_valid"] is True
    assert 0.0 < mrz["confidence"] <= 1.0


def test_the_block_is_absent_when_there_is_no_zone(client, image_bytes, auth_headers):
    """Absence means no zone was found - not an error, and not an empty shape."""
    body = post(client, image_bytes, auth_headers, mrz="true").json()
    assert body["success"] is True
    assert "mrz" not in body


def test_parsing_can_be_turned_off(client, image_bytes, auth_headers):
    seed_zone(client)
    body = post(client, image_bytes, auth_headers, mrz="false").json()
    assert "mrz" not in body


def test_the_rest_of_the_response_is_unchanged(client, image_bytes, auth_headers):
    """The block is additive: every existing field still behaves as before."""
    seed_zone(client)
    body = post(client, image_bytes, auth_headers, mrz="true").json()
    for key in ("text", "languages", "confidence", "lines", "image", "preprocessing"):
        assert key in body
    assert body["line_count"] == len(body["lines"])


# -------------------------------------------------------------------- fields
def test_fields_carry_value_validity_and_confidence(client, image_bytes, auth_headers):
    seed_zone(client)
    fields = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]["fields"]

    number = fields["document_number"]
    assert number["value"] == "L898902C3"
    assert number["valid"] is True          # confirmed by its check digit
    assert number["confidence"] > 0.9
    assert number["errors"] == []


def test_dates_are_iso_formatted(client, image_bytes, auth_headers):
    seed_zone(client)
    fields = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]["fields"]
    assert fields["birth_date"]["value"] == "1974-08-12"
    assert fields["expiry_date"]["value"] == "2012-04-15"


def test_unprotected_fields_report_unknown_validity(client, image_bytes, auth_headers):
    """The standard defines no check digit for the issuing state, so validity
    is unknown. Nulls are stripped from responses, so the key is absent - which
    is deliberately different from `false`, meaning "a check digit disagreed"."""
    seed_zone(client)
    fields = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]["fields"]
    assert fields["issuing_state"]["value"] == "UTO"
    assert "valid" not in fields["issuing_state"]
    # A protected field, by contrast, carries an explicit verdict.
    assert fields["document_number"]["valid"] is True


def test_names_are_split(client, image_bytes, auth_headers):
    seed_zone(client)
    mrz = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]
    assert mrz["fields"]["surname"]["value"] == "ERIKSSON"
    assert mrz["fields"]["given_names"]["value"] == "ANNA MARIA"
    assert mrz["full_name"] == "ANNA MARIA ERIKSSON"


def test_check_digit_verdicts_are_reported(client, image_bytes, auth_headers):
    seed_zone(client)
    checks = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]["check_digits"]
    assert checks["document_number"] is True
    assert checks["composite"] is True


# ----------------------------------------------------------------- raw output
def test_the_raw_zone_is_withheld_by_default(client, image_bytes, auth_headers):
    """It reproduces every field in one string, so callers must opt in."""
    seed_zone(client)
    mrz = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]
    assert "raw" not in mrz


def test_the_raw_zone_can_be_requested(client, image_bytes, auth_headers):
    seed_zone(client)
    mrz = post(
        client, image_bytes, auth_headers, mrz="true", include_mrz_raw="true"
    ).json()["mrz"]
    assert mrz["raw"].startswith("P<UTO")


# -------------------------------------------------------------------- damage
def test_a_failed_check_digit_is_reported_not_hidden(client, image_bytes, auth_headers):
    damaged = list(SPECIMEN)
    # Break the document number's check digit.
    damaged[1] = damaged[1][:9] + str((int(damaged[1][9]) + 5) % 10) + damaged[1][10:]
    seed_zone(client, damaged)

    mrz = post(client, image_bytes, auth_headers, mrz="true").json()["mrz"]
    assert mrz["valid"] is False
    assert mrz["check_digits"]["document_number"] is False
    assert mrz["fields"]["document_number"]["errors"]


def test_prose_is_not_reported_as_a_zone(client, image_bytes, auth_headers):
    client.stub.set_blocks(
        "en",
        [
            block("Dear customer, your order has shipped.", 40, 60, 400, 24, 0.95),
            block("Thank you for your business.", 40, 100, 380, 24, 0.95),
        ],
    )
    assert "mrz" not in post(client, image_bytes, auth_headers, mrz="true").json()


# ------------------------------------------------------------------- privacy
def test_zone_contents_never_reach_the_logs(client, image_bytes, auth_headers, caplog):
    seed_zone(client)
    with caplog.at_level(logging.DEBUG):
        body = post(client, image_bytes, auth_headers, mrz="true").json()

    assert body["mrz"]["fields"]["document_number"]["value"] == "L898902C3"
    logged = " ".join(record.getMessage() for record in caplog.records)
    for secret in ("L898902C3", "ERIKSSON", "740812", "P<UTO"):
        assert secret not in logged
