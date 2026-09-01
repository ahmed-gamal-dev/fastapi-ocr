# Test fixtures

The test suite is **entirely synthetic**. Images are drawn with OpenCV at test
time by `tests/conftest.py`, and recognised text is scripted into the stub OCR
provider. No sample documents and no real data are stored in this repository.

## Trying your own images locally

Put them in `local/` — that path is git-ignored, so nothing you drop there can
be committed by accident:

```bash
mkdir -p tests/fixtures/local
cp ~/Desktop/my-scan.jpg tests/fixtures/local/
./scripts/smoke.sh http://127.0.0.1:8000 tests/fixtures/local/my-scan.jpg
```

Or against the running service directly:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ocr \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "image=@tests/fixtures/local/my-scan.jpg"
```

If you add a fixture to the committed suite, generate it in code (as
`tests/conftest.py::make_image` does) rather than checking in a binary.
