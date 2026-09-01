# Document OCR Service

A self-contained OCR microservice: upload an image, get back structured text
with per-line geometry and confidence scores.

Everything runs locally on open-source components. There are no paid OCR APIs,
no external calls, and no cloud dependencies of any kind.

```
FastAPI  ->  OpenCV preprocessing  ->  OCRProvider (PaddleOCR)  ->  layout analysis  ->  JSON
```

> **Engine status.** Generic OCR infrastructure and parser are tested.
> PaddleOCR package initialization was verified, but inference is
> environment-dependent and must be validated on the target Linux server.
>
> The service ships with two providers. `OCR_PROVIDER=stub` runs everything -
> API, pipeline, layout, tests - without loading any model, and never imports
> PaddleOCR at all. `OCR_PROVIDER=paddle` is the production engine. Switching
> is one line in `.env`; see [Swapping the OCR engine](#swapping-the-ocr-engine)
> and the [deployment guide](#production-deployment-ubuntu-2204--2404-no-docker).

---

## Table of contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [First run and model download](#first-run-and-model-download)
5. [Configuration](#configuration)
6. [API reference](#api-reference)
7. [Examples](#examples)
8. [Architecture](#architecture)
9. [Swapping the OCR engine](#swapping-the-ocr-engine)
10. [Security and privacy](#security-and-privacy)
11. [Error codes](#error-codes)
12. [Testing](#testing)
13. [Performance and concurrency](#performance-and-concurrency)
14. [Production deployment (Ubuntu, no Docker)](#production-deployment-ubuntu-2204--2404-no-docker)
15. [Deployment checklist](#deployment-checklist)
16. [Troubleshooting](#troubleshooting)
17. [Project layout](#project-layout)

---

## What it does

* Accepts an image upload and returns every piece of text it can read.
* Groups recognition boxes into **lines** (in reading order, right-to-left aware)
  and optionally into paragraph-like **regions**.
* Reports a **confidence score** per block, per line and for the page overall.
* Preprocesses with OpenCV: perspective correction, downscaling, deskewing and
  contrast enhancement, reporting exactly which steps it applied.
* Supports multiple languages in one pass (English and Arabic are configured by
  default; any PaddleOCR language works).
* Never invents text. If nothing is readable, you get an empty result and a
  warning, not a guess.

---

## Requirements

| | |
|---|---|
| Python | 3.11+ recommended (3.9+ supported; verified on 3.9.6) |
| RAM | 2 GB minimum, 4 GB recommended per worker |
| Disk | ~230 MB for the English + Arabic models |
| OS | Linux or macOS (verified on macOS arm64) |

No system packages are needed beyond a working Python. `opencv-python-headless`
avoids the X11/GUI libraries that the standard OpenCV wheel pulls in.

---

## Quick start

```bash
git clone <your-repo> ocr && cd ocr
DEV=1 ./scripts/setup.sh      # creates .venv, installs deps, writes .env with a generated API key
./scripts/run.sh              # starts on http://127.0.0.1:8000
```

In another terminal:

```bash
./scripts/smoke.sh
```

Doing it by hand instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OCR_API_KEY in .env, then:
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs are at <http://127.0.0.1:8000/docs> when `DOCS_ENABLED=true`.

### Scripts

| Script | Purpose |
|---|---|
| `./scripts/setup.sh` | Create the virtualenv, install dependencies, seed `.env`. `DEV=1` also installs test dependencies. |
| `./scripts/run.sh` | Start the service. `--reload` for development. |
| `./scripts/test.sh` | Run the test suite. Arguments pass through to pytest. |
| `./scripts/smoke.sh` | End-to-end check against a running instance. |

---

## First run and model download

PaddleOCR downloads its detection and recognition models the first time each
language is used, cached in `~/.paddlex/official_models`. With
`OCR_WARMUP_ON_STARTUP=true` (the default) this happens during startup, so the
**first boot on a new machine needs network access and takes a few minutes**.
Every boot after that is fast and fully offline.

Measured model sizes for the default configuration:

| Model | Role | Size |
|---|---|---|
| `PP-OCRv6_medium_det` | English detection | 59 MB |
| `PP-OCRv6_medium_rec` | English recognition | 73 MB |
| `PP-OCRv5_server_det` | Arabic detection | 84 MB |
| `arabic_PP-OCRv5_mobile_rec` | Arabic recognition | 7.8 MB |
| `PP-LCNet_x1_0_textline_ori` | Text-line orientation | 6.6 MB |
| | **Total (both languages)** | **231 MB** |

To pre-download without starting the server:

```bash
.venv/bin/python -c "
from app.services.ocr.registry import create_provider
create_provider('paddle').warmup(['en', 'arabic'])
"
```

To run the API without the OCR models at all — useful for integration work on
the client side — set `OCR_PROVIDER=stub`. The stub returns only text it was
explicitly scripted with, so it can never fabricate a result.

---

## Configuration

Everything is an environment variable, read from `.env` or the process
environment. See [`.env.example`](.env.example) for the annotated full list.

The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `OCR_API_KEY` | *(empty)* | **Set this.** Comma separated for zero-downtime rotation. Empty disables auth and logs a warning. |
| `OCR_PROVIDER` | `paddle` | `paddle` or `stub`. |
| `OCR_LANGUAGES` | `en,arabic` | Comma separated. Each language is a separate model in memory. |
| `OCR_MAX_CONCURRENCY` | `1` | Concurrent inferences per worker. See [Performance](#performance-and-concurrency). |
| `OCR_TIMEOUT_SECONDS` | `45` | Per-inference ceiling. |
| `MAX_UPLOAD_SIZE` | `10485760` | Bytes. Enforced twice: on `Content-Length` and while streaming. |
| `MAX_IMAGE_PIXELS` | `50000000` | Decompression-bomb guard. |
| `RATE_LIMIT_REQUESTS` | `60` | Per `RATE_LIMIT_WINDOW_SECONDS`, per API key. |
| `REDIS_URL` | *(unset)* | Optional. Without it the limiter is per-process. |
| `WORKERS` | `1` | Each worker loads its own copy of the models. |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | |
| `STORE_UPLOADS` | `false` | Images are processed in memory and discarded unless you turn this on. |

Lists accept both `a,b,c` and `["a","b","c"]`.

---

## API reference

### `POST /api/v1/ocr`

Recognise text in an image. Requires authentication.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | **Required.** JPEG, PNG, WebP, BMP or TIFF. |

**Query parameters** — all optional

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `languages` | string | `OCR_LANGUAGES` | Comma separated, e.g. `en,arabic`. Aliases like `ar` and `english` are accepted. |
| `preprocess` | bool | `true` | Run the OpenCV chain. |
| `detect_orientation` | bool | config | Retry other page rotations when the first pass finds almost nothing. |
| `include_blocks` | bool | `true` | Include every raw recognition box. |
| `include_regions` | bool | `false` | Include paragraph-like line groupings. |
| `min_confidence` | float | `0.0` | Drop boxes below this score. |

**Authentication** — either header works:

```
X-API-Key: <key>
Authorization: Bearer <key>
```

**Response `200`**

```json
{
  "success": true,
  "request_id": "f4ced2f6bfcb444ebb9753246fa0c061",
  "text": "INVOICE 2026-09\nTOTAL 1240.00",
  "languages": ["en"],
  "confidence": { "mean": 0.94, "min": 0.88, "max": 0.99 },
  "line_count": 2,
  "word_count": 4,
  "lines": [
    {
      "text": "INVOICE 2026-09",
      "confidence": 0.97,
      "min_confidence": 0.95,
      "languages": ["en"],
      "bbox": { "x": 40, "y": 40, "width": 250, "height": 22 }
    }
  ],
  "blocks": [
    {
      "text": "INVOICE",
      "confidence": 0.99,
      "lang": "en",
      "bbox": { "x": 40, "y": 40, "width": 120, "height": 22 },
      "polygon": [[40, 40], [160, 40], [160, 62], [40, 62]]
    }
  ],
  "image": {
    "format": "image/png",
    "size_bytes": 84213,
    "original_width": 1100,
    "original_height": 700,
    "processed_width": 1100,
    "processed_height": 700
  },
  "preprocessing": {
    "steps": ["deskew", "enhance"],
    "scale": 1.0,
    "rotation": 0,
    "skew_angle": -0.8,
    "perspective_corrected": false
  },
  "timings_ms": { "decode_ms": 12.1, "preprocess_ms": 96.0, "ocr_ms": 812.4, "layout_ms": 0.3 },
  "processing_time_ms": 921.7,
  "warnings": []
}
```

Notes on the response:

* **Coordinates** are in the *processed* image space. When preprocessing rescales
  or warps the page, `processed_width`/`processed_height` and `preprocessing`
  tell you the relationship to the original.
* **`languages`** lists the models that were run, even if one contributed no text.
* **`warnings`** is where partial success is reported: low confidence, dropped
  blocks, a language that failed, or nothing readable at all. A request that
  produces no text is still a `200` — an unreadable image is a result, not an error.

### `GET /health`

Liveness. No authentication. Deliberately does not touch the OCR engine, so a
slow model load cannot make a healthy process look dead.

```json
{ "status": "ok" }
```

### `GET /ready`

Readiness. No authentication. Returns `503` while the models are still loading.

```json
{ "status": "ready", "ocr_ready": true, "provider": "paddleocr", "languages": ["en", "arabic"] }
```

### `GET /api/v1/version`

Version, engine state and the effective limits. No authentication.

```json
{
  "name": "document-ocr-service",
  "version": "1.0.0",
  "api_version": "v1",
  "environment": "production",
  "ocr": { "provider": "paddleocr", "ready": true, "languages": ["en", "arabic"], "concurrency": 1 },
  "limits": { "max_upload_size": 10485760, "rate_limit_requests": 60 }
}
```

---

## Examples

```bash
# Basic
curl -X POST http://127.0.0.1:8000/api/v1/ocr \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "image=@page.jpg"

# Arabic + English, paragraphs, no raw boxes
curl -X POST "http://127.0.0.1:8000/api/v1/ocr?languages=en,arabic&include_regions=true&include_blocks=false" \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "image=@page.jpg"

# Only high-confidence text, no preprocessing
curl -X POST "http://127.0.0.1:8000/api/v1/ocr?min_confidence=0.8&preprocess=false" \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "image=@page.jpg"

# Just the plain text
curl -s -X POST http://127.0.0.1:8000/api/v1/ocr \
  -H "X-API-Key: $OCR_API_KEY" \
  -F "image=@page.jpg" | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])"
```

Python:

```python
import requests

with open("page.jpg", "rb") as fh:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/ocr",
        headers={"X-API-Key": "..."},
        files={"image": ("page.jpg", fh, "image/jpeg")},
        params={"languages": "en,arabic"},
        timeout=60,
    )

response.raise_for_status()
result = response.json()
print(result["text"])
for line in result["lines"]:
    print(f"{line['confidence']:.2f}  {line['text']}")
```

---

## Architecture

```
                  HTTP request
                       |
        RequestContextMiddleware      request id, timing
        MaxBodySizeMiddleware         Content-Length guard
        SecurityHeadersMiddleware     nosniff, no-store, ...
                       |
        require_api_key -> enforce_rate_limit
                       |
              POST /api/v1/ocr
                       |
         streaming read with a byte ceiling
                       |
   +---------------- pipeline -----------------+
   |  loader      validate + decode (in memory)|
   |  preprocess  perspective, scale, deskew,  |
   |              enhance                      |
   |  engine      OCRProvider per language     |
   |  layout      lines, reading order, regions|
   +-------------------------------------------+
                       |
              Pydantic response model
```

Design decisions worth knowing:

* **Models load once.** `OCREngine.startup()` warms every configured language
  during application startup and the provider instances live for the process
  lifetime. Nothing loads a model per request.
* **Inference runs in a bounded thread pool.** PaddleOCR inference is blocking
  C++ work, so it cannot run on the event loop. The pool is deliberately small
  (`OCR_MAX_CONCURRENCY`) because each concurrent inference multiplies peak
  memory and oversubscribes the CPU.
* **Languages run sequentially** within one request, for the same reason.
* **Nothing touches the disk.** Upload bytes go straight from the request into a
  numpy array. `app/utils/files.py` provides secure temp-file handling for
  callers that need it (random names, `0600`, overwrite-then-unlink) and an
  opt-in store behind `STORE_UPLOADS`.
* **Failure is partial, not total.** A language that fails, a block below
  threshold or an unreadable page all produce a `200` with a warning. Only a
  genuinely unusable request is an error.

---

## Swapping the OCR engine

Nothing outside `app/services/ocr/` knows that PaddleOCR exists. To add an
engine, implement `OCRProvider` and register it:

```python
from app.services.ocr.base import OCRProvider, OCRResult, TextBlock
from app.services.ocr.registry import register_provider


class MyEngineProvider(OCRProvider):
    name = "myengine"

    def supported_languages(self):
        return ["en"]

    def warmup(self, languages=None):
        self._model = load_my_model()      # called once, at startup

    def is_ready(self):
        return getattr(self, "_model", None) is not None

    def recognize(self, image, lang="en"):  # image is a BGR numpy array
        blocks = [
            TextBlock(text=item.text, confidence=item.score, polygon=item.quad, lang=lang)
            for item in self._model.run(image)
        ]
        return OCRResult(blocks=blocks, lang=lang, provider=self.name)


register_provider("myengine", MyEngineProvider)
```

Then set `OCR_PROVIDER=myengine`. The API, pipeline, schemas and tests are
unchanged. `tests/test_ocr_provider.py::test_a_new_provider_can_be_registered`
exercises exactly this path.

The PaddleOCR adapter introspects the installed version's constructor and call
signatures, so it works across the 2.x and 3.x APIs without pinning you to one.

---

## Security and privacy

* **API key authentication** on the OCR endpoint, compared in constant time,
  supporting multiple keys for rotation. Probes stay unauthenticated.
* **Rate limiting** per key (falling back to client address), in-process by
  default or shared through Redis. A Redis outage degrades to the in-process
  limiter rather than taking the service down.
* **Upload validation**: size checked against `Content-Length` *and* while
  streaming, type identified by **file signature** rather than the client's
  `Content-Type`, extension allow-list, pixel-count ceiling and a minimum
  dimension check.
* **The client's filename is never used** to name anything. Temp files get
  random 32-hex names, `0600` permissions, and are overwritten before being
  unlinked. Orphans from a crashed worker are swept at startup.
* **Images are not persisted** unless `STORE_UPLOADS=true` is set deliberately.
* **Logs never contain document content.** The formatter redacts a list of
  sensitive keys, summarises raw bytes as `<n bytes>`, and scrubs long
  machine-readable runs out of messages. Recognised text, client filenames and
  API keys are never logged. `LOG_SENSITIVE_DATA` can lift this for local
  debugging and warns loudly at startup when it is on.
* **No stack traces leave the process.** Unhandled exceptions are logged with
  their traceback and answered with a generic `PROCESSING_ERROR`.
* **CORS is off by default.** Set `ALLOWED_ORIGINS` only if a browser calls the
  API directly; server-to-server callers do not need it.
* Responses are `Cache-Control: no-store`, plus `nosniff`, `DENY` and
  `no-referrer`.

If you put this behind a reverse proxy, set `TRUST_PROXY_HEADERS=true` — but
only then, or `X-Forwarded-For` becomes a way to walk around the rate limiter.

---

## Error codes

Every failure returns the same envelope:

```json
{
  "success": false,
  "error": { "code": "UNSUPPORTED_FORMAT", "message": "Image type 'application/pdf' is not allowed" },
  "request_id": "9e9f9fcb2fc741a3bc63b3eb3a16a652"
}
```

| Code | HTTP | Meaning |
|---|---|---|
| `MISSING_FILE` | 422 | No `image` field in the request |
| `INVALID_IMAGE` | 422 | The file could not be decoded |
| `UNSUPPORTED_FORMAT` | 415 | Not an allowed image type |
| `IMAGE_TOO_LARGE` | 413 | Over `MAX_UPLOAD_SIZE` or `MAX_IMAGE_PIXELS` |
| `IMAGE_TOO_SMALL` | 422 | Below the minimum usable dimensions |
| `UNAUTHORIZED` | 401 | Missing or wrong API key |
| `RATE_LIMITED` | 429 | Over the rate limit; see the `Retry-After` header |
| `OCR_FAILED` | 500 | The engine failed |
| `OCR_TIMEOUT` | 504 | Inference exceeded `OCR_TIMEOUT_SECONDS` |
| `SERVICE_UNAVAILABLE` | 503 | The engine is not loaded yet |
| `PROCESSING_ERROR` | 500 | Anything unexpected |

`request_id` is echoed in the `X-Request-ID` header and appears in every log
line for that request — quote it when reporting a problem.

---

## Testing

```bash
./scripts/test.sh                      # everything
./scripts/test.sh --cov=app            # with coverage
./scripts/test.sh tests/test_layout.py -v
```

The default suite runs against the scripted **stub provider**, so it needs no
OCR models and no network. Images are drawn with OpenCV at test time. There are
no sample documents in the repository and no real data of any kind.

`tests/integration/` is different: it drives the **real PaddleOCR engine**
end to end - English, Arabic, digits, dates, mixed Arabic/English, rotated
pages, bounding boxes, confidence and reading order, plus the HTTP endpoint.
Those tests are skipped automatically when the engine is not installed, and are
still built entirely from generated images:

```bash
./scripts/test.sh tests/integration          # real engine (slow, ~4 min)
./scripts/test.sh -m "not integration"       # skip them
```

Arabic pages are rendered with `arabic-reshaper` + `python-bidi` so the glyphs
are joined and reordered the way a real page prints them; without that the
engine is being asked to read something no document looks like.

Covered: the endpoint contract, API key authentication, rate limiting, upload
validation (size, type, signature, pixel ceiling), image preprocessing
(downscale, rotate, deskew convergence, perspective, enhance), layout analysis
(line grouping, reading order, RTL, regions, deduplication, confidence),
the provider abstraction and both PaddleOCR output shapes, engine lifecycle
(load-once, concurrency ceiling, timeout, per-language failure isolation),
error handling and the no-leak guarantee, log redaction, and temp-file cleanup.

To try real images, drop them in `tests/fixtures/local/` — that path is
git-ignored — and point the smoke script at one:

```bash
./scripts/smoke.sh http://127.0.0.1:8000 tests/fixtures/local/my-page.jpg
```

---

## Performance and concurrency

OCR is CPU-bound and memory-hungry. The defaults are tuned for correctness
under load rather than for a single fast request.

The figures below were observed on a development machine (Apple M-series,
10 cores, `OCR_CPU_THREADS=1`, a 1200x410 page) and are indicative only.
**Inference performance is environment-dependent: measure it on your own
server before sizing anything.** Step 4 of the deployment guide walks through
validating the engine on the target host.

| Configuration | Time per page | Steady RSS |
|---|---|---|
| English, default models | ~3.1 s | 1454 MB |
| English, **mobile models** | ~1.0 s | 922 MB |
| Arabic, mobile models | ~0.77 s | — |
| English + Arabic, mobile models, both loaded | — | 791 MB |
| English + Arabic, default models, both passes | ~9.6-12.4 s | 2822 MB |

Two things follow from that table:

* **The mobile models are worth it.** Roughly 3x faster and a third less
  memory, for 0.994 vs 0.998 mean confidence on clean synthetic text. Set
  `OCR_DET_MODEL_NAME` / `OCR_REC_MODEL_NAME` (see below).
* **Each configured language is a separate full pass over the image.** Running
  `en,arabic` costs about the sum of both, so list only what you need and let
  callers narrow it further with `?languages=`.

`OCR_CPU_THREADS` past 1 made little difference here (2.8 s vs 3.1 s): the
model, not thread count, is the bottleneck. Keep it at 1 when requests overlap.

**Recognition models are script-specific.** A bare
`OCR_REC_MODEL_NAME=PP-OCRv5_mobile_rec` is a Latin model and would be applied
to every language, making Arabic pages come back as transliterated nonsense at
~0.58 confidence. With more than one language, use the per-language form:

```bash
OCR_DET_MODEL_NAME=PP-OCRv5_mobile_det
OCR_REC_MODEL_NAME=en:PP-OCRv5_mobile_rec,arabic:arabic_PP-OCRv5_mobile_rec
```

**Memory.** Roughly 400-700 MB per loaded language model, plus the working set
for the image. RSS plateaus after the first few requests - it does not grow
unbounded.

**The important arithmetic:** every uvicorn worker loads its *own* copy of every
model. Four workers with two languages is eight models in RAM. More workers is
not automatically better.

| Machine | Recommended |
|---|---|
| 2 vCPU / 4 GB | `WORKERS=1`, `OCR_MAX_CONCURRENCY=1` |
| 4 vCPU / 8 GB | `WORKERS=2`, `OCR_MAX_CONCURRENCY=1` |
| 8 vCPU / 16 GB | `WORKERS=3`, `OCR_MAX_CONCURRENCY=2` |

Rules of thumb:

* Start with `WORKERS=1` and raise it only while you have RAM headroom to spare.
* Keep `OCR_MAX_CONCURRENCY=1` unless you have measured otherwise. Parallel
  inferences inside one process contend for the same cores and usually make
  *both* requests slower while doubling peak memory.
* `run.sh` pins `OMP_NUM_THREADS=1`. Without it the math libraries spawn one
  thread per core inside every worker and the workers fight each other.
* Drop `OCR_LANGUAGES` to just the languages you need. Each one is a separate
  model load and a separate inference pass per request.
* `OCR_DET_LIMIT_SIDE_LEN` and `IMAGE_MAX_DIMENSION` trade accuracy for speed.
  Lowering them to `1280` roughly halves inference time on large scans.
* Use `REDIS_URL` once you run more than one worker, otherwise each worker
  enforces the rate limit independently.

Expect roughly 0.5-2 s per page per language on a modern CPU at default
settings. Set your client timeout accordingly — well above `OCR_TIMEOUT_SECONDS`.

---

---

## Production deployment (Ubuntu 22.04 / 24.04, no Docker)

> **Engine status.** Generic OCR infrastructure and parser are tested.
> PaddleOCR package initialization was verified, but inference is
> environment-dependent and must be validated on the target Linux server.
>
> Deploy with `OCR_PROVIDER=stub` first to prove the HTTP path, systemd unit,
> Nginx and TLS all work, then switch to `OCR_PROVIDER=paddle` and validate
> recognition on the server itself. The switch is one line in `.env` plus a
> restart.

Target: a Linux VPS running Ubuntu 22.04 or 24.04, Python 3.11+, a virtualenv,
Uvicorn behind Nginx, managed by systemd. No Docker anywhere.

```
Internet ──► Nginx (443, TLS)  ──►  Uvicorn 127.0.0.1:8000  ──►  FastAPI + PaddleOCR
             public entry point     loopback only, systemd
```

### 1. Server preparation

```bash
sudo apt update && sudo apt upgrade -y
sudo timedatectl set-timezone UTC
sudo hostnamectl set-hostname ocr-01
```

A 2 vCPU / 4 GB instance runs one worker comfortably. Give it at least 5 GB of
free disk: the virtualenv is roughly 2.5 GB (PaddlePaddle is large) and the
models add about 230 MB.

### 2. System packages

```bash
sudo apt install -y python3 python3-venv python3-dev                     build-essential git curl                     nginx ufw
```

Ubuntu 24.04 ships Python 3.12 and 22.04 ships 3.10. If you are on 22.04 and
want 3.11+:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

`opencv-python-headless` needs no X11 or GUI libraries, which is exactly why
the project uses it. Some minimal images still lack these:

```bash
sudo apt install -y libgl1 libglib2.0-0    # only if OpenCV fails to import
```

### 3. Application user and directory

The service must not run as root, and must not own its own source code — a
compromised process should not be able to rewrite the application.

```bash
sudo useradd --system --create-home --home-dir /opt/passport-ocr \
             --shell /usr/sbin/nologin ocr

sudo -u ocr git clone <your-repo-url> /opt/passport-ocr
cd /opt/passport-ocr

# Writable state: the model cache and (optionally) stored uploads.
sudo -u ocr mkdir -p /opt/passport-ocr/models /opt/passport-ocr/data
```

### 4. Virtualenv and requirements

```bash
sudo -u ocr python3.11 -m venv /opt/passport-ocr/.venv
sudo -u ocr /opt/passport-ocr/.venv/bin/pip install --upgrade pip setuptools wheel
sudo -u ocr /opt/passport-ocr/.venv/bin/pip install -r /opt/passport-ocr/requirements.txt
```

This pulls PaddlePaddle and takes a few minutes. Verify the engine imports
before going further — this is package initialization, not inference:

```bash
sudo -u ocr /opt/passport-ocr/.venv/bin/python -c \
  "import paddle, paddleocr; print(paddle.__version__, paddleocr.__version__)"
```

Then pre-download the models, so the first request does not pay for it and the
service can later run without outbound network access:

```bash
sudo -u ocr PADDLE_PDX_CACHE_HOME=/opt/passport-ocr/models \
  /opt/passport-ocr/.venv/bin/python -c \
  "import sys; sys.path.insert(0, '/opt/passport-ocr'); \
   from app.services.ocr.registry import create_provider; \
   create_provider('paddle').warmup(['en', 'arabic'])"
```

**Validate inference on this server now**, before wiring up systemd:

```bash
cd /opt/passport-ocr
sudo -u ocr .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from app.services.ocr.registry import create_provider
img = np.full((400, 1100, 3), 245, np.uint8)
cv2.putText(img, 'DEPLOYMENT CHECK 12345', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,20,20), 3)
p = create_provider('paddle'); p.warmup(['en'])
for b in p.recognize(img, 'en').blocks:
    print(round(b.confidence, 3), repr(b.text))
"
```

You should see the drawn text back with a high confidence. If this hangs or
crashes, the engine is not viable on this machine — keep `OCR_PROVIDER=stub`
and see [Troubleshooting](#troubleshooting) before continuing.

### 5. Environment configuration

```bash
sudo -u ocr cp /opt/passport-ocr/.env.production.example /opt/passport-ocr/.env
sudo -u ocr /opt/passport-ocr/.venv/bin/python -c \
  "import secrets; print(secrets.token_urlsafe(32))"
sudo -u ocr nano /opt/passport-ocr/.env
```

At minimum set `OCR_API_KEY` to the generated value and `ALLOWED_HOSTS` to your
domain. Start with `OCR_PROVIDER=stub` if you want to prove the plumbing first.

### 6. Permissions

```bash
# Source owned by root, readable by the service: it cannot rewrite itself.
sudo chown -R root:ocr /opt/passport-ocr
sudo chmod -R g=rX,o= /opt/passport-ocr

# The virtualenv and writable state belong to the service user.
sudo chown -R ocr:ocr /opt/passport-ocr/.venv \
                      /opt/passport-ocr/models \
                      /opt/passport-ocr/data

# The environment file holds a secret.
sudo chown ocr:ocr /opt/passport-ocr/.env
sudo chmod 600 /opt/passport-ocr/.env
```

### 7. systemd service

[`deploy/passport-ocr.service`](deploy/passport-ocr.service) binds Uvicorn to
`127.0.0.1:8000`, loads `.env`, pins the math libraries to one thread each, and
applies a hardening profile (`ProtectSystem=strict`, `NoNewPrivileges`, a
syscall filter and a 3 GB memory cap).

```bash
sudo cp /opt/passport-ocr/deploy/passport-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now passport-ocr
sudo systemctl status passport-ocr --no-pager
```

Startup allows 300 s, because loading models on a cold cache is slow. Confirm
the service is up and listening only on loopback:

```bash
curl -s http://127.0.0.1:8000/health     # {"status":"ok"}
curl -s http://127.0.0.1:8000/ready      # ocr_ready: true once models load
sudo ss -lntp | grep 8000                # must show 127.0.0.1:8000, not 0.0.0.0
```

### 8. Nginx reverse proxy

[`deploy/nginx.conf`](deploy/nginx.conf) terminates TLS, rate limits at the
edge, caps uploads at 12 MB, sets the proxy timeouts above
`OCR_TIMEOUT_SECONDS`, and streams uploads straight through rather than
spooling them to disk.

```bash
sudo cp /opt/passport-ocr/deploy/nginx.conf /etc/nginx/sites-available/passport-ocr
sudo sed -i 's/ocr.example.com/YOUR_DOMAIN/g' /etc/nginx/sites-available/passport-ocr
sudo ln -sf /etc/nginx/sites-available/passport-ocr /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

The shipped file already contains the `443` block. Before you have a
certificate, comment it out (or delete it) so `nginx -t` passes, run certbot,
and it will write the TLS block itself.

### 9. HTTPS with Let's Encrypt

Point an `A` record for your domain at the server, then:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d ocr.example.com --agree-tos -m you@example.com --redirect
```

Certbot installs a systemd timer that renews automatically. Verify it:

```bash
sudo certbot renew --dry-run
systemctl list-timers | grep certbot
```

### 10. Firewall

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status verbose
```

Port 8000 is deliberately **not** opened: the application listens on loopback
only and is reachable exclusively through Nginx.

### 11. Health checks

| Check | Command | Expected |
|---|---|---|
| Process | `systemctl is-active passport-ocr` | `active` |
| Liveness | `curl -s https://ocr.example.com/health` | `{"status":"ok"}` |
| Readiness | `curl -s https://ocr.example.com/ready` | `"ocr_ready": true` |
| Engine | `curl -s https://ocr.example.com/api/v1/version` | reports the loaded provider |
| Recognition | `POST /api/v1/ocr` with a test image | 200 with recognised text |

`/health` never touches the engine, so a slow model load cannot make a healthy
process look dead. `/ready` returns 503 until the models are resident — use
`/health` for restart policies and `/ready` for load-balancer membership.

An end-to-end check from the server:

```bash
cd /opt/passport-ocr
sudo -u ocr .venv/bin/python -c "
import cv2, numpy as np
img = np.full((400, 1100, 3), 245, np.uint8)
cv2.putText(img, 'HEALTH CHECK 2026', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,20,20), 3)
cv2.imwrite('/tmp/check.png', img)"
curl -s -X POST https://ocr.example.com/api/v1/ocr \
     -H "X-API-Key: $OCR_API_KEY" -F "image=@/tmp/check.png" | head -c 300
rm -f /tmp/check.png
```

### 12. Logs

The application writes structured JSON to stdout; journald collects it. No
document image, recognised text, client filename or API key ever reaches the
log stream.

```bash
sudo journalctl -u passport-ocr -f                  # follow
sudo journalctl -u passport-ocr -n 200 --no-pager   # recent
sudo journalctl -u passport-ocr -p err --since today
sudo journalctl -u passport-ocr --since "1 hour ago" -o cat | jq 'select(.level=="ERROR")'

sudo tail -f /var/log/nginx/passport-ocr.access.log
sudo tail -f /var/log/nginx/passport-ocr.error.log
```

Every response carries `X-Request-ID`, and that id appears in every log line
for the request — quote it when investigating a specific call.

Cap journald so logs cannot fill the disk:

```bash
sudo sed -i 's/^#SystemMaxUse=.*/SystemMaxUse=500M/' /etc/systemd/journald.conf
sudo systemctl restart systemd-journald
```

### 13. Restart procedure

```bash
sudo systemctl restart passport-ocr     # full restart, reloads models (~30-60s)
sudo systemctl reload nginx             # config change only, no dropped connections
sudo systemctl status passport-ocr --no-pager
```

A restart drops in-flight requests. `KillSignal=SIGINT` gives Uvicorn a chance
to finish them within `TimeoutStopSec=30`.

### 14. Update procedure

```bash
cd /opt/passport-ocr
sudo -u ocr git pull

# Only when requirements.txt changed:
sudo -u ocr .venv/bin/pip install -r requirements.txt

# Optional but recommended: run the suite against the stub before restarting.
# pytest lives in requirements-dev.txt, so install that once if you want this
# check available on the server.
#   sudo -u ocr .venv/bin/pip install -r requirements-dev.txt
sudo -u ocr env OCR_PROVIDER=stub .venv/bin/pytest -m "not integration"

sudo systemctl restart passport-ocr
sleep 15
curl -fsS http://127.0.0.1:8000/health && echo " OK"
curl -s http://127.0.0.1:8000/ready
```

If the health check fails, roll back and restart:

```bash
sudo -u ocr git reset --hard HEAD@{1}
sudo systemctl restart passport-ocr
```

### Switching the engine

The engine is one line in `/opt/passport-ocr/.env`:

```bash
OCR_PROVIDER=stub      # no models loaded, returns only scripted text
OCR_PROVIDER=paddle    # real PaddleOCR
```

```bash
sudo -u ocr sed -i 's/^OCR_PROVIDER=.*/OCR_PROVIDER=paddle/' /opt/passport-ocr/.env
sudo systemctl restart passport-ocr
curl -s http://127.0.0.1:8000/api/v1/version | grep -o '"provider":"[^"]*"'
```

Nothing else changes: same code, same unit file, same Nginx configuration. With
`stub` the PaddleOCR packages are never imported at all, which the test-suite
enforces in `tests/test_provider_isolation.py`.

### Deployment checklist

**Server**
- [ ] Ubuntu 22.04 or 24.04, packages updated, timezone set
- [ ] Python 3.11+ available (`python3.11 --version`)
- [ ] `nginx`, `ufw`, `build-essential`, `python3-venv` installed
- [ ] At least 4 GB RAM and 5 GB free disk

**Application**
- [ ] `ocr` system user created, `nologin` shell
- [ ] Repository at `/opt/passport-ocr`, source owned by `root:ocr`
- [ ] `.venv` created and requirements installed
- [ ] `paddle` and `paddleocr` import successfully
- [ ] Models pre-downloaded into `/opt/passport-ocr/models`
- [ ] **Inference validated on this server** (step 4)

**Configuration**
- [ ] `.env` created from `.env.production.example`
- [ ] `OCR_API_KEY` set to a generated secret, not `CHANGE_ME`
- [ ] `.env` is `chmod 600`, owned by `ocr`
- [ ] `ALLOWED_HOSTS` set to the real domain
- [ ] `TRUST_PROXY_HEADERS=true`, `DOCS_ENABLED=false`, `DEBUG=false`
- [ ] `LOG_SENSITIVE_DATA=false`
- [ ] `OCR_PROVIDER` set deliberately (`stub` to prove plumbing, then `paddle`)

**Service**
- [ ] Unit installed, `daemon-reload` run, service enabled and active
- [ ] `ss -lntp` shows `127.0.0.1:8000`, never `0.0.0.0:8000`
- [ ] `/health` returns `ok`; `/ready` reports `ocr_ready: true`

**Edge**
- [ ] Nginx config installed with the real domain substituted, `nginx -t` passes
- [ ] Default site removed
- [ ] TLS certificate issued; `certbot renew --dry-run` passes
- [ ] HTTP redirects to HTTPS
- [ ] `client_max_body_size` exceeds `MAX_UPLOAD_SIZE`
- [ ] `proxy_read_timeout` exceeds `OCR_TIMEOUT_SECONDS`

**Security**
- [ ] `ufw` active; only SSH and Nginx allowed; 8000 not exposed
- [ ] A request without an API key returns 401
- [ ] Logs contain no document text, filenames or keys
- [ ] `STORE_UPLOADS=false`, or a retention policy exists

**Operations**
- [ ] `journalctl -u passport-ocr` shows structured JSON
- [ ] journald size capped
- [ ] Restart and rollback procedures rehearsed once
- [ ] An end-to-end `POST /api/v1/ocr` returns correct text for a test image

## Troubleshooting

**`OCR_FAILED: PaddleOCR is not installed`**
The models are not in the virtualenv. `pip install -r requirements.txt`. To work
on the API without them, set `OCR_PROVIDER=stub`.

**Startup hangs on the first run**
PaddleOCR is downloading models. It needs network access once; afterwards it is
offline. Watch with `LOG_LEVEL=DEBUG`, or pre-download (see
[First run](#first-run-and-model-download)).

**`paddlepaddle` will not install**
Wheel availability varies by platform and Python version — Apple Silicon in
particular is patchy. Check <https://www.paddlepaddle.org.cn/en> for the wheel
matching your platform. `OCR_PROVIDER=stub` keeps the rest of the service usable
meanwhile.

**`/ready` returns 503**
Models are still loading, or a load failed. Check the logs for
`ocr_warmup_failed`. `/health` stays `200` throughout by design.

**Everything comes back empty**
Check `confidence` and `warnings` in the response. Usually resolution: aim for
at least 1000 px on the long side and 25+ px of text height. Try
`preprocess=false` to rule out the enhancement chain, and confirm the language
is right — an Arabic page will not read with `languages=en`.

**Text is there but garbled**
Try `detect_orientation=true` for a rotated page, or `preprocess=false` if the
page is already clean — aggressive enhancement can hurt a high-quality scan.

**429 responses under normal load**
Raise `RATE_LIMIT_REQUESTS`, or set `REDIS_URL` if the limit is being applied
per worker rather than per service.

**Client times out**
Your client's timeout must exceed `OCR_TIMEOUT_SECONDS` (default 45).

---

## Project layout

```
app/
  main.py                        application factory, lifespan, middleware wiring
  api/
    deps.py                      API key auth, rate limit dependency
    errors.py                    exception handlers, error envelope
    middleware.py                request id, body size guard, security headers
    v1/
      ocr.py                     POST /api/v1/ocr
      system.py                  GET /health, GET /ready
      version.py                 GET /api/v1/version
      router.py
  core/
    config.py                    environment-driven settings
    exceptions.py                error codes and the exception hierarchy
    logging.py                   structured JSON logging with PII redaction
    ratelimit.py                 in-memory and Redis limiters
    security.py                  constant-time key comparison
  schemas/                       Pydantic request/response models
  services/
    pipeline.py                  the orchestration described above
    layout.py                    lines, reading order, regions, deduplication
    image_processing/
      loader.py                  validation and decoding
      preprocess.py              perspective, downscale, deskew, enhance
    ocr/
      base.py                    OCRProvider, TextBlock, OCRResult
      paddle.py                  PaddleOCRProvider
      stub.py                    scripted provider for tests
      registry.py                name -> provider
      engine.py                  load-once engine, thread pool, timeouts
  utils/                         text, date and secure file helpers
tests/                           pytest suite, synthetic fixtures only
scripts/                         setup, run, test, smoke
```

### Standalone library: machine-readable text parser

`app/services/mrz/` is an independent, standards-based parser for ICAO Doc 9303
machine-readable zones (TD1, TD2, TD3 and MRV). It takes **already-recognised
text lines** and returns a structured, per-field result:

```python
from app.services.mrz import create_parser

parser = create_parser("icao9303")
document = parser.parse_lines([line_one, line_two], ocr_confidence=0.94)

if document is not None:
    field = document.get("document_number")
    field.value        # the value read, or None if it could not be read
    field.valid        # True/False from the check digit, None if unprotected
    field.confidence   # how much the standard's redundancy supports it
    field.errors       # what went wrong, if anything
```

It is **not wired into the API**. It imports nothing from FastAPI, the OCR
engines, the image pipeline, or even numpy/OpenCV - a property the test-suite
enforces in a subprocess. `app/services/mrz/detector.py` is the optional bridge
that finds a zone in positioned OCR boxes; it is the only module there that
touches the OCR layer, and nothing else depends on it.

See [`app/services/mrz/__init__.py`](app/services/mrz/__init__.py) for the layer
map, and `tests/mrz/` for 259 tests built entirely on synthetic fixtures.

### Not wired into the API

These modules are used only by the parser library above, not by any endpoint:

* `app/utils/text.py` - Arabic/Latin text normalisation and string similarity.
* `app/utils/dates.py` - date parsing and plausibility checks (the parser uses
  it for century resolution).

`app/utils/files.py` **is** used by the API (secure temp-file handling).
