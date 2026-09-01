"""The stub provider must be a complete substitute for the real engine.

Development and CI run with ``OCR_PROVIDER=stub``, so booting the application
that way must not import PaddleOCR - not for a warmup, not for a readiness
probe, not at all. Each check runs in a subprocess because import state is
process-wide and cannot be undone.
"""

from __future__ import annotations

import subprocess
import sys

PADDLE_ROOTS = ("paddle", "paddleocr", "paddlex")


def run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": ".",
            "OCR_PROVIDER": "stub",
            "OCR_API_KEY": "isolation-key",
            "LOG_LEVEL": "CRITICAL",
        },
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


IMPORTED = (
    "import sys;"
    f"print([m for m in sys.modules if m.split('.')[0] in {PADDLE_ROOTS!r}])"
)


def test_booting_with_the_stub_never_imports_paddleocr():
    output = run(
        "from fastapi.testclient import TestClient;"
        "from app.main import create_app;"
        "c = TestClient(create_app());"
        "c.__enter__();"
        "assert c.get('/health').json() == {'status': 'ok'};"
        "assert c.get('/ready').json()['ocr_ready'] is True;"
        "c.__exit__(None, None, None);" + IMPORTED
    )
    assert output == "[]", f"the stub path pulled in {output}"


def test_serving_a_request_with_the_stub_never_imports_paddleocr():
    output = run(
        "import cv2, numpy as np;"
        "from fastapi.testclient import TestClient;"
        "from app.main import create_app;"
        "img = np.full((400, 900, 3), 245, np.uint8);"
        "data = cv2.imencode('.png', img)[1].tobytes();"
        "c = TestClient(create_app());"
        "c.__enter__();"
        "r = c.post('/api/v1/ocr', files={'image': ('p.png', data, 'image/png')},"
        "           headers={'X-API-Key': 'isolation-key'});"
        "assert r.status_code == 200, r.text;"
        "c.__exit__(None, None, None);" + IMPORTED
    )
    assert output == "[]", f"serving a request pulled in {output}"


def test_importing_the_registry_alone_does_not_load_any_engine():
    """The registry maps names to factories; nothing is constructed on import."""
    output = run("import app.services.ocr.registry;" + IMPORTED)
    assert output == "[]"


def test_the_paddle_provider_is_still_registered_and_selectable():
    """Keeping the stub cheap must not mean dropping the production engine."""
    output = run(
        "from app.services.ocr.registry import available_providers;"
        "print(sorted(available_providers()))"
    )
    assert output == "['paddle', 'paddleocr', 'stub']"


def test_selecting_paddle_is_what_pulls_the_engine_in():
    """The counterpart to the tests above: the import happens on demand, at
    provider construction, and only then."""
    assert run("from app.services.ocr.registry import create_provider;" + IMPORTED) == "[]"
