#!/usr/bin/env python
"""Time the recognition engine under different settings, on this machine.

Run it on the host that serves traffic - an ARM laptop says nothing about an
x86 server, and oneDNN in particular exists only on x86. It never touches the
running service: it loads its own engine, in its own process, per setting.

Each setting runs in a subprocess so that a crash - oneDNN has crashed this
engine before, in ConvertPirAttribute2RuntimeAttribute - is reported as a
result instead of ending the run.

The page is synthetic and drawn with OpenCV, so there is nothing to install
and no document involved:

    .venv/bin/python scripts/bench_engine.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

SETTINGS = {
    "current (oneDNN off)": {"PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "False"},
    "oneDNN on": {"PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "True"},
    "oneDNN on, legacy IR": {
        "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "True",
        "FLAGS_enable_pir_api": "0",
    },
}
RUNS = 3


def draw_page(path: str) -> None:
    import cv2
    import numpy as np

    page = np.full((872, 1280, 3), 244, np.uint8)
    lines = [
        "PASSPORT   UTO   L898902C3",
        "SURNAME  ERIKSSON",
        "GIVEN NAMES  ANNA MARIA",
        "DATE OF BIRTH 12 AUG 1974    SEX F",
        "DATE OF ISSUE 16 APR 2012",
        "DATE OF EXPIRY 15 APR 2022",
        "AUTHORITY  PASSPORT OFFICE",
    ]
    for i, text in enumerate(lines):
        cv2.putText(page, text, (440, 120 + i * 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (30, 30, 30), 2, cv2.LINE_AA)
    for i, text in enumerate([
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]):
        cv2.putText(page, text, (60, 740 + i * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.imwrite(path, page)


def child(image: str) -> None:
    """One setting, one process. Prints a single JSON line."""
    import warnings

    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCR

    lang = os.environ.get("BENCH_LANG", "ar")
    engine = PaddleOCR(
        lang=lang,
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
    )
    engine.predict(image)  # the first call builds kernels; do not time it
    times = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = engine.predict(image)
        times.append((time.perf_counter() - started) * 1000)
    boxes = len(result[0]["rec_texts"]) if result else 0
    print(json.dumps({"ms": statistics.median(times), "boxes": boxes}))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return

    image = os.path.join(tempfile.mkdtemp(), "bench.png")
    draw_page(image)
    print(f"cpus={os.cpu_count()}  runs={RUNS}  language={os.environ.get('BENCH_LANG', 'ar')}")

    for name, overrides in SETTINGS.items():
        env = {**os.environ, **overrides}
        try:
            done = subprocess.run(
                [sys.executable, __file__, "--child", image],
                env=env, capture_output=True, text=True, timeout=900,
            )
        except subprocess.TimeoutExpired:
            print(f"  {name:<24} timed out")
            continue
        line = next(
            (row for row in reversed(done.stdout.splitlines()) if row.startswith("{")), None
        )
        if done.returncode != 0 or line is None:
            reason = (done.stderr.strip().splitlines() or ["no output"])[-1][:90]
            print(f"  {name:<24} FAILED  {reason}")
            continue
        result = json.loads(line)
        print(f"  {name:<24} {result['ms']:8.0f} ms   boxes={result['boxes']}")


if __name__ == "__main__":
    main()
