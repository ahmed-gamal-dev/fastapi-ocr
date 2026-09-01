#!/usr/bin/env bash
# End-to-end check against a running instance.
#
#   ./scripts/smoke.sh [BASE_URL] [IMAGE_PATH]
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${1:-http://127.0.0.1:8000}"
IMAGE="${2:-}"

# Load .env, but let variables already set in the environment win: running
# `OCR_API_KEY=... ./scripts/x.sh` must not be silently overridden by the file.
load_dotenv() {
    [ -f .env ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key=${line%%=*}
        value=${line#*=}
        # Only well-formed KEY names.
        case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
        # Strip an unquoted trailing inline comment ("VALUE  # note"), then any
        # trailing whitespace. Without this a commented value is exported with
        # the comment attached and fails validation at startup.
        case "$value" in
            \"*\"|\'*\') ;;
            *) value=$(printf '%s' "$value" | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/[[:space:]]*$//') ;;
        esac
        # Drop surrounding quotes if present.
        case "$value" in
            \"*\") value=${value#\"}; value=${value%\"} ;;
            \'*\') value=${value#\'}; value=${value%\'} ;;
        esac
        if [ -z "${!key+set}" ]; then
            export "$key=$value"
        fi
    done < .env
}

load_dotenv

echo "==> GET ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health"; echo

echo "==> GET ${BASE_URL}/ready"
curl -sS "${BASE_URL}/ready"; echo

echo "==> GET ${BASE_URL}/api/v1/version"
curl -fsS "${BASE_URL}/api/v1/version"; echo

if [ -z "$IMAGE" ]; then
    echo "==> Generating a synthetic test image"
    IMAGE="$(mktemp -t ocr-smoke).png"
    "${VENV:-.venv}/bin/python" - "$IMAGE" <<'PY'
import sys
import cv2
import numpy as np
image = np.full((640, 1000, 3), 245, np.uint8)
for i, text in enumerate(["SMOKE TEST PAGE", "SECOND LINE 12345"]):
    cv2.putText(image, text, (40, 100 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
cv2.imwrite(sys.argv[1], image)
PY
fi

echo "==> POST ${BASE_URL}/api/v1/ocr"
curl -fsS -X POST "${BASE_URL}/api/v1/ocr" \
    -H "X-API-Key: ${OCR_API_KEY:-}" \
    -F "image=@${IMAGE}"
echo
echo "==> Smoke test finished"
