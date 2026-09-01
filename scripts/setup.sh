#!/usr/bin/env bash
# Create the virtualenv and install dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Python 3.9+ is required (3.11+ recommended). Found: $("$PYTHON" --version)" >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "==> Creating virtualenv in $VENV"
    "$PYTHON" -m venv "$VENV"
fi

echo "==> Installing dependencies"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -r requirements.txt

if [ "${DEV:-0}" = "1" ]; then
    echo "==> Installing development dependencies"
    "$VENV/bin/python" -m pip install -r requirements-dev.txt
fi

if [ ! -f .env ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
    KEY="$("$VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    # Seed a strong API key so the service is never accidentally left open.
    if sed --version >/dev/null 2>&1; then
        sed -i "s|^OCR_API_KEY=.*|OCR_API_KEY=${KEY}|" .env
    else
        sed -i '' "s|^OCR_API_KEY=.*|OCR_API_KEY=${KEY}|" .env
    fi
    echo "==> Generated API key: ${KEY}"
fi

echo
echo "Setup complete. Start the service with: ./scripts/run.sh"
