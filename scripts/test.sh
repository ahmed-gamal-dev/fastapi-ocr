#!/usr/bin/env bash
# Run the test suite. Passes any extra arguments through to pytest.
#
#   ./scripts/test.sh
#   ./scripts/test.sh --cov=app
#   ./scripts/test.sh tests/test_layout.py -v
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"

if [ ! -x "$VENV/bin/pytest" ]; then
    echo "pytest not installed. Run: DEV=1 ./scripts/setup.sh" >&2
    exit 1
fi

# The suite uses the scripted stub provider, so it never needs the OCR models.
export OCR_PROVIDER=stub

exec "$VENV/bin/pytest" "$@"
