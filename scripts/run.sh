#!/usr/bin/env bash
# Run the OCR service from the local virtualenv.
#
#   ./scripts/run.sh              # production-style, reads .env
#   ./scripts/run.sh --reload     # development, auto-reload on file changes
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "No virtualenv found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

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

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Each OCR worker is already multi-threaded internally. Letting the math
# libraries spawn one thread per core on top of that oversubscribes the CPU and
# makes every request slower.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# No --log-config: the app installs its own JSON logging (and takes over
# uvicorn's loggers) during startup.
ARGS=(app.main:app --host "$HOST" --port "$PORT")

if [ "${1:-}" = "--reload" ]; then
    echo "==> Starting in reload mode on http://${HOST}:${PORT}"
    ARGS+=(--reload)
elif [ "$WORKERS" -gt 1 ]; then
    echo "==> Starting ${WORKERS} workers on http://${HOST}:${PORT}"
    echo "    Note: each worker loads its own copy of the OCR models."
    ARGS+=(--workers "$WORKERS")
else
    echo "==> Starting on http://${HOST}:${PORT}"
fi

exec "$VENV/bin/uvicorn" "${ARGS[@]}"
