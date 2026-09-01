#!/usr/bin/env bash
# Update a running deployment, in place, safely.
#
# Runs ON THE SERVER, as the user that owns the checkout:
#
#   cd /opt/passport-ocr && sudo -u ocr ./scripts/deploy.sh
#
# It pulls, installs only when dependencies changed, runs the stub-provider
# test suite, restarts the service and health-checks it. If the health check
# fails it rolls back to the previous commit and restarts again, so a bad
# deploy does not leave the service down.
set -euo pipefail

cd "$(dirname "$0")/.."

APP_DIR="$(pwd)"
VENV="${VENV:-$APP_DIR/.venv}"
SERVICE="${SERVICE:-passport-ocr}"
BRANCH="${BRANCH:-master}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-30}"
RUN_TESTS="${RUN_TESTS:-1}"

log() { printf '==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$VENV/bin/python" ] || fail "no virtualenv at $VENV - run scripts/setup.sh first"

PREVIOUS="$(git rev-parse HEAD)"
log "current commit: $PREVIOUS"

# ---------------------------------------------------------------- fetch
log "fetching origin/$BRANCH"
git fetch origin "$BRANCH" --quiet
TARGET="$(git rev-parse "origin/$BRANCH")"

if [ "$PREVIOUS" = "$TARGET" ]; then
    log "already up to date; nothing to deploy"
    exit 0
fi

# Refuse to clobber uncommitted work on the server.
if ! git diff --quiet || ! git diff --cached --quiet; then
    fail "the working tree has uncommitted changes - resolve them first"
fi

REQ_BEFORE="$(git rev-parse HEAD:requirements.txt 2>/dev/null || echo none)"

log "updating to $TARGET"
git merge --ff-only "origin/$BRANCH"

# ------------------------------------------------------- dependencies
REQ_AFTER="$(git rev-parse HEAD:requirements.txt 2>/dev/null || echo none)"
if [ "$REQ_BEFORE" != "$REQ_AFTER" ]; then
    log "requirements.txt changed - installing"
    "$VENV/bin/pip" install --upgrade pip --quiet
    "$VENV/bin/pip" install -r requirements.txt
else
    log "requirements unchanged - skipping install"
fi

# ------------------------------------------------------------- tests
if [ "$RUN_TESTS" = "1" ] && [ -x "$VENV/bin/pytest" ]; then
    log "running the test suite (stub provider)"
    if ! OCR_PROVIDER=stub "$VENV/bin/pytest" -m "not integration" -q; then
        log "tests failed - rolling back to $PREVIOUS"
        git reset --hard "$PREVIOUS"
        fail "deployment aborted; nothing was restarted"
    fi
else
    log "skipping tests (pytest not installed, or RUN_TESTS=0)"
fi

# ----------------------------------------------------------- restart
log "restarting $SERVICE"
if ! sudo -n systemctl restart "$SERVICE" 2>/dev/null; then
    systemctl restart "$SERVICE" \
        || fail "could not restart $SERVICE - check sudo permissions"
fi

# ------------------------------------------------------ health check
log "waiting for $HEALTH_URL"
healthy=0
for _ in $(seq 1 "$HEALTH_RETRIES"); do
    if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 2
done

if [ "$healthy" != "1" ]; then
    log "health check FAILED - rolling back to $PREVIOUS"
    git reset --hard "$PREVIOUS"
    sudo -n systemctl restart "$SERVICE" 2>/dev/null || systemctl restart "$SERVICE" || true
    fail "rolled back; inspect: journalctl -u $SERVICE -n 100 --no-pager"
fi

log "health check passed"
curl -fsS --max-time 5 "${HEALTH_URL%/health}/ready" 2>/dev/null || true
echo
log "deployed $TARGET"
