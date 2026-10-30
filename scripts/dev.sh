#!/usr/bin/env bash
#
# Start the API and the Streamlit UI with one shared API key.
#
# Reads API_KEY from the repo-root .env, exports it as both API_KEY (backend)
# and BACKEND_API_KEY (frontend), starts uvicorn with exactly one worker, waits
# for /health, then runs Streamlit in the foreground. Stopping Streamlit stops
# the API too.
#
# Usage:
#   ./scripts/dev.sh
#   PYTHON=/path/to/python ./scripts/dev.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO/.env"
PYTHON="${PYTHON:-python}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "No .env at $ENV_FILE. Copy .env.example to .env and add the three keys." >&2
    exit 1
fi

# Read only API_KEY. Never echo the value.
API_KEY_VALUE="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?API_KEY[[:space:]]*=[[:space:]]*//p' "$ENV_FILE" | tail -n 1 | sed -E 's/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/' | tr -d '\r')"

if [[ -z "$API_KEY_VALUE" ]]; then
    echo "API_KEY is missing from $ENV_FILE. Generate one with:" >&2
    echo '  python -c "import secrets; print(secrets.token_urlsafe(32))"' >&2
    exit 1
fi

export API_KEY="$API_KEY_VALUE"
export BACKEND_API_KEY="$API_KEY_VALUE"
export PYTHONUNBUFFERED=1

echo "Starting API on http://127.0.0.1:$API_PORT (workers=1) ..."

(cd "$REPO/backend" && "$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --workers 1) &
API_PID=$!

cleanup() {
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "The API exited during startup. Read the output above." >&2
        exit 1
    fi

    if "$PYTHON" - "$API_PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2)
PY
    then
        echo "API ready. Starting UI on http://127.0.0.1:$UI_PORT ..."
        # No exec: keeping this shell lets the EXIT trap stop the API when
        # Streamlit exits.
        "$PYTHON" -m streamlit run "$REPO/frontend/app.py" \
            --server.address 127.0.0.1 \
            --server.port "$UI_PORT"
        exit $?
    fi

    sleep 0.5
done

echo "The API did not become healthy on port $API_PORT within 45 seconds." >&2
exit 1
