#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "${SCRIPT_DIR}")" = "deploy" ]; then
  APP_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  APP_DEFAULT="${SCRIPT_DIR}"
fi

APP="${COTAIL_APP_DIR:-${APP_DEFAULT}}"
PY="${COTAIL_PYTHON:-python3}"
HOST="${COTAIL_HOST:-0.0.0.0}"
PORT="${COTAIL_PORT:-8899}"

cd "${APP}"
mkdir -p storage/logs

if [ -f server.pid ]; then
  old="$(cat server.pid || true)"
  if [ -n "${old:-}" ] && kill -0 "${old}" 2>/dev/null; then
    echo "CoTail is already running: pid=${old}"
    exit 0
  fi
fi

export PATH="${COTAIL_EXTRA_PATH:-}:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
nohup "${PY}" -m uvicorn backend.app.main:app --host "${HOST}" --port "${PORT}" > server.log 2> server.err.log &
echo "$!" > server.pid
echo "Started CoTail: pid=$(cat server.pid), url=http://$(hostname -I | awk '{print $1}'):${PORT}"
