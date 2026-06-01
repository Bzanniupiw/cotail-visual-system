#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "${SCRIPT_DIR}")" = "deploy" ]; then
  APP_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  APP_DEFAULT="${SCRIPT_DIR}"
fi

APP="${COTAIL_APP_DIR:-${APP_DEFAULT}}"
PORT="${COTAIL_PORT:-8899}"

if [ -f "${APP}/server.pid" ]; then
  pid="$(cat "${APP}/server.pid" || true)"
  ps -p "${pid}" -o pid,etime,cmd || true
else
  echo "No server.pid found."
fi

ss -ltnp 2>/dev/null | grep ":${PORT}" || true
curl -fsS "http://127.0.0.1:${PORT}/api/health" || true
echo
