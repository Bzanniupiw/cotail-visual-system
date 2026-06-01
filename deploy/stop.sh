#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "${SCRIPT_DIR}")" = "deploy" ]; then
  APP_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  APP_DEFAULT="${SCRIPT_DIR}"
fi

APP="${COTAIL_APP_DIR:-${APP_DEFAULT}}"

if [ ! -f "${APP}/server.pid" ]; then
  echo "No server.pid found."
  exit 0
fi

pid="$(cat "${APP}/server.pid" || true)"
if [ -n "${pid:-}" ] && kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}" || true
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" || true
  fi
  echo "Stopped CoTail: pid=${pid}"
else
  echo "CoTail process is not running."
fi

rm -f "${APP}/server.pid"
