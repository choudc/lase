#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/lase.pids"

backend_ok=0
frontend_ok=0

if [ -f "$PID_FILE" ]; then
  mapfile -t PIDS < "$PID_FILE"
  BACKEND_PID="${PIDS[0]:-}"
  FRONTEND_PID="${PIDS[1]:-}"

  if [ -n "${BACKEND_PID}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    backend_ok=1
  fi
  if [ -n "${FRONTEND_PID}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    frontend_ok=1
  fi
fi

if [ "$backend_ok" -eq 1 ] || [ "$frontend_ok" -eq 1 ]; then
  echo "Status: running"
  echo "Backend PID: ${BACKEND_PID:-n/a} ($( [ "$backend_ok" -eq 1 ] && echo up || echo down ))"
  echo "Frontend PID: ${FRONTEND_PID:-n/a} ($( [ "$frontend_ok" -eq 1 ] && echo up || echo down ))"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "http://localhost:5000/api/health" >/dev/null 2>&1; then
      echo "API: ok"
    else
      echo "API: not responding"
    fi
    if curl -fsS "http://localhost:5173" >/dev/null 2>&1; then
      echo "UI: ok"
    else
      echo "UI: not responding"
    fi
  fi
  exit 0
fi

echo "Status: not running"
exit 1
