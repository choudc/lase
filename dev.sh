#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/lase-backend"
FRONTEND_DIR="$ROOT_DIR/lase-frontend"

if [ ! -d "$BACKEND_DIR/venv" ]; then
  cd "$BACKEND_DIR"
  python3 -m venv venv
  source venv/bin/activate
  python -m pip install -U pip
  python -m pip install -r requirements.txt
fi

cd "$BACKEND_DIR"
source venv/bin/activate
export LASE_DEBUG=1

python -m src.main &
BACK_PID=$!

cleanup() {
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$FRONTEND_DIR"
if command -v pnpm >/dev/null 2>&1; then
  pnpm install
  pnpm run dev
else
  npm install
  npm run dev
fi

