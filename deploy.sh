#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/lase-backend"
FRONTEND_DIR="$ROOT_DIR/lase-frontend"

log() { echo "[deploy] $*"; }

MODE="all"
NO_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --backend-only) MODE="backend" ;;
    --frontend-only) MODE="frontend" ;;
    --no-install) NO_INSTALL=1 ;;
    *)
      echo "Unknown arg: $arg" >&2
      echo "Usage: ./deploy.sh [--backend-only|--frontend-only] [--no-install]" >&2
      exit 2
      ;;
  esac
done

if [ ! -d "$BACKEND_DIR" ]; then
  echo "Missing $BACKEND_DIR" >&2
  exit 1
fi

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "Missing $FRONTEND_DIR" >&2
  exit 1
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "backend" ]; then
  log "Setting up backend venv + deps..."
  cd "$BACKEND_DIR"
  if [ ! -d venv ]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  python -m pip install -U pip
  python -m pip install -r requirements.txt
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "frontend" ]; then
  log "Building frontend..."
  cd "$FRONTEND_DIR"
  if command -v pnpm >/dev/null 2>&1; then
    if [ "$NO_INSTALL" -eq 0 ]; then
      pnpm install
    fi
    pnpm run build
  else
    if [ "$NO_INSTALL" -eq 0 ]; then
      npm install
    fi
    npm run build
  fi
fi

if [ "$MODE" = "all" ]; then
  log "Copying frontend build into backend static dir..."
  rm -rf "$BACKEND_DIR/src/static"
  mkdir -p "$BACKEND_DIR/src/static"
  cp -r "$FRONTEND_DIR/dist/"* "$BACKEND_DIR/src/static/"
fi

log "Done."
