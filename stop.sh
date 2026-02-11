#!/bin/bash

# Directory setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/lase.pids"

echo "🛑 Stopping LASE..."

# 1. Kill from PID file if exists
if [ -f "$PID_FILE" ]; then
    while read -r PID; do
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "   Killing PID $PID..."
            kill "$PID" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm "$PID_FILE"
fi

# 2. Fallback: Kill by pattern (cleanup)
# Backend
pkill -f "python -m src.main" && echo "   Cleaned up lingering backend processes." || true
# Frontend (vite)
pkill -f "vite" && echo "   Cleaned up lingering frontend processes." || true

# 3. Aggressive Cleanup: Kill anything on ports 5000 (Backend) and 5173 (Frontend)
if command -v lsof >/dev/null; then
    # Port 5000
    PIDS_5000=$(lsof -t -i:5000)
    if [ -n "$PIDS_5000" ]; then
        echo "   Killing processes on port 5000: $PIDS_5000"
        kill -9 $PIDS_5000 2>/dev/null || true
    fi

    # Port 5173
    PIDS_5173=$(lsof -t -i:5173)
    if [ -n "$PIDS_5173" ]; then
        echo "   Killing processes on port 5173: $PIDS_5173"
        kill -9 $PIDS_5173 2>/dev/null || true
    fi
fi

echo "✅ All services stopped."
