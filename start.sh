#!/bin/bash
set -e

# Directory setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/lase-backend"
FRONTEND_DIR="$ROOT_DIR/lase-frontend"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/lase.pids"

mkdir -p "$LOG_DIR"

# Stop existing processes first
if [ -f "$ROOT_DIR/stop.sh" ]; then
    "$ROOT_DIR/stop.sh"
    echo "Wait for ports to clear..."
    sleep 2
fi

echo "🚀 Starting LASE..."

# 1. Start Backend
echo "   --> Starting Backend..."
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
    echo "       Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt # Ensure dependencies are always installed
    playwright install chromium
else
    source venv/bin/activate
    pip install -r requirements.txt # Ensure dependencies are always installed
fi

# Run backend in background
nohup python -m src.main > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "       Backend PID: $BACKEND_PID"

# 2. Start Frontend
echo "   --> Starting Frontend..."
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "       Installing frontend dependencies..."
    pnpm install
fi

# Run frontend in background
nohup pnpm dev --port 5173 --host > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "       Frontend PID: $FRONTEND_PID"

# Save PIDs
echo "$BACKEND_PID" > "$PID_FILE"
echo "$FRONTEND_PID" >> "$PID_FILE"

# Wait for startup
echo "⏳ Waiting for services to come online..."
sleep 5

# Check if processes are still running
if kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; then
    echo ""
    echo "✅ LASE is running!"
    echo "   🏠 Mission Control: http://localhost:5173"
    echo "   ⚙️  Backend API:    http://localhost:5000"
    echo ""
    echo "Logs are being written to $LOG_DIR"
    echo "Run ./stop.sh to stop all services."
else
    echo "❌ Failed to start one or more services."
    echo "Check logs in $LOG_DIR"
    cat "$LOG_DIR/backend.log"
    cat "$LOG_DIR/frontend.log"
fi
