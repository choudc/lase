#!/bin/bash

# LASE Enhanced Self-Improvement System v0.5.0 Deployment Script
# This script deploys the complete LASE system with self-improvement capabilities
# Author: Manus AI
# License: MIT

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LASE_DIR="$(pwd)"

detect_project_dir() {
    local dir_name="$1"

    if [ -d "$LASE_DIR/$dir_name" ]; then
        echo "$LASE_DIR/$dir_name"
        return 0
    fi

    # Support archives that keep versioned directories (e.g. ./lase-frontend-v0.6.0/lase-frontend).
    # Pick the first match in lexical order to keep behavior deterministic.
    local match
    match="$(find "$LASE_DIR" -maxdepth 3 -type d -name "$dir_name" 2>/dev/null | sort | head -n 1 || true)"
    if [ -n "${match:-}" ]; then
        echo "$match"
        return 0
    fi

    return 1
}

BACKEND_DIR="$(detect_project_dir "lase-backend" || true)"
FRONTEND_DIR="$(detect_project_dir "lase-frontend" || true)"
LOG_FILE="$LASE_DIR/deployment.log"
PYTHON_VERSION="3.8"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8+ is required, found $PYTHON_VER"
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is required but not installed"
    fi
    
    NODE_VER=$(node --version)
    info "Found Node.js $NODE_VER"
    
    # Check npm/pnpm
    if command -v pnpm &> /dev/null; then
        PACKAGE_MANAGER="pnpm"
    elif command -v npm &> /dev/null; then
        PACKAGE_MANAGER="npm"
    else
        error "npm or pnpm is required but not installed"
    fi
    
    info "Using package manager: $PACKAGE_MANAGER"
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 8 ]; then
        warning "System has ${MEMORY_GB}GB RAM. 8GB+ recommended for optimal performance"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 10 ]; then
        warning "Available disk space: ${DISK_SPACE}GB. 10GB+ recommended"
    fi
    
    log "System requirements check completed"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv python3-dev build-essential curl git
    elif command -v yum &> /dev/null; then
        sudo yum update -y
        sudo yum install -y python3-pip python3-devel gcc gcc-c++ make curl git
    elif command -v brew &> /dev/null; then
        brew update
        brew install python3 node npm
    else
        warning "Could not detect package manager. Please install dependencies manually"
    fi
    
    log "System dependencies installed"
}

# Setup backend
setup_backend() {
    log "Setting up LASE backend..."
    
    if [ -z "${BACKEND_DIR:-}" ] || [ ! -d "$BACKEND_DIR" ]; then
        error "Backend directory not found. Expected ./lase-backend (or a versioned subdir containing it)."
    fi
    
    cd "$BACKEND_DIR"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        log "Installing Python dependencies..."
        pip install -r requirements.txt
    else
        log "Installing core dependencies..."
        pip install flask flask-cors flask-sqlalchemy requests psutil pyyaml
        pip install bandit pylint flake8 mypy safety
        pip install numpy pandas scikit-learn
    fi
    
    # Create necessary directories
    mkdir -p src/database
    mkdir -p src/config
    mkdir -p /tmp/lase_workspaces
    mkdir -p logs
    mkdir -p generated_images
    
    # Create models configuration if it doesn't exist
    if [ ! -f "src/config/models.yaml" ]; then
        log "Creating models configuration file..."
        cat > src/config/models.yaml << 'EOF'
default_models:
  coding:
    provider: ollama
    name: qwen2.5-coder:32b
  general:
    provider: ollama
    name: gemma3:27b
  reasoning:
    provider: ollama
    name: gemma3:27b
  vision:
    provider: ollama
    name: gemma3:27b
  image_generation:
    provider: local_sdxl
    name: stable-diffusion-xl-base-1.0

ollama_settings:
  base_url: http://localhost:11434
  api_key: ""
  timeout: 300

openai_settings:
  api_key: YOUR_OPENAI_API_KEY
  base_url: https://api.openai.com/v1
  timeout: 60

local_sdxl_settings:
  api_url: http://localhost:7860/sdapi/v1/
  default_width: 1024
  default_height: 1024
  default_steps: 20
  default_cfg_scale: 7.0
  default_sampler: "DPM++ 2M Karras"
  timeout: 120
EOF
        log "Models configuration created at src/config/models.yaml"
    else
        log "Models configuration already exists"
    fi
    
    # Initialize database
    log "Initializing database..."
    python3 -c "
from src.models.session import db
from src.main import app
with app.app_context():
    db.create_all()
    print(\'Database initialized successfully\')
"

    # Run database migration for last_output column
    log "Running database migration for last_output column..."
    python3 -c "
import sqlite3
import os

# In python -c, __file__ is not defined; use current working directory (backend root).
db_path = os.path.join(os.getcwd(), \'src\', \'database\', \'app.db\')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute(\"PRAGMA table_info(task)\")
    columns = [column[1] for column in cursor.fetchall()]
    if \'last_output\' not in columns:
        cursor.execute(\"ALTER TABLE task ADD COLUMN last_output TEXT\")
        print(\"Added last_output column to task table\")
    else:
        print(\"last_output column already exists\")
    conn.commit()
except Exception as e:
    print(f\"Error during migration: {e}\")
    conn.rollback()
finally:
    conn.close()
"
    
    log "Backend setup completed"
}

# Setup frontend
setup_frontend() {
    log "Setting up LASE frontend..."
    
    if [ -z "${FRONTEND_DIR:-}" ] || [ ! -d "$FRONTEND_DIR" ]; then
        error "Frontend directory not found. Expected ./lase-frontend (or a versioned subdir containing it)."
    fi
    
    cd "$FRONTEND_DIR"
    
    # Install dependencies
    log "Installing frontend dependencies..."
    $PACKAGE_MANAGER install
    
    # Build frontend
    log "Building frontend..."
    $PACKAGE_MANAGER run build
    
    # Copy built files to backend static directory
    log "Deploying frontend to backend..."
    rm -rf "$BACKEND_DIR/src/static"
    mkdir -p "$BACKEND_DIR/src/static"
    cp -r dist/* "$BACKEND_DIR/src/static/"
    
    log "Frontend setup completed"
}

# Configure services
configure_services() {
    log "Configuring LASE services..."
    
    # Create systemd service file
    if command -v systemctl &> /dev/null; then
        log "Creating systemd service..."
        
        cat > /tmp/lase.service << EOF
[Unit]
Description=LASE Enhanced Self-Improvement System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$BACKEND_DIR/venv/bin
ExecStart=$BACKEND_DIR/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo mv /tmp/lase.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable lase
        
        log "Systemd service configured"
    fi
    
    # Create startup script
    cat > "$LASE_DIR/start.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/lase-backend"

echo "Starting LASE Enhanced Self-Improvement System..."

cd "$BACKEND_DIR"
source venv/bin/activate

# Check if already running
if pgrep -f "python src/main.py" > /dev/null; then
    echo "LASE is already running"
    echo "Access the interface at: http://localhost:5000"
    exit 0
fi

# Start the backend
echo "Starting backend server..."
nohup python src/main.py > logs/lase.log 2>&1 &

# Wait for server to start
sleep 5

if pgrep -f "python src/main.py" > /dev/null; then
    echo "✅ LASE started successfully!"
    echo "🌐 Access the interface at: http://localhost:5000"
    echo "📊 Monitor logs: tail -f $BACKEND_DIR/logs/lase.log"
    echo "🛑 Stop with: ./stop.sh"
else
    echo "❌ Failed to start LASE. Check logs: $BACKEND_DIR/logs/lase.log"
    exit 1
fi
EOF
    
    # Create stop script
    cat > "$LASE_DIR/stop.sh" << 'EOF'
#!/bin/bash

echo "Stopping LASE Enhanced Self-Improvement System..."

# Kill the backend process
pkill -f "python src/main.py" 2>/dev/null || true

# Wait for process to stop
sleep 2

if ! pgrep -f "python src/main.py" > /dev/null; then
    echo "✅ LASE stopped successfully"
else
    echo "⚠️  Force killing LASE processes..."
    pkill -9 -f "python src/main.py" 2>/dev/null || true
    echo "✅ LASE force stopped"
fi
EOF
    
    # Create status script
    cat > "$LASE_DIR/status.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/lase-backend"

echo "LASE Enhanced Self-Improvement System Status"
echo "============================================"

if pgrep -f "python src/main.py" > /dev/null; then
    PID=$(pgrep -f "python src/main.py")
    echo "Status: ✅ Running (PID: $PID)"
    echo "Interface: http://localhost:5000"
    
    # Check if port is accessible
    if curl -s http://localhost:5000/api/sessions > /dev/null 2>&1; then
        echo "API: ✅ Accessible"
    else
        echo "API: ❌ Not accessible"
    fi
    
    # Show resource usage
    echo "Resource Usage:"
    ps -p $PID -o pid,ppid,cmd,%mem,%cpu --no-headers
    
else
    echo "Status: ❌ Not running"
fi

echo ""
echo "Recent logs:"
if [ -f "$BACKEND_DIR/logs/lase.log" ]; then
    tail -5 "$BACKEND_DIR/logs/lase.log"
else
    echo "No logs found"
fi
EOF
    
    # Make scripts executable
    chmod +x "$LASE_DIR/start.sh"
    chmod +x "$LASE_DIR/stop.sh"
    chmod +x "$LASE_DIR/status.sh"
    
    log "Service configuration completed"
}

# Run tests
run_tests() {
    log "Running system tests..."
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    # Test imports
    log "Testing module imports..."
    python3 -c "
try:
    from src.core.code_quality_auditor import CodeQualityAuditor
    from src.core.predictive_analyzer import PredictiveAnalyzer
    from src.core.natural_language_interface import NaturalLanguageInterface
    print('✅ All enhanced modules imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    # Test basic functionality
    log "Testing basic functionality..."
    timeout 30 python3 -c "
import sys
sys.path.insert(0, '.')
from src.main import app
with app.test_client() as client:
    response = client.get('/api/sessions')
    if response.status_code == 200:
        print('✅ API endpoints accessible')
    else:
        print(f'❌ API test failed: {response.status_code}')
        exit(1)
"
    
    log "Tests completed successfully"
}

# Create documentation
create_docs() {
    log "Creating documentation..."
    
    # Create quick start guide
    cat > "$LASE_DIR/QUICK_START.md" << 'EOF'
# LASE Enhanced Self-Improvement System - Quick Start

## Starting LASE

```bash
./start.sh
```

## Accessing the Interface

Open your browser and navigate to: http://localhost:5000

## Basic Usage

1. **Create a Session**: Click "New" to create a new work session
2. **Natural Language Tasks**: Describe what you want to accomplish in plain English
3. **Self-Improvement**: Use phrases like "make the system faster" or "fix memory leaks"
4. **Monitor Progress**: Use the Tasks and Timeline tabs to track progress

## Enhanced Features

### Natural Language Interface
- "Improve system performance"
- "Fix the bug in database connections"
- "Add better error handling"
- "Optimize memory usage"

### Code Quality Auditing
- Automatic security scanning with Bandit
- Code quality analysis with Pylint
- Style checking with Flake8
- Type analysis with MyPy
- Dependency vulnerability scanning with Safety

### Predictive Analytics
- Performance trend analysis
- Resource usage prediction
- Anomaly detection
- Proactive problem identification

## Stopping LASE

```bash
./stop.sh
```

## Checking Status

```bash
./status.sh
```

## Logs

View real-time logs:
```bash
tail -f lase-backend/logs/lase.log
```

## Support

- Documentation: See LASE_Enhanced_Documentation.md
- Issues: Check the troubleshooting section in the documentation
- Community: Join the LASE community forums
EOF
    
    log "Documentation created"
}

# Main deployment function
main() {
    log "Starting LASE Enhanced Self-Improvement System deployment..."
    log "Deployment directory: $LASE_DIR"
    
    # Create log file
    touch "$LOG_FILE"
    
    # Run deployment steps
    check_requirements
    install_system_deps
    setup_backend
    setup_frontend
    configure_services
    run_tests
    create_docs
    
    log "🎉 LASE Enhanced Self-Improvement System deployed successfully!"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETE                      ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  LASE Enhanced Self-Improvement System v0.5.0               ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║  🚀 Start LASE:    ./start.sh                               ║${NC}"
    echo -e "${GREEN}║  🌐 Interface:     http://localhost:5000                    ║${NC}"
    echo -e "${GREEN}║  📊 Status:        ./status.sh                              ║${NC}"
    echo -e "${GREEN}║  🛑 Stop LASE:     ./stop.sh                                ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║  📚 Documentation: LASE_Enhanced_Documentation.md           ║${NC}"
    echo -e "${GREEN}║  🚀 Quick Start:   QUICK_START.md                           ║${NC}"
    echo -e "${GREEN}║  📝 Logs:          tail -f lase-backend/logs/lase.log       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Enhanced Features:${NC}"
    echo -e "  ✅ Advanced Code Quality Auditing (Bandit, Pylint, Flake8, MyPy, Safety)"
    echo -e "  ✅ Proactive Problem Prediction with Machine Learning"
    echo -e "  ✅ Natural Language Interface for Self-Improvement"
    echo -e "  ✅ Comprehensive Monitoring and Analytics"
    echo -e "  ✅ Automated Testing and Validation"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "  1. Run ${GREEN}./start.sh${NC} to start LASE"
    echo -e "  2. Open ${GREEN}http://localhost:5000${NC} in your browser"
    echo -e "  3. Try natural language commands like ${GREEN}'Make the system faster'${NC}"
    echo -e "  4. Explore the enhanced self-improvement capabilities"
    echo ""
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"
