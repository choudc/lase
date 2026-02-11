# LASE - Local Autonomous Software Engineer

LASE (Local Autonomous Software Engineer) is a fully local, privacy-preserving autonomous agent that transforms natural language tasks into finished, shippable software artifacts. Built with a focus on local execution, LASE provides a robust UI and auditable execution trace while maintaining complete privacy and control over your development process.

## 🚀 Features

### Core Capabilities
- **Autonomous Task Execution**: Plan → Choose Tool → Execute → Observe → Update Plan loop
- **Local-First Architecture**: No cloud dependencies by default, complete privacy control
- **Multi-Modal Support**: Text, code, and visual processing capabilities
- **Comprehensive Tooling**: Shell, filesystem, git, browser, packaging, and documentation tools
- **Real-Time Monitoring**: Live execution logs, progress tracking, and session replay
- **Flexible Deployment**: Docker support, systemd integration, and development mode

### Agent Modes
- **Chat Mode**: Interactive conversation with the agent
- **Agent Mode**: Fully autonomous execution without supervision
- **Adaptive Mode**: Intelligently selects between chat and agent modes based on task complexity

### Security & Privacy
- **Offline by Default**: Network access disabled unless explicitly allowed
- **Sandboxed Execution**: Rootless Docker containers with resource limits
- **Credential Management**: Secure storage with OS keyring integration
- **Audit Trail**: Complete execution logs with replay capability

## 📋 Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04+ (other Linux distributions supported with manual adjustments)
- **Python**: 3.8 or higher
- **Node.js**: 18.0 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space minimum
- **CPU**: Multi-core processor recommended for parallel execution

### Optional Dependencies
- **Docker**: For enhanced sandboxing and containerized execution
- **Ollama**: For local LLM support (recommended)
- **NVIDIA Container Toolkit**: For GPU acceleration (if available)

## 🛠 Installation

### Quick Start (Recommended)

1. **Clone or extract LASE to your desired directory**
2. **Run the deployment script**:
   ```bash
   ./deploy.sh
   ```
3. **Start LASE**:
   ```bash
   ./start.sh
   ```
4. **Open your browser** and navigate to `http://localhost:5000`

### Development Setup

For development with hot-reload:

```bash
./deploy.sh --dev
./dev.sh
```

This will start:
- Backend API server at `http://localhost:5000`
- Frontend development server at `http://localhost:5173`

### Manual Installation

If you prefer manual setup or need to customize the installation:

#### Backend Setup
```bash
cd lase-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd lase-frontend
pnpm install
pnpm run build
cp -r dist ../lase-backend/src/static
```

#### Start Services
```bash
cd lase-backend
source venv/bin/activate
python src/main.py
```

## 🎯 Usage

### Creating Your First Task

1. **Start LASE** using `./start.sh` or the development script
2. **Open the web interface** at `http://localhost:5000`
3. **Create a new session** by clicking the "New" button in the sessions panel
4. **Describe your task** in the text area, for example:
   - "Build me a Flask dashboard for BTC/ETH with TA overlays"
   - "Create a React todo app with local storage"
   - "Write a Python CLI tool for file organization"
5. **Click Execute** to start the autonomous execution

### Understanding the Interface

#### Left Panel - Sessions
- View all your LASE sessions
- Create new sessions for different projects
- Switch between active sessions
- Monitor session status and task counts

#### Center Panel - Main Workspace
- **Task Input**: Describe what you want LASE to build
- **Tasks Tab**: View all tasks with progress indicators and status
- **Timeline Tab**: Real-time execution logs and events
- **Models Tab**: Available LLM models and their status
- **Tools Tab**: All available tools and their capabilities

#### Right Panel - Development Tools
- **Files Tab**: Browse workspace files and project structure
- **Terminal Tab**: Command-line interface for manual operations
- **Browser Tab**: Web browser for testing and research

### Task Examples

#### Web Application
```
Create a modern React dashboard for cryptocurrency tracking with:
- Real-time price charts for BTC, ETH, and top 10 coins
- Portfolio tracking with profit/loss calculations
- Dark/light theme toggle
- Responsive design for mobile and desktop
- Local storage for user preferences
```

#### Python CLI Tool
```
Build a Python command-line tool that:
- Organizes files in a directory by type and date
- Supports custom rules via configuration file
- Includes comprehensive logging and error handling
- Has unit tests with >90% coverage
- Generates detailed reports of operations performed
```

#### Research Report
```
Research and write a comprehensive report on:
- Current state of local AI development tools
- Comparison of different autonomous agent architectures
- Privacy implications of cloud vs local AI systems
- Include citations, diagrams, and executive summary
- Generate both Markdown and PDF versions
```

## ⚙️ Configuration

### Main Configuration File

LASE uses a YAML configuration file located at `~/.lase/config/lase.yaml`:

```yaml
# Model Configuration
models:
  reason: ollama:llama2        # For planning and reasoning
  code: ollama:codellama       # For code generation
  vision: ollama:llava         # For image processing

# Model Routing Rules
router:
  rules:
    - when: phase == "plan"
      use: reason
    - when: tool == "editor.apply_patch"
      use: code
    - when: needs_vision == true
      use: vision

# API Endpoints
endpoints:
  ollama: http://localhost:11434
  openai: http://localhost:8000/v1

# Sandbox Configuration
sandbox:
  network: off               # Network disabled by default
  cpu: 6                    # CPU cores limit
  mem_gb: 16               # Memory limit in GB
  disk_gb: 30              # Disk space limit in GB

# UI Configuration
ui:
  sse_buffer: punctuation   # Server-sent events buffering
  theme: dark              # UI theme (dark/light)

default_model: reason
```

### Environment Variables

You can override configuration using environment variables:

```bash
export LASE_OLLAMA_ENDPOINT=http://localhost:11434
export LASE_OPENAI_ENDPOINT=http://localhost:8000/v1
export LASE_DEFAULT_MODEL=reason
export LASE_NETWORK_ENABLED=false
```

### Model Management

#### Using Ollama (Recommended)

1. **Install Ollama**: Visit [https://ollama.ai/](https://ollama.ai/) for installation instructions
2. **Pull required models**:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull llava
   ```
3. **Verify models are running**:
   ```bash
   ollama list
   ```

#### Using OpenAI-Compatible APIs

For vLLM, llama.cpp, or other OpenAI-compatible endpoints:

```yaml
endpoints:
  openai: http://your-endpoint:8000/v1

models:
  reason: your-model-name
  code: your-code-model
```

## 🔧 Advanced Configuration

### Network Access Control

By default, LASE operates offline. To enable network access for specific tasks:

1. **Modify the session configuration** when creating a new session
2. **Use the allow-list feature** to specify permitted domains
3. **Monitor network activity** through the audit logs

### Custom Tools

LASE supports custom tool development. To add a new tool:

1. **Define the tool schema** in `src/core/tool_bus.py`
2. **Implement the execution logic** following the existing pattern
3. **Register the tool** in the tool registry
4. **Test the tool** using the validation endpoint

### Workspace Management

Each session gets its own isolated workspace with the following structure:

```
workspace/
├── src/                 # Source code
├── tests/              # Test files
├── docs/               # Documentation
├── scripts/            # Build and utility scripts
├── artifacts/          # Generated artifacts
├── memory/             # Agent memory and knowledge base
├── task_manifest.yaml  # Task metadata
├── plan.md            # Current execution plan
├── todo.md            # Task checklist
├── decisions.md       # Decision log
└── run_log.jsonl      # Execution log
```

## 🐳 Docker Deployment

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  lase-backend:
    build: ./lase-backend
    ports:
      - "5000:5000"
    volumes:
      - ./workspaces:/tmp/lase_workspaces
      - ~/.lase:/root/.lase
    environment:
      - LASE_NETWORK_ENABLED=false
    
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    
volumes:
  ollama_data:
```

Run with:
```bash
docker-compose up -d
```

### Standalone Docker

```bash
# Build the image
docker build -t lase ./lase-backend

# Run the container
docker run -d \
  --name lase \
  -p 5000:5000 \
  -v $(pwd)/workspaces:/tmp/lase_workspaces \
  -v ~/.lase:/root/.lase \
  lase
```

## 🔍 Monitoring and Debugging

### Logs and Debugging

- **Application logs**: Check `deployment.log` for installation issues
- **Runtime logs**: Session logs are stored in each workspace's `run_log.jsonl`
- **System logs**: Use `journalctl -u lase` if using systemd service

### Performance Monitoring

LASE includes built-in performance monitoring:

- **Task execution times** in the timeline view
- **Resource usage** displayed in the header
- **Model performance** metrics in the models tab

### Troubleshooting Common Issues

#### Backend Won't Start
```bash
# Check Python environment
cd lase-backend
source venv/bin/activate
python -c "import flask; print('Flask OK')"

# Check database
python -c "from src.models.session import db; print('Database OK')"
```

#### Frontend Build Fails
```bash
# Clear cache and reinstall
cd lase-frontend
rm -rf node_modules dist
pnpm install
pnpm run build
```

#### Models Not Loading
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Test model endpoint
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "messages": [{"role": "user", "content": "Hello"}]}'
```

## 🤝 Contributing

We welcome contributions to LASE! Here's how you can help:

### Development Setup

1. **Fork the repository** and clone your fork
2. **Set up development environment**:
   ```bash
   ./deploy.sh --dev
   ```
3. **Make your changes** following the coding standards
4. **Run tests**:
   ```bash
   ./deploy.sh --test
   ```
5. **Submit a pull request** with a clear description of your changes

### Code Structure

- `lase-backend/src/core/`: Core agent components
- `lase-backend/src/routes/`: API endpoints
- `lase-backend/src/models/`: Database models
- `lase-frontend/src/`: React frontend application

### Testing

- **Backend tests**: Located in `lase-backend/tests/`
- **Frontend tests**: Located in `lase-frontend/src/__tests__/`
- **Integration tests**: End-to-end testing scenarios

## 📄 License

LASE is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

LASE builds upon the excellent work of many open-source projects:

- **Flask** for the backend API framework
- **React** for the frontend user interface
- **Ollama** for local LLM support
- **Docker** for containerization and sandboxing
- **Tailwind CSS** for modern UI styling

## 📞 Support

### Getting Help

- **Documentation**: This README and inline code documentation
- **Issues**: Report bugs and request features via GitHub issues
- **Discussions**: Join community discussions for questions and ideas

### Reporting Issues

When reporting issues, please include:

1. **LASE version** and installation method
2. **Operating system** and version
3. **Steps to reproduce** the issue
4. **Expected vs actual behavior**
5. **Relevant log files** (sanitized of sensitive information)

### Security Issues

For security-related issues, please email the maintainers directly rather than opening a public issue.

---

**LASE - Empowering local, private, and autonomous software development.**

