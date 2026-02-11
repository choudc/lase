# LASE (Local Autonomous Software Engineer)

Local-first autonomous task runner with a web UI.

## What’s In This Repo

- `lase-backend/`: Flask API + SQLite storage + lightweight agent orchestrator
- `lase-frontend/`: Vite + React UI
- `docs/legacy/`: original v0.5 design/docs that this implementation follows

## Quick Start (Production Mode)

1. `./deploy.sh`
2. `./start.sh` (foreground) or `./start.sh --daemon` (background)
3. Open `http://localhost:5000`

To stop: `./stop.sh`  
To check: `./status.sh`

Note: `./deploy.sh` runs `pnpm install` by default. If you are offline but already have `node_modules/`,
use `./deploy.sh --no-install` to rebuild without fetching packages.

## Development Mode (Hot Reload)

Run:

```bash
./dev.sh
```

- Backend: `http://localhost:5000`
- Frontend: `http://localhost:5173` (proxies `/api` to the backend)

## API (Used By The UI)

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/:id/tasks`
- `GET /api/sessions/:id/logs?limit=50`
- `POST /api/tasks`
- `GET /api/models`
- `GET /api/tools`

## Design Notes

This code preserves the key features described in `docs/legacy/LASE_Enhanced_Documentation.md`:

- session isolation via per-session workspace directories
- Plan -> Execute -> Observe loop with frequent progress updates
- task `last_output` capture
- timeline events (including `progress_update`)
- model routing driven by `lase-backend/src/config/models.yaml`
