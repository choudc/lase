# Architecture

## Components

- Frontend (`lase-frontend/`): React UI that talks to the backend via `/api/*`.
- Backend (`lase-backend/`): Flask app that provides:
  - sessions + workspaces
  - tasks + orchestrated execution
  - timeline logs/events
  - models + tools metadata

## Data Model

SQLite database: `lase-backend/src/database/app.db`

- `Session`
  - `id` (uuid)
  - `name`
  - `status`
  - `config_json`
  - `workspace_path`
- `Task`
  - `id` (uuid)
  - `session_id`
  - `description`
  - `status`: `queued|running|completed|failed|stopped`
  - `progress`: `0..1`
  - `last_output`: last significant output payload as text
- `LogEntry`
  - `event_type`: includes `task_started`, `progress_update`, `task_completed`, `task_failed`
  - `data_json`

## Task Execution Loop

Implemented in `lase-backend/src/core/orchestrator.py`:

1. classify intent (`coding|general|reasoning|vision|image_generation`)
2. pick model based on `lase-backend/src/config/models.yaml`
3. generate a deterministic plan (auditable)
4. execute steps
5. after each step:
   - update `Task.progress`
   - capture `Task.last_output`
   - emit `LogEntry` with `event_type=progress_update`

