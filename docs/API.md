# API

Base URL: `/api`

## Sessions

### `GET /api/sessions`
Returns list of sessions.

### `POST /api/sessions`
Body:

```json
{
  "name": "Session name",
  "config": { "autonomy_level": "agent", "network_enabled": false }
}
```

## Tasks

### `GET /api/sessions/:session_id/tasks`
Returns tasks for a session (newest first).

### `POST /api/tasks`
Body:

```json
{
  "session_id": "uuid",
  "description": "Describe what to do",
  "auto_start": true
}
```

## Logs (Timeline)

### `GET /api/sessions/:session_id/logs?limit=50`
Returns most recent events for a session (chronological order).

## Models

### `GET /api/models`
Returns model entries derived from `lase-backend/src/config/models.yaml`.

## Tools

### `GET /api/tools`
Returns tool metadata entries used by the UI.

## Natural Language (Minimal)

### `POST /api/nl/intent`
Body:

```json
{ "text": "refactor the backend to be faster" }
```

## Code Quality (Minimal)

### `POST /api/quality/audit`
Body:

```json
{ "path": "/path/to/project" }
```

## Prediction (Minimal)

### `GET /api/predict`
Returns a rule-based risk summary from recent tasks (last 24h), including
stability/throughput warnings when thresholds are exceeded.
