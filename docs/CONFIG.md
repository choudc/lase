# Configuration

## Backend Environment Variables

- `LASE_HOST` (default `127.0.0.1`)
- `LASE_PORT` (default `5000`)
- `LASE_DEBUG` (default `0`)
- `LASE_DB_PATH` (default `lase-backend/src/database/app.db`)
- `LASE_WORKSPACES_DIR` (default `lase-backend/output`)
- `LASE_MODELS_YAML` (default `lase-backend/src/config/models.yaml`)
- `LASE_ENABLE_KEYRING` (default `1`) enables secure key storage via keyring
- `LASE_KEYRING_BACKEND` (optional) set to `plaintext` only for local debugging
- Optional API key env vars (for secret managers/CI):
  - `LASE_OPENAI_API_KEY`
  - `LASE_GEMINI_API_KEY`
  - `LASE_ANTHROPIC_API_KEY`
  - `LASE_DEEPSEEK_API_KEY`
  - `LASE_STABILITY_API_KEY`

## Model Routing

Edit `lase-backend/src/config/models.yaml`.

The backend uses the `default_models` mapping and a simple keyword-based classifier
to choose `coding|general|reasoning|vision|image_generation`.

### Stability Image Generation

Configure image generation under `stability_settings` in `models.yaml`:

- `base_url` (default `https://api.stability.ai`)
- `default_aspect_ratio` (e.g. `1:1`, `16:9`, `9:16`)
- `default_style_preset` (optional, e.g. `photographic`)
- `default_output_format` (`png|jpeg|webp`)
- `timeout`

Set the Stability API key in Settings UI (`/api/models/config`) as `stability_settings.api_key`.
Keys are stored through secure key storage (keyring) and are not persisted as plain text.
