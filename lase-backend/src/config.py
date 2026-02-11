import os
import yaml
import shutil
import time
from dataclasses import dataclass
from .core import security

@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    debug: bool
    database_path: str
    workspaces_dir: str
    models_config_path: str
    generated_images_dir: str
    logs_dir: str
    static_dir: str


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: str) -> Settings:
    # project_root is lase-backend/
    host = os.getenv("LASE_HOST", "127.0.0.1")
    port = int(os.getenv("LASE_PORT", "5000"))
    debug = _env_bool("LASE_DEBUG", False)

    data_root = os.path.join(project_root, "src")
    database_path = os.getenv("LASE_DB_PATH", os.path.join(data_root, "database", "app.db"))
    workspaces_dir = os.getenv("LASE_WORKSPACES_DIR", os.path.join(project_root, "output"))
    models_config_path = os.getenv("LASE_MODELS_YAML", os.path.join(data_root, "config", "models.yaml"))
    generated_images_dir = os.getenv("LASE_GENERATED_IMAGES_DIR", os.path.join(project_root, "generated_images"))
    logs_dir = os.getenv("LASE_LOGS_DIR", os.path.join(project_root, "logs"))
    static_dir = os.getenv("LASE_STATIC_DIR", os.path.join(data_root, "static"))

    return Settings(
        host=host,
        port=port,
        debug=debug,
        database_path=database_path,
        workspaces_dir=workspaces_dir,
        models_config_path=models_config_path,
        generated_images_dir=generated_images_dir,
        logs_dir=logs_dir,
        static_dir=static_dir,
    )


def get_default_config():
    return {
        "default_models": {
            "coding": {"provider": "ollama", "name": "qwen2.5-coder:32b"},
            "general": {"provider": "ollama", "name": "gemma3:27b"},
            "reasoning": {"provider": "ollama", "name": "gemma3:27b"},
            "vision": {"provider": "ollama", "name": "gemma3:27b"},
            "image_generation": {"provider": "stability", "name": "stable-image-core"},
        },
        "ollama_settings": {"base_url": "http://localhost:11434", "api_key": "", "timeout": 300},
        "openai_settings": {"api_key": "YOUR_OPENAI_API_KEY", "base_url": "https://api.openai.com/v1", "timeout": 60},
        "gemini_settings": {"api_key": "YOUR_GEMINI_API_KEY", "timeout": 60},
        "anthropic_settings": {"api_key": "YOUR_ANTHROPIC_API_KEY", "timeout": 60},
        "deepseek_settings": {"api_key": "YOUR_DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "timeout": 60},
        "stability_settings": {
            "api_key": "YOUR_STABILITY_API_KEY",
            "base_url": "https://api.stability.ai",
            "default_aspect_ratio": "1:1",
            "default_style_preset": "",
            "default_output_format": "png",
            "timeout": 120,
        },
    }

def load_config_safely(path):
    # Try loading primary
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or get_default_config()
    except Exception as e:
        print(f"Config corruption detected at {path}: {e}")

    # Try backup
    backup_path = path + ".bak"
    try:
        if os.path.exists(backup_path):
            print(f"Attempting to recover from backup {backup_path}...")
            with open(backup_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                # Restore backup to primary
                shutil.copy2(backup_path, path)
                print("Restored config from backup.")
                return data
    except Exception as e:
        print(f"Backup also corrupted: {e}")

    # Reset to default
    print("Resetting config to defaults.")
    defaults = get_default_config()
    save_config_safely(path, defaults)
    return defaults

def save_config_safely(path, config):
    # Atomic write: write to tmp, backup existing, move tmp to primary
    tmp_path = path + ".tmp"
    backup_path = path + ".bak"
    
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
            
        if os.path.exists(path):
            shutil.copy2(path, backup_path)
            
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"Failed to save config safely: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
