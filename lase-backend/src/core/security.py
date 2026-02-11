import sys
import logging
import os

try:
    import keyring
    import keyrings.alt
    # Keyring defaults to enabled; set LASE_ENABLE_KEYRING=0 to disable explicitly.
    KEYRING_AVAILABLE = os.getenv("LASE_ENABLE_KEYRING", "1").strip().lower() in {"1", "true", "yes", "on"}
except ImportError:
    keyring = None
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Service name for keyring
SERVICE_NAME = "lase-ai-agent"

# Ensure we have a specialized backend if headles.
# For standard Linux servers without X11, keyrings.alt.file.PlaintextKeyring 
# might be used. We try to be safe.
# In a real production app, we might want to warn the user if using plaintext.

def _get_keyring_backend():
    # If we are in a headless environment, keyring might fail to find a backend
    # or hang trying to connect to dbus/secret-service.
    # We can perform a check or just let keyring handle it.
    
    # Priority:
    # 1. System Keyring (Gnome/KDE/macOS/Windows)
    # 2. Encrypted File (if configured - simpler: just use alt.file.PlaintextKeyring for now as fallback)
    if not KEYRING_AVAILABLE or keyring is None:
        return None

    backend_mode = os.getenv("LASE_KEYRING_BACKEND", "auto").strip().lower()
    if backend_mode == "plaintext":
        try:
            backend = keyrings.alt.file.PlaintextKeyring()
            keyring.set_keyring(backend)
            return backend
        except Exception as e:
            logger.warning(f"Failed to configure plaintext keyring backend: {e}")
            return None

    try:
        return keyring.get_keyring()
    except Exception as e:
        logger.warning(f"Failed to get keyring backend: {e}")
        return None

def set_api_key(provider: str, key: str) -> bool:
    """
    Securely saves the API key for the given provider.
    Returns True if successful.
    """
    if not key:
        if not KEYRING_AVAILABLE or keyring is None:
            return False
        try:
            keyring.delete_password(SERVICE_NAME, provider)
            return True
        except keyring.errors.PasswordDeleteError:
            return True # Already deleted
        except Exception as e:
            logger.error(f"Failed to delete key for {provider}: {e}")
            return False

    try:
        if KEYRING_AVAILABLE:
            if _get_keyring_backend() is None:
                return False
            keyring.set_password(SERVICE_NAME, provider, key)
            return True
        else:
             logger.warning(f"Keyring not available. Skipping secure storage for {provider}.")
             return False
    except Exception as e:
        logger.error(f"Failed to set key for {provider}: {e}")
        return False

def get_api_key(provider: str) -> str | None:
    """
    Retrieves the API key for the given provider.
    """
    try:
        if KEYRING_AVAILABLE:
            if _get_keyring_backend() is None:
                return None
            return keyring.get_password(SERVICE_NAME, provider)
        return None
    except Exception as e:
        logger.error(f"Failed to get key for {provider}: {e}")
        return None
