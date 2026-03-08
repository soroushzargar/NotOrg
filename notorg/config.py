"""Configuration management for NotOrg.

All persistent data lives under ``~/.notorg/``.
"""

import json
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_DATA_DIR = Path(os.environ.get("NOTORG_DATA_DIR", Path.home() / ".notorg"))
_CONFIG_FILE = _DATA_DIR / "config.json"

PENDING_FILE = _DATA_DIR / "pending.json"
MINDMAP_FILE = _DATA_DIR / "mindmap.json"

# Default Ollama settings
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

_DEFAULTS: dict = {
    "ollama_model": DEFAULT_OLLAMA_MODEL,
    "ollama_host": DEFAULT_OLLAMA_HOST,
}


def data_dir() -> Path:
    """Return (and create if needed) the data directory."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def load() -> dict:
    """Return the current configuration, merging defaults with on-disk values."""
    data_dir()
    if _CONFIG_FILE.exists():
        with _CONFIG_FILE.open() as fh:
            user = json.load(fh)
    else:
        user = {}
    cfg = dict(_DEFAULTS)
    cfg.update(user)
    return cfg


def save(cfg: dict) -> None:
    """Persist *cfg* to disk."""
    data_dir()
    with _CONFIG_FILE.open("w") as fh:
        json.dump(cfg, fh, indent=2)
