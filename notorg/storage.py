"""Low-level JSON storage helpers for pending ideas and the mind-map."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notorg import config


# ------------------------------------------------------------------
# Pending (unorganised) ideas
# ------------------------------------------------------------------

def _pending_path() -> Path:
    config.data_dir()
    return config.PENDING_FILE


def load_pending() -> list[dict]:
    """Return the list of pending ideas (may be empty)."""
    path = _pending_path()
    if not path.exists():
        return []
    with path.open() as fh:
        return json.load(fh)


def save_pending(ideas: list[dict]) -> None:
    """Overwrite the pending-ideas file with *ideas*."""
    with _pending_path().open("w") as fh:
        json.dump(ideas, fh, indent=2)


def append_pending(text: str) -> dict:
    """Append a new idea to the pending file and return the new entry."""
    ideas = load_pending()
    entry: dict[str, Any] = {
        "id": _new_id(ideas),
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ideas.append(entry)
    save_pending(ideas)
    return entry


def clear_pending() -> None:
    """Remove all pending ideas (after they have been organised)."""
    save_pending([])


# ------------------------------------------------------------------
# Mind-map file
# ------------------------------------------------------------------

def _mindmap_path() -> Path:
    config.data_dir()
    return config.MINDMAP_FILE


def load_mindmap() -> dict:
    """Return the mind-map tree (creates an empty one if not found)."""
    path = _mindmap_path()
    if not path.exists():
        return {"_ideas": [], "branches": {}}
    with path.open() as fh:
        return json.load(fh)


def save_mindmap(tree: dict) -> None:
    """Overwrite the mind-map file with *tree*."""
    with _mindmap_path().open("w") as fh:
        json.dump(tree, fh, indent=2)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _new_id(existing: list[dict]) -> int:
    if not existing:
        return 1
    return max(e.get("id", 0) for e in existing) + 1
