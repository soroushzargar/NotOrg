"""--organize command: move pending ideas into the mind-map using an LLM."""

from __future__ import annotations

import logging

from notorg import mindmap, storage
from notorg.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  LLM prompt helpers
# ------------------------------------------------------------------ #

_SYSTEM_CONTEXT = (
    "You are an intelligent mind-map organizer. "
    "Your job is to decide where a new idea belongs inside an existing "
    "hierarchical mind-map tree."
)

_PLACEMENT_PROMPT = """{system}

Current mind-map branches (each line is a branch path, root = top level):
{branches}

New idea to place:
\"{idea}\"

Rules:
1. Choose the most specific existing branch path that fits the idea.
2. If no existing branch fits, propose a new path (you may add new intermediate nodes).
3. Use lowercase, underscore-separated names for branch segments.
4. Path depth should be 1–4 levels.

Respond with a JSON object with a single key "path" whose value is a JSON array
of strings representing the branch path. Example:
{{"path": ["research", "conformal_prediction", "llm"]}}
"""


def _build_placement_prompt(idea_text: str, tree: dict) -> str:
    branches = mindmap.branch_summary(tree) or "(empty – no branches yet)"
    return _PLACEMENT_PROMPT.format(
        system=_SYSTEM_CONTEXT,
        branches=branches,
        idea=idea_text,
    )


def _ask_placement(client: OllamaClient, idea_text: str, tree: dict) -> list[str]:
    """Ask the LLM where *idea_text* belongs and return a path list."""
    prompt = _build_placement_prompt(idea_text, tree)
    data = client.generate_json(prompt)

    if not isinstance(data, dict) or "path" not in data:
        raise ValueError(f"Unexpected LLM response shape: {data!r}")

    path = data["path"]
    if not isinstance(path, list) or not all(isinstance(s, str) for s in path):
        raise ValueError(f"LLM returned invalid path: {path!r}")

    # Sanitise: lowercase and strip whitespace
    return [segment.strip().lower().replace(" ", "_") for segment in path if segment.strip()]


# ------------------------------------------------------------------ #
#  Public entry point
# ------------------------------------------------------------------ #

def organize_ideas(verbose: bool = False) -> list[dict]:
    """Organise all pending ideas into the mind-map.

    Returns a list of records describing where each idea was placed::

        [{"idea": {...}, "path": ["research", "llm"]}, ...]
    """
    pending = storage.load_pending()
    if not pending:
        if verbose:
            print("No pending ideas to organise.")
        return []

    if verbose:
        print(f"Organising {len(pending)} idea(s) …")

    results: list[dict] = []
    tree = storage.load_mindmap()

    with OllamaClient() as client:
        for idea in pending:
            text = idea["text"]
            if verbose:
                print(f"  • Placing: \"{text}\"")
            try:
                path = _ask_placement(client, text, tree)
            except Exception as exc:
                logger.warning("Could not place idea %r: %s – using root.", text, exc)
                path = []

            mindmap.insert_idea(tree, path, idea)
            results.append({"idea": idea, "path": path})
            if verbose:
                label = " -> ".join(path) if path else "(root)"
                print(f"    → {label}")

    storage.save_mindmap(tree)
    storage.clear_pending()

    if verbose:
        print("Done. Mind-map updated.")

    return results
