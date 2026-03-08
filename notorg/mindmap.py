"""Mind-map tree operations.

The mind-map is a nested JSON object with the following shape::

    {
        "_ideas": [{"id": 1, "text": "...", "timestamp": "..."}],
        "branches": {
            "research": {
                "_ideas": [],
                "branches": {
                    "conformal_prediction": {
                        "_ideas": [],
                        "branches": {
                            "llm": {
                                "_ideas": [{"id": 2, "text": "...", ...}],
                                "branches": {}
                            }
                        }
                    }
                }
            }
        }
    }

Paths are represented as lists of strings, e.g. ``["research",
"conformal_prediction", "llm"]``.
"""

from __future__ import annotations

from typing import Generator


def _empty_node() -> dict:
    return {"_ideas": [], "branches": {}}


# ------------------------------------------------------------------
# Node access
# ------------------------------------------------------------------

def get_node(tree: dict, path: list[str]) -> dict | None:
    """Return the node at *path*, or ``None`` if it does not exist."""
    node = tree
    for segment in path:
        branches = node.get("branches", {})
        if segment not in branches:
            return None
        node = branches[segment]
    return node


def ensure_node(tree: dict, path: list[str]) -> dict:
    """Return the node at *path*, creating missing nodes along the way."""
    node = tree
    for segment in path:
        branches = node.setdefault("branches", {})
        if segment not in branches:
            branches[segment] = _empty_node()
        node = branches[segment]
    return node


# ------------------------------------------------------------------
# Idea insertion
# ------------------------------------------------------------------

def insert_idea(tree: dict, path: list[str], idea: dict) -> None:
    """Insert *idea* at the node identified by *path*."""
    node = ensure_node(tree, path)
    node.setdefault("_ideas", [])
    # Avoid duplicates (same id)
    existing_ids = {i.get("id") for i in node["_ideas"]}
    if idea.get("id") not in existing_ids:
        node["_ideas"].append(idea)


# ------------------------------------------------------------------
# Tree traversal
# ------------------------------------------------------------------

def all_paths(tree: dict) -> list[list[str]]:
    """Return every branch path in the tree (depth-first)."""
    result: list[list[str]] = []

    def _walk(node: dict, current: list[str]) -> None:
        result.append(list(current))
        for name, child in node.get("branches", {}).items():
            _walk(child, current + [name])

    _walk(tree, [])
    return result


def all_ideas(tree: dict) -> Generator[tuple[list[str], dict], None, None]:
    """Yield *(path, idea)* for every idea in the tree (depth-first)."""

    def _walk(node: dict, current: list[str]) -> Generator:
        for idea in node.get("_ideas", []):
            yield list(current), idea
        for name, child in node.get("branches", {}).items():
            yield from _walk(child, current + [name])

    yield from _walk(tree, [])


def collect_ideas_under(tree: dict, path: list[str]) -> list[dict]:
    """Return all ideas in the subtree rooted at *path*."""
    node = get_node(tree, path)
    if node is None:
        return []
    return [idea for _, idea in all_ideas(node)]


def branch_summary(tree: dict) -> str:
    """Return a human-readable summary of all branches, one per line.

    Example output::

        (root)
        research
        research -> conformal_prediction
        research -> conformal_prediction -> llm
        engineering
    """
    lines: list[str] = []
    for path in all_paths(tree):
        if not path:
            lines.append("(root)")
        else:
            lines.append(" -> ".join(path))
    return "\n".join(lines)
