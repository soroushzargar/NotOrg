"""--fetch command: semantic search over the mind-map using an LLM."""

from __future__ import annotations

import logging

from notorg import mindmap, storage
from notorg.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  LLM prompt helpers
# ------------------------------------------------------------------ #

_RELEVANCE_PROMPT = """You are a search assistant for a mind-map.

Query: "{query}"

Below are all branch paths in the mind-map, one per line:
{branches}

Return a JSON object with a single key "relevant_paths" whose value is a JSON
array of the branch paths (as arrays of strings) that are likely to contain
ideas relevant to the query.

Consider a branch relevant if its name OR the ideas it might contain relate to
the query topic — err on the side of inclusion.

Example response:
{{"relevant_paths": [["research", "conformal_prediction"], ["research", "conformal_prediction", "llm"]]}}

If no branches seem relevant, return {{"relevant_paths": []}}.
"""

_IDEA_FILTER_PROMPT = """You are a search assistant.

Query: "{query}"

Below is a list of ideas (JSON array). Return a JSON object with a single key
"matching_ids" whose value is an array of the integer IDs of ideas that are
relevant to the query.

Ideas:
{ideas_json}

If none match, return {{"matching_ids": []}}.
"""


def _paths_from_tree(tree: dict) -> list[list[str]]:
    """Return all non-root paths in the tree."""
    return [p for p in mindmap.all_paths(tree) if p]


def _ask_relevant_branches(
    client: OllamaClient, query: str, tree: dict
) -> list[list[str]]:
    """Ask the LLM which branches are relevant to *query*."""
    all_branch_paths = _paths_from_tree(tree)
    if not all_branch_paths:
        return []

    branches_text = "\n".join(" -> ".join(p) for p in all_branch_paths)
    prompt = _RELEVANCE_PROMPT.format(query=query, branches=branches_text)
    data = client.generate_json(prompt)

    if not isinstance(data, dict) or "relevant_paths" not in data:
        raise ValueError(f"Unexpected response shape: {data!r}")

    relevant = data["relevant_paths"]
    if not isinstance(relevant, list):
        return []

    result: list[list[str]] = []
    for item in relevant:
        if isinstance(item, list) and all(isinstance(s, str) for s in item):
            result.append([s.strip().lower() for s in item])
        elif isinstance(item, str):
            # LLM sometimes returns a flattened string
            result.append([s.strip().lower() for s in item.split("->")])
    return result


def _ask_filter_ideas(
    client: OllamaClient, query: str, ideas: list[dict]
) -> list[dict]:
    """Filter *ideas* to those relevant to *query* using the LLM."""
    if not ideas:
        return []

    import json

    ideas_json = json.dumps(
        [{"id": i.get("id"), "text": i.get("text")} for i in ideas],
        indent=2,
    )
    prompt = _IDEA_FILTER_PROMPT.format(query=query, ideas_json=ideas_json)
    data = client.generate_json(prompt)

    if not isinstance(data, dict) or "matching_ids" not in data:
        return ideas  # fallback: return all

    matching_ids = set(data["matching_ids"])
    return [i for i in ideas if i.get("id") in matching_ids]


# ------------------------------------------------------------------ #
#  Public entry point
# ------------------------------------------------------------------ #

def fetch_ideas(query: str, verbose: bool = False) -> list[dict]:
    """Search the mind-map for ideas relevant to *query*.

    Returns a list of result dicts::

        [{"path": ["research", "llm"], "idea": {"id": 1, "text": "...", ...}}, ...]
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    tree = storage.load_mindmap()
    all_root_ideas = list(mindmap.all_ideas(tree))
    if not all_root_ideas:
        if verbose:
            print("Mind-map is empty.")
        return []

    if verbose:
        print(f"Searching mind-map for: \"{query}\" …")

    with OllamaClient() as client:
        # Step 1: identify relevant branches
        try:
            relevant_paths = _ask_relevant_branches(client, query, tree)
        except Exception as exc:
            logger.warning("Branch relevance check failed: %s – scanning all.", exc)
            relevant_paths = _paths_from_tree(tree)

        # Collect candidate ideas from relevant branches
        # Include root-level ideas too
        candidate_ideas: list[tuple[list[str], dict]] = []

        # Root ideas
        for idea in tree.get("_ideas", []):
            candidate_ideas.append(([], idea))

        # Ideas from relevant subtrees (de-duplicated by id)
        seen_ids: set = {i.get("id") for _, i in candidate_ideas}
        for path in relevant_paths:
            for idea in mindmap.collect_ideas_under(tree, path):
                if idea.get("id") not in seen_ids:
                    candidate_ideas.append((path, idea))
                    seen_ids.add(idea.get("id"))

        if verbose:
            print(f"  Found {len(candidate_ideas)} candidate idea(s) in relevant branches.")

        # Step 2: fine-grained idea filtering
        flat_ideas = [idea for _, idea in candidate_ideas]
        path_by_id = {idea.get("id"): path for path, idea in candidate_ideas}

        try:
            matched = _ask_filter_ideas(client, query, flat_ideas)
        except Exception as exc:
            logger.warning("Idea filtering failed: %s – returning all candidates.", exc)
            matched = flat_ideas

    results = [
        {"path": path_by_id.get(idea.get("id"), []), "idea": idea}
        for idea in matched
    ]

    return results
