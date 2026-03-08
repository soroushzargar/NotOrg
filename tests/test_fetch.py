"""Tests for the fetch command (LLM calls are mocked)."""

import pytest

from notorg import mindmap, storage
from notorg.fetch import fetch_ideas


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    import notorg.config as cfg_mod
    cfg_mod._DATA_DIR = tmp_path
    cfg_mod.PENDING_FILE = tmp_path / "pending.json"
    cfg_mod.MINDMAP_FILE = tmp_path / "mindmap.json"
    import importlib
    import notorg.storage as st_mod
    importlib.reload(st_mod)
    yield tmp_path


@pytest.fixture()
def filled_mindmap(isolated_data_dir):
    """Build a small populated mind-map."""
    tree = {"_ideas": [], "branches": {}}
    mindmap.insert_idea(tree, ["research", "conformal_prediction", "llm"], {
        "id": 1, "text": "Use CP on LLM graph structures", "timestamp": "t1"
    })
    mindmap.insert_idea(tree, ["research", "conformal_prediction"], {
        "id": 2, "text": "CP theory overview", "timestamp": "t2"
    })
    mindmap.insert_idea(tree, ["engineering", "web"], {
        "id": 3, "text": "Build a React dashboard", "timestamp": "t3"
    })
    storage.save_mindmap(tree)
    return tree


def _make_mock_ollama(relevant_paths, matching_ids):
    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_json(self, prompt):
            if "relevant_paths" in prompt or "branch" in prompt.lower():
                return {"relevant_paths": relevant_paths}
            return {"matching_ids": matching_ids}

    return lambda: _FakeClient()


def test_fetch_empty_mindmap(isolated_data_dir, monkeypatch, capsys):
    monkeypatch.setattr("notorg.fetch.OllamaClient", _make_mock_ollama([], []))
    results = fetch_ideas("anything", verbose=True)
    assert results == []
    assert "empty" in capsys.readouterr().out.lower()


def test_fetch_returns_relevant_ideas(filled_mindmap, monkeypatch):
    relevant = [["research", "conformal_prediction"], ["research", "conformal_prediction", "llm"]]
    matching = [1, 2]
    monkeypatch.setattr("notorg.fetch.OllamaClient", _make_mock_ollama(relevant, matching))

    results = fetch_ideas("conformal prediction")

    result_ids = {r["idea"]["id"] for r in results}
    assert 1 in result_ids
    assert 2 in result_ids
    assert 3 not in result_ids


def test_fetch_no_matches(filled_mindmap, monkeypatch):
    monkeypatch.setattr("notorg.fetch.OllamaClient", _make_mock_ollama([], []))
    results = fetch_ideas("quantum computing")
    assert results == []


def test_fetch_empty_query_raises(filled_mindmap):
    with pytest.raises(ValueError):
        fetch_ideas("")
