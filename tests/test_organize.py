"""Tests for the organize command (LLM calls are mocked)."""

import pytest

from notorg import mindmap, storage
from notorg.organize import organize_ideas


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
def mock_ollama(monkeypatch):
    """Replace OllamaClient with a stub that places all ideas under ['research', 'testing']."""

    class _FakeClient:
        model = "test-model"
        host = "http://localhost:11434"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_json(self, prompt):
            return {"path": ["research", "testing"]}

    monkeypatch.setattr("notorg.organize.OllamaClient", lambda: _FakeClient())
    return _FakeClient


def test_organize_no_pending(isolated_data_dir, mock_ollama, capsys):
    results = organize_ideas(verbose=True)
    assert results == []
    out = capsys.readouterr().out
    assert "No pending" in out


def test_organize_places_ideas(isolated_data_dir, mock_ollama):
    storage.append_pending("Idea about testing LLMs")
    storage.append_pending("Another test idea")

    results = organize_ideas()

    assert len(results) == 2
    for r in results:
        assert r["path"] == ["research", "testing"]

    # Mind-map should have been updated
    tree = storage.load_mindmap()
    node = mindmap.get_node(tree, ["research", "testing"])
    assert node is not None
    assert len(node["_ideas"]) == 2


def test_organize_clears_pending(isolated_data_dir, mock_ollama):
    storage.append_pending("Some idea")
    organize_ideas()
    assert storage.load_pending() == []


def test_organize_falls_back_to_root_on_bad_response(isolated_data_dir, monkeypatch):
    """If the LLM returns garbage, the idea is placed at root."""

    class _BadClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_json(self, _prompt):
            return {"unexpected_key": "oops"}

    monkeypatch.setattr("notorg.organize.OllamaClient", lambda: _BadClient())

    storage.append_pending("Fallback idea")
    results = organize_ideas()

    assert len(results) == 1
    assert results[0]["path"] == []

    tree = storage.load_mindmap()
    root_ideas = tree.get("_ideas", [])
    assert any(i["text"] == "Fallback idea" for i in root_ideas)
