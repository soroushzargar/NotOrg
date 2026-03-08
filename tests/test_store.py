"""Tests for the store command and underlying storage layer."""

import pytest

from notorg import storage, store


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all file I/O to a temporary directory."""
    monkeypatch.setenv("NOTORG_DATA_DIR", str(tmp_path))
    # Reload config module-level paths
    import importlib
    import notorg.config as cfg_mod
    cfg_mod._DATA_DIR = tmp_path
    cfg_mod.PENDING_FILE = tmp_path / "pending.json"
    cfg_mod.MINDMAP_FILE = tmp_path / "mindmap.json"
    import notorg.storage as st_mod
    importlib.reload(st_mod)
    yield tmp_path


# ------------------------------------------------------------------
# storage layer
# ------------------------------------------------------------------

def test_load_pending_empty(isolated_data_dir):
    assert storage.load_pending() == []


def test_append_and_load_pending(isolated_data_dir):
    entry = storage.append_pending("First idea")
    assert entry["text"] == "First idea"
    assert entry["id"] == 1
    assert "timestamp" in entry

    pending = storage.load_pending()
    assert len(pending) == 1
    assert pending[0]["text"] == "First idea"


def test_append_multiple_pending(isolated_data_dir):
    storage.append_pending("Idea A")
    storage.append_pending("Idea B")
    pending = storage.load_pending()
    assert len(pending) == 2
    ids = [p["id"] for p in pending]
    assert ids == sorted(ids)
    assert ids[0] < ids[1]


def test_clear_pending(isolated_data_dir):
    storage.append_pending("temp idea")
    storage.clear_pending()
    assert storage.load_pending() == []


def test_load_mindmap_empty(isolated_data_dir):
    tree = storage.load_mindmap()
    assert tree == {"_ideas": [], "branches": {}}


def test_save_and_load_mindmap(isolated_data_dir):
    tree = {"_ideas": [{"id": 1, "text": "x"}], "branches": {"a": {"_ideas": [], "branches": {}}}}
    storage.save_mindmap(tree)
    loaded = storage.load_mindmap()
    assert loaded == tree


# ------------------------------------------------------------------
# store command
# ------------------------------------------------------------------

def test_store_idea(isolated_data_dir):
    entry = store.store_idea("A great idea")
    assert entry["text"] == "A great idea"
    # Verify it was persisted
    pending = storage.load_pending()
    assert any(p["id"] == entry["id"] for p in pending)


def test_store_idea_empty_raises(isolated_data_dir):
    with pytest.raises(ValueError):
        store.store_idea("")


def test_store_idea_whitespace_raises(isolated_data_dir):
    with pytest.raises(ValueError):
        store.store_idea("   ")
