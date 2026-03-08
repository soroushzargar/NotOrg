"""Tests for the CLI entry point."""

import json
import pytest

from notorg.cli import main


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


# ------------------------------------------------------------------
# --store
# ------------------------------------------------------------------

def test_cli_store(isolated_data_dir, capsys):
    rc = main(["--store", "Great idea"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Great idea" in out
    assert "Stored" in out


def test_cli_store_empty_fails(isolated_data_dir, capsys):
    rc = main(["--store", ""])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error" in err


# ------------------------------------------------------------------
# --list-pending
# ------------------------------------------------------------------

def test_cli_list_pending_empty(isolated_data_dir, capsys):
    rc = main(["--list-pending"])
    assert rc == 0
    assert "No pending" in capsys.readouterr().out


def test_cli_list_pending_shows_ideas(isolated_data_dir, capsys):
    main(["--store", "Idea 1"])
    main(["--store", "Idea 2"])
    capsys.readouterr()  # clear previous output
    rc = main(["--list-pending"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Idea 1" in out
    assert "Idea 2" in out


# ------------------------------------------------------------------
# --show-mindmap
# ------------------------------------------------------------------

def test_cli_show_mindmap_empty(isolated_data_dir, capsys):
    rc = main(["--show-mindmap"])
    assert rc == 0
    out = capsys.readouterr().out
    tree = json.loads(out)
    assert "_ideas" in tree
    assert "branches" in tree


# ------------------------------------------------------------------
# --organize (mocked)
# ------------------------------------------------------------------

def test_cli_organize(isolated_data_dir, monkeypatch, capsys):
    class _FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def generate_json(self, _):
            return {"path": ["testing"]}

    monkeypatch.setattr("notorg.organize.OllamaClient", lambda: _FakeClient())

    main(["--store", "An idea to organize"])
    capsys.readouterr()
    rc = main(["--organize"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Done" in out


# ------------------------------------------------------------------
# --fetch (mocked)
# ------------------------------------------------------------------

def test_cli_fetch(isolated_data_dir, monkeypatch, capsys):
    from notorg import mindmap, storage

    tree = {"_ideas": [], "branches": {}}
    mindmap.insert_idea(tree, ["research"], {
        "id": 1, "text": "A relevant idea", "timestamp": "2024-01-01T00:00:00+00:00"
    })
    storage.save_mindmap(tree)

    class _FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def generate_json(self, prompt):
            if "relevant_paths" in prompt or "branch" in prompt.lower():
                return {"relevant_paths": [["research"]]}
            return {"matching_ids": [1]}

    monkeypatch.setattr("notorg.fetch.OllamaClient", lambda: _FakeClient())
    capsys.readouterr()
    rc = main(["--fetch", "research idea"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "A relevant idea" in out


def test_cli_fetch_no_results(isolated_data_dir, monkeypatch, capsys):
    class _FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def generate_json(self, _):
            return {"relevant_paths": [], "matching_ids": []}

    monkeypatch.setattr("notorg.fetch.OllamaClient", lambda: _FakeClient())
    rc = main(["--fetch", "nothing here"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No matching" in out
