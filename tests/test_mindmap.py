"""Tests for the mind-map data structure."""

import pytest

from notorg import mindmap


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def empty_tree():
    return {"_ideas": [], "branches": {}}


@pytest.fixture()
def populated_tree():
    tree = {"_ideas": [], "branches": {}}
    mindmap.insert_idea(tree, ["research", "conformal_prediction", "llm"], {
        "id": 1, "text": "Use CP on LLM graphs", "timestamp": "2024-01-01T00:00:00+00:00"
    })
    mindmap.insert_idea(tree, ["research", "conformal_prediction"], {
        "id": 2, "text": "CP theory overview", "timestamp": "2024-01-02T00:00:00+00:00"
    })
    mindmap.insert_idea(tree, ["engineering", "web"], {
        "id": 3, "text": "Build a dashboard", "timestamp": "2024-01-03T00:00:00+00:00"
    })
    mindmap.insert_idea(tree, [], {
        "id": 4, "text": "Misc root idea", "timestamp": "2024-01-04T00:00:00+00:00"
    })
    return tree


# ------------------------------------------------------------------
# ensure_node / get_node
# ------------------------------------------------------------------

def test_ensure_node_creates_missing(empty_tree):
    node = mindmap.ensure_node(empty_tree, ["a", "b", "c"])
    assert node == {"_ideas": [], "branches": {}}
    # Check path was created
    assert "a" in empty_tree["branches"]
    assert "b" in empty_tree["branches"]["a"]["branches"]


def test_get_node_returns_none_for_missing(empty_tree):
    assert mindmap.get_node(empty_tree, ["nonexistent"]) is None


def test_get_node_returns_root_for_empty_path(empty_tree):
    assert mindmap.get_node(empty_tree, []) is empty_tree


# ------------------------------------------------------------------
# insert_idea
# ------------------------------------------------------------------

def test_insert_idea_at_root(empty_tree):
    idea = {"id": 1, "text": "hello", "timestamp": "t"}
    mindmap.insert_idea(empty_tree, [], idea)
    assert empty_tree["_ideas"] == [idea]


def test_insert_idea_deep_path(empty_tree):
    idea = {"id": 1, "text": "deep idea", "timestamp": "t"}
    mindmap.insert_idea(empty_tree, ["a", "b", "c"], idea)
    node = mindmap.get_node(empty_tree, ["a", "b", "c"])
    assert idea in node["_ideas"]


def test_insert_idea_no_duplicate(empty_tree):
    idea = {"id": 42, "text": "once", "timestamp": "t"}
    mindmap.insert_idea(empty_tree, ["x"], idea)
    mindmap.insert_idea(empty_tree, ["x"], idea)  # same id
    node = mindmap.get_node(empty_tree, ["x"])
    assert len(node["_ideas"]) == 1


# ------------------------------------------------------------------
# all_paths
# ------------------------------------------------------------------

def test_all_paths_empty(empty_tree):
    paths = mindmap.all_paths(empty_tree)
    assert [] in paths  # root is always included


def test_all_paths_populated(populated_tree):
    paths = mindmap.all_paths(populated_tree)
    assert ["research"] in paths
    assert ["research", "conformal_prediction"] in paths
    assert ["research", "conformal_prediction", "llm"] in paths
    assert ["engineering", "web"] in paths


# ------------------------------------------------------------------
# all_ideas
# ------------------------------------------------------------------

def test_all_ideas(populated_tree):
    ideas = list(mindmap.all_ideas(populated_tree))
    idea_ids = [i.get("id") for _, i in ideas]
    assert set(idea_ids) == {1, 2, 3, 4}


# ------------------------------------------------------------------
# collect_ideas_under
# ------------------------------------------------------------------

def test_collect_ideas_under_subtree(populated_tree):
    ideas = mindmap.collect_ideas_under(populated_tree, ["research"])
    ids = {i["id"] for i in ideas}
    # Should include ideas in research, research->cp, research->cp->llm
    assert 1 in ids
    assert 2 in ids
    assert 3 not in ids  # engineering subtree
    assert 4 not in ids  # root level


def test_collect_ideas_under_missing_path(populated_tree):
    assert mindmap.collect_ideas_under(populated_tree, ["nonexistent"]) == []


# ------------------------------------------------------------------
# branch_summary
# ------------------------------------------------------------------

def test_branch_summary_empty(empty_tree):
    summary = mindmap.branch_summary(empty_tree)
    assert "(root)" in summary


def test_branch_summary_populated(populated_tree):
    summary = mindmap.branch_summary(populated_tree)
    assert "research" in summary
    assert "conformal_prediction" in summary
    assert "engineering" in summary
