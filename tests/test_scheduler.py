"""Tests for scheduler: pick_next and downstream counting."""

import pytest
from specq.scheduler import pick_next, count_downstream
from specq.models import Status, WorkItem


def test_pick_single_ready():
    """Single ready item → return it."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", status=Status.READY),
        WorkItem(id="002", change_dir="c/002", title="", description="", status=Status.BLOCKED),
    ]
    assert pick_next(items).id == "001"


def test_pick_none_all_blocked():
    """All blocked → None."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", status=Status.BLOCKED),
    ]
    assert pick_next(items) is None


def test_pick_none_empty():
    """Empty list → None."""
    assert pick_next([]) is None


def test_pick_by_unlock_count():
    """Prefer item that unlocks more downstream."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="",
                 status=Status.READY, deps=[]),
        WorkItem(id="002", change_dir="c/002", title="", description="",
                 status=Status.BLOCKED, deps=["001"]),
        WorkItem(id="003", change_dir="c/003", title="", description="",
                 status=Status.BLOCKED, deps=["001"]),
        WorkItem(id="004", change_dir="c/004", title="", description="",
                 status=Status.BLOCKED, deps=["001"]),
        WorkItem(id="005", change_dir="c/005", title="", description="",
                 status=Status.READY, deps=[]),
        WorkItem(id="006", change_dir="c/006", title="", description="",
                 status=Status.BLOCKED, deps=["005"]),
    ]
    result = pick_next(items)
    assert result.id == "001"  # Unlocks 3 > unlocks 1


def test_pick_by_priority_when_unlock_equal():
    """Equal unlock count → higher priority wins."""
    items = [
        WorkItem(id="A", change_dir="c/A", title="", description="",
                 status=Status.READY, priority=5),
        WorkItem(id="B", change_dir="c/B", title="", description="",
                 status=Status.READY, priority=10),
    ]
    assert pick_next(items).id == "B"


def test_pick_low_risk_first():
    """Equal otherwise → low risk preferred."""
    items = [
        WorkItem(id="A", change_dir="c/A", title="", description="",
                 status=Status.READY, risk="high"),
        WorkItem(id="B", change_dir="c/B", title="", description="",
                 status=Status.READY, risk="low"),
    ]
    assert pick_next(items).id == "B"


def test_pick_target_id():
    """target_id → pick only that item."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", status=Status.READY),
        WorkItem(id="002", change_dir="c/002", title="", description="", status=Status.READY),
    ]
    assert pick_next(items, target_id="002").id == "002"


def test_pick_target_id_not_ready():
    """target_id not ready → None."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", status=Status.BLOCKED),
    ]
    assert pick_next(items, target_id="001") is None


# --- Downstream counting ---

def test_count_downstream_diamond():
    """Diamond DAG: 001's downstream = 3 (transitive)."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", deps=[]),
        WorkItem(id="002", change_dir="c/002", title="", description="", deps=["001"]),
        WorkItem(id="003", change_dir="c/003", title="", description="", deps=["001"]),
        WorkItem(id="004", change_dir="c/004", title="", description="", deps=["002", "003"]),
    ]
    assert count_downstream(items[0], items) == 3


def test_count_downstream_leaf_is_zero():
    """Leaf node downstream = 0."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", deps=[]),
        WorkItem(id="002", change_dir="c/002", title="", description="", deps=["001"]),
    ]
    assert count_downstream(items[1], items) == 0
