"""Tests for DAG topological sort, cycle detection, and status updates."""

import pytest
from specq.dag import build_dag, topological_order, check_cycle, update_blocked_ready, SpecqDAGError
from specq.scanner import scan_changes
from specq.config import load_config
from specq.models import Status, WorkItem


# --- Topological sort ---

def test_linear_chain():
    """Linear dependency: A → B → C."""
    graph = {"A": set(), "B": {"A"}, "C": {"B"}}
    order = topological_order(graph)
    assert order.index("A") < order.index("B") < order.index("C")


def test_diamond_dag(multi_changes):
    """Diamond DAG: 001 → {002,003} → 004."""
    config = load_config(multi_changes)
    items = scan_changes(multi_changes, config)
    graph = build_dag(items)
    order = topological_order(graph)
    idx = {name: i for i, name in enumerate(order)}
    assert idx["001-user-model"] < idx["002-add-auth"]
    assert idx["001-user-model"] < idx["003-add-api"]
    assert idx["002-add-auth"] < idx["004-rate-limit"]
    assert idx["003-add-api"] < idx["004-rate-limit"]


def test_independent_items_all_in_order():
    """Independent items all appear in the order."""
    graph = {"X": set(), "Y": set(), "Z": set()}
    order = topological_order(graph)
    assert set(order) == {"X", "Y", "Z"}


def test_empty_graph():
    """Empty graph → empty list."""
    assert topological_order({}) == []


# --- Cycle detection ---

def test_cycle_three_nodes():
    """A → B → C → A cycle raises error."""
    graph = {"A": {"C"}, "B": {"A"}, "C": {"B"}}
    with pytest.raises(SpecqDAGError, match="[Cc]ycle"):
        check_cycle(graph)


def test_self_cycle():
    """Self-reference A → A."""
    graph = {"A": {"A"}}
    with pytest.raises(SpecqDAGError, match="[Cc]ycle"):
        check_cycle(graph)


def test_no_cycle_passes():
    """Acyclic graph passes without error."""
    graph = {"A": set(), "B": {"A"}, "C": {"A"}}
    check_cycle(graph)


def test_missing_dep_detected():
    """Reference to non-existent dependency raises error."""
    graph = {"B": {"nonexistent"}}
    with pytest.raises(SpecqDAGError):
        check_cycle(graph)


# --- blocked/ready status ---

def test_no_deps_becomes_ready():
    """No dependencies → ready."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="", deps=[]),
        WorkItem(id="002", change_dir="c/002", title="", description="", deps=["001"]),
    ]
    update_blocked_ready(items)
    assert items[0].status == Status.READY
    assert items[1].status == Status.BLOCKED


def test_dep_accepted_unblocks_downstream():
    """Dependency accepted → downstream becomes ready."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="",
                 deps=[], status=Status.ACCEPTED),
        WorkItem(id="002", change_dir="c/002", title="", description="", deps=["001"]),
    ]
    update_blocked_ready(items)
    assert items[1].status == Status.READY


def test_partial_deps_stays_blocked():
    """Partial deps satisfied → still blocked."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="",
                 deps=[], status=Status.ACCEPTED),
        WorkItem(id="002", change_dir="c/002", title="", description="",
                 deps=[], status=Status.RUNNING),
        WorkItem(id="003", change_dir="c/003", title="", description="",
                 deps=["001", "002"]),
    ]
    update_blocked_ready(items)
    assert items[2].status == Status.BLOCKED


def test_running_items_not_touched():
    """Items in RUNNING/VERIFYING state are not reset."""
    items = [
        WorkItem(id="001", change_dir="c/001", title="", description="",
                 deps=[], status=Status.RUNNING),
    ]
    update_blocked_ready(items)
    assert items[0].status == Status.RUNNING
