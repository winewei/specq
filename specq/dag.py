"""DAG construction, topological sort, and cycle detection."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from .models import Status, WorkItem


class SpecqDAGError(Exception):
    """Raised when DAG validation fails."""


def build_dag(items: list[WorkItem]) -> dict[str, set[str]]:
    """Build dependency graph from work items."""
    graph: dict[str, set[str]] = {}
    for item in items:
        graph[item.id] = set(item.deps)
    return graph


def topological_order(graph: dict[str, set[str]]) -> list[str]:
    """Return topologically sorted node list. Raises on cycle."""
    if not graph:
        return []
    ts = TopologicalSorter(graph)
    try:
        return list(ts.static_order())
    except CycleError as e:
        raise SpecqDAGError(f"Dependency cycle detected: {e}") from e


def check_cycle(graph: dict[str, set[str]]) -> None:
    """Validate DAG: check for cycles and missing dependencies."""
    known = set(graph.keys())
    for node, deps in graph.items():
        missing = deps - known
        if missing:
            raise SpecqDAGError(
                f"Node '{node}' depends on unknown nodes: {missing}"
            )
    # Attempt topological sort — raises on cycle
    topological_order(graph)


def update_blocked_ready(items: list[WorkItem]) -> None:
    """Update status of PENDING/BLOCKED items based on dependency state.

    Does not touch items in other states (RUNNING, VERIFYING, etc.).
    """
    accepted_ids = {i.id for i in items if i.status == Status.ACCEPTED}

    for item in items:
        if item.status not in (Status.PENDING, Status.BLOCKED):
            continue
        if all(dep in accepted_ids for dep in item.deps):
            item.status = Status.READY
        else:
            item.status = Status.BLOCKED
