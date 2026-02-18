"""Scheduler: pick the next ready WorkItem to execute."""

from __future__ import annotations

from .models import Status, WorkItem

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def count_downstream(item: WorkItem, items: list[WorkItem]) -> int:
    """Count how many items (recursively) depend on this item."""
    # Build adjacency: item_id → set of direct dependents
    dependents: dict[str, set[str]] = {}
    for i in items:
        dependents.setdefault(i.id, set())
        for dep in i.deps:
            dependents.setdefault(dep, set()).add(i.id)

    # BFS/DFS to find all transitive dependents
    visited: set[str] = set()
    stack = list(dependents.get(item.id, set()))
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        stack.extend(dependents.get(nid, set()))

    return len(visited)


def pick_next(
    items: list[WorkItem], target_id: str | None = None
) -> WorkItem | None:
    """Select next WorkItem to execute.

    Priority: unlock-count (desc) → priority (desc) → risk (asc, low first).
    """
    if target_id:
        return next(
            (i for i in items if i.id == target_id and i.status == Status.READY),
            None,
        )

    ready = [i for i in items if i.status == Status.READY]
    if not ready:
        return None

    ready.sort(
        key=lambda i: (
            -count_downstream(i, items),
            -i.priority,
            _RISK_ORDER.get(i.risk, 1),
        )
    )
    return ready[0]
