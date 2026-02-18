"""Vote aggregation: skip / majority / unanimous."""

from __future__ import annotations

from .models import VoteResult


def aggregate_votes(
    results: list[VoteResult],
    strategy: str,
    risk: str,
) -> tuple[str, list[dict]]:
    """Aggregate vote results.

    Returns (decision, all_findings).
    decision: "approved" | "rejected" | "needs_review"
    """
    if strategy == "skip":
        return "approved", []

    all_findings = [f for r in results for f in r.findings]

    # No voters → rejected
    pass_count = sum(1 for r in results if r.verdict == "pass")
    total = len(results)

    if total == 0:
        return "rejected", all_findings

    has_critical = any(
        f.get("severity") == "critical" for f in all_findings
    )

    if strategy == "majority":
        passed = pass_count > total / 2
    elif strategy == "unanimous":
        passed = pass_count == total
    else:
        passed = pass_count > total / 2  # default to majority

    if not passed:
        return "rejected", all_findings

    # Passed but risk escalation
    if has_critical or risk == "high":
        return "needs_review", all_findings

    return "approved", all_findings
