"""Tests for data models and Status enum."""

import pytest
from specq.models import Status, WorkItem, TaskItem, VoteResult, ExecutionResult


# --- Status enum completeness ---

def test_status_enum_completeness():
    """11 statuses exist with correct values."""
    expected = [
        "pending", "blocked", "ready", "compiling", "running",
        "verifying", "needs_review", "accepted",
        "rejected", "failed", "skipped",
    ]
    assert [s.value for s in Status] == expected
    assert len(Status) == 11


def test_status_is_str_enum():
    """Status value can be used as string."""
    assert Status.PENDING == "pending"
    assert Status.RUNNING.value == "running"


# --- WorkItem defaults ---

def test_work_item_mutable_defaults_isolated():
    """Multiple WorkItems have independent list fields."""
    wi1 = WorkItem(id="001", change_dir="c/001", title="a", description="")
    wi2 = WorkItem(id="002", change_dir="c/002", title="b", description="")
    wi1.deps.append("xxx")
    wi1.tasks.append(TaskItem(id="t1", title="t", description=""))
    assert wi2.deps == []
    assert wi2.tasks == []


def test_work_item_risk_default():
    """Default risk = medium."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    assert wi.risk == "medium"


def test_work_item_verification_default():
    """Default verification_strategy = "" (resolved via risk_policy at runtime)."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    assert wi.verification_strategy == ""


# --- TaskItem metrics ---

def test_task_metrics_zeroed():
    """TaskItem metrics initialized to zero."""
    t = TaskItem(id="task-1", title="x", description="")
    assert t.turns_used == 0
    assert t.tokens_in == 0
    assert t.tokens_out == 0
    assert t.duration_sec == 0.0
    assert t.commit_hash == ""
    assert t.files_changed == []


# --- VoteResult ---

def test_vote_result_with_multiple_findings():
    """VoteResult supports multiple findings."""
    vr = VoteResult(
        voter="openai/gpt-4o",
        verdict="fail",
        confidence=0.7,
        findings=[
            {"severity": "critical", "category": "spec_compliance", "description": "缺少刷新 token"},
            {"severity": "warning", "category": "regression_risk", "description": "未处理并发"},
            {"severity": "info", "category": "architecture", "description": "建议拆分模块"},
        ],
        summary="实现不完整",
    )
    assert len(vr.findings) == 3
    assert vr.findings[0]["severity"] == "critical"


# --- ExecutionResult ---

def test_execution_result_defaults():
    """ExecutionResult has sensible defaults."""
    er = ExecutionResult(success=True, output="done")
    assert er.files_changed == []
    assert er.commit_hash == ""
    assert er.turns_used == 0
