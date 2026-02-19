"""End-to-end integration tests for the pipeline."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from specq.models import Status, WorkItem, TaskItem, VoteResult
from specq.config import load_config


def _mock_exec_result(success=True, files=None):
    """Build mock ExecutionResult."""
    r = MagicMock()
    r.success = success
    r.output = "done" if success else "error"
    r.files_changed = files or []
    r.commit_hash = "abc123" if success else ""
    r.turns_used = 5
    r.tokens_in = 2000
    r.tokens_out = 1000
    r.duration_sec = 30.0
    return r


# --- End-to-end ---

@pytest.mark.asyncio
async def test_single_low_risk_auto_accept(tmp_project, sample_change, memory_db):
    """low risk → skip verification → auto accepted."""
    proposal = (sample_change / "proposal.md").read_text()
    (sample_change / "proposal.md").write_text(proposal.replace("risk: medium", "risk: low"))

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value="brief")
    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=_mock_exec_result())

    with patch("specq.pipeline._create_compiler", return_value=mock_compiler):
        with patch("specq.pipeline._create_executor", return_value=mock_executor):
            from specq.pipeline import run_pipeline
            config = load_config(tmp_project)
            await run_pipeline(config, memory_db)

    wi = await memory_db.get_work_item("001-add-auth")
    assert wi.status == Status.ACCEPTED


@pytest.mark.asyncio
async def test_high_risk_needs_review(tmp_project, memory_db):
    """high risk + all pass → needs_review."""
    c = tmp_project / "changes" / "001-critical"
    c.mkdir()
    (c / "proposal.md").write_text("---\nrisk: high\n---\n# Critical")
    (c / "tasks.md").write_text("## task-1: Do thing\nwork\n")

    async def all_pass(*a, **kw):
        return [
            VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
            VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
        ]

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value="brief")
    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=_mock_exec_result())

    with patch("specq.pipeline._create_compiler", return_value=mock_compiler):
        with patch("specq.pipeline._create_executor", return_value=mock_executor):
            with patch("specq.pipeline.run_voters", side_effect=all_pass):
                from specq.pipeline import run_pipeline
                config = load_config(tmp_project)
                await run_pipeline(config, memory_db)

    wi = await memory_db.get_work_item("001-critical")
    assert wi.status == Status.NEEDS_REVIEW


@pytest.mark.asyncio
async def test_max_retries_then_failed(tmp_project, sample_change, memory_db):
    """Rejected 3 times → failed."""
    async def always_fail_vote(*a, **kw):
        return [VoteResult(voter="a", verdict="fail", confidence=0.9,
                           findings=[{"severity": "critical", "description": "always bad"}],
                           summary="")]

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value="brief")
    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=_mock_exec_result())

    with patch("specq.pipeline._create_compiler", return_value=mock_compiler):
        with patch("specq.pipeline._create_executor", return_value=mock_executor):
            with patch("specq.pipeline.run_voters", side_effect=always_fail_vote):
                from specq.pipeline import run_pipeline
                config = load_config(tmp_project)
                await run_pipeline(config, memory_db)

    wi = await memory_db.get_work_item("001-add-auth")
    assert wi.status == Status.FAILED
    assert wi.retry_count == 3
