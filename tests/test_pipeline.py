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
        with patch("specq.pipeline._create_executor_for_item", return_value=mock_executor):
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
        with patch("specq.pipeline._create_executor_for_item", return_value=mock_executor):
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
        with patch("specq.pipeline._create_executor_for_item", return_value=mock_executor):
            with patch("specq.pipeline.run_voters", side_effect=always_fail_vote):
                from specq.pipeline import run_pipeline
                config = load_config(tmp_project)
                await run_pipeline(config, memory_db)

    wi = await memory_db.get_work_item("001-add-auth")
    assert wi.status == Status.FAILED
    assert wi.retry_count == 3


# ---------------------------------------------------------------------------
# _create_executor_for_item factory
# ---------------------------------------------------------------------------

from specq.pipeline import _create_executor_for_item
from specq.providers import GeminiCLIAgent, CodexAgent, ClaudeCodeAgent
from specq.executor import Executor


def _make_work_item(**kwargs) -> WorkItem:
    defaults = dict(id="test", change_dir="changes/test", title="t",
                    description="d", deps=[], priority=0, risk="low",
                    tasks=[TaskItem(id="task-1", title="t", description="d")])
    defaults.update(kwargs)
    return WorkItem(**defaults)


def test_factory_default_is_claude_code(tmp_project):
    config = load_config(tmp_project)
    wi = _make_work_item()
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor, Executor)
    assert isinstance(executor.agent, ClaudeCodeAgent)


def test_factory_gemini_cli(tmp_project):
    config = load_config(tmp_project)
    wi = _make_work_item(executor_type="gemini_cli", executor_model="gemini-2.5-pro")
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor.agent, GeminiCLIAgent)
    assert executor.agent._cmd == ["gemini", "--experimental-acp", "--model", "gemini-2.5-pro"]


def test_factory_codex(tmp_project):
    config = load_config(tmp_project)
    wi = _make_work_item(executor_type="codex", executor_model="o3")
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor.agent, CodexAgent)
    assert executor.agent._cmd == ["codex", "--mode", "acp", "--model", "o3"]


def test_factory_gemini_cli_falls_back_to_config_model(tmp_project):
    """When no per-item model is set, the global config model is used."""
    config = load_config(tmp_project)
    wi = _make_work_item(executor_type="gemini_cli")
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor.agent, GeminiCLIAgent)
    # Factory falls back to config.executor.model
    assert "--model" in executor.agent._cmd
    assert config.executor.model in executor.agent._cmd


def test_factory_codex_falls_back_to_config_model(tmp_project):
    """When no per-item model is set, the global config model is used."""
    config = load_config(tmp_project)
    wi = _make_work_item(executor_type="codex")
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor.agent, CodexAgent)
    assert "--model" in executor.agent._cmd
    assert config.executor.model in executor.agent._cmd


def test_factory_per_item_overrides_global(tmp_project):
    """executor_type on a WorkItem overrides the global config default."""
    config = load_config(tmp_project)
    # Global config has type=claude_code; item overrides to codex
    wi = _make_work_item(executor_type="codex")
    executor = _create_executor_for_item(config, wi)
    assert isinstance(executor.agent, CodexAgent)
