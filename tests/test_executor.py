"""Tests for Executor — domain wrapper around ClaudeCodeAgent."""

import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from specq.executor import Executor, _DEFAULT_TOOLS
from specq.providers import ClaudeCodeAgent
from specq.models import WorkItem, TaskItem


class FakeMessage:
    """Fake SDK message for testing."""
    def __init__(self, text="", input_tokens=0, output_tokens=0):
        self.content = [MagicMock(text=text)]
        self.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)


def _make_mock_sdk(messages=None, raise_exc=None):
    """Create a mock claude_code_sdk module."""
    mock_sdk = types.ModuleType("claude_code_sdk")

    class MockOptions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_sdk.ClaudeCodeOptions = MockOptions
    mock_sdk.Message = FakeMessage

    async def mock_query(prompt, options):
        if raise_exc:
            raise raise_exc
        if messages:
            for msg in messages:
                yield msg

    mock_sdk.query = mock_query
    return mock_sdk


def _make_executor(model="claude-sonnet-4-5", max_turns=50) -> Executor:
    agent = ClaudeCodeAgent(
        model=model,
        max_turns=max_turns,
        allowed_tools=_DEFAULT_TOOLS,
        system_prompt="",  # overridden per-run by Executor.execute()
    )
    return Executor(agent=agent)


@pytest.mark.asyncio
async def test_executor_calls_sdk_with_correct_params():
    """SDK called with correct model, max_turns, cwd, system_prompt."""
    captured = {}
    mock_sdk = _make_mock_sdk()
    original_query = mock_sdk.query

    async def capturing_query(prompt, options):
        captured["model"] = options.model
        captured["max_turns"] = options.max_turns
        captured["cwd"] = options.cwd
        captured["system_prompt"] = options.system_prompt
        async for msg in original_query(prompt, options):
            yield msg

    mock_sdk.query = capturing_query

    with patch.dict(sys.modules, {"claude_code_sdk": mock_sdk}):
        executor = _make_executor(model="claude-sonnet-4-5", max_turns=50)
        wi = WorkItem(id="001", change_dir="c/001", title="", description="")
        task = TaskItem(id="task-1", title="JWT", description="")
        with patch.object(executor, "_get_changed_files", new_callable=AsyncMock, return_value=[]):
            with patch.object(executor, "_get_latest_commit", new_callable=AsyncMock, return_value=""):
                result = await executor.execute(wi, task, Path("/project"), "impl jwt")

    assert result.success is True
    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["max_turns"] == 50


@pytest.mark.asyncio
async def test_executor_collects_token_usage():
    """Token usage accumulated from SDK messages."""
    messages = [
        FakeMessage("reading...", 1000, 500),
        FakeMessage("writing...", 2000, 1500),
        FakeMessage("testing...", 1500, 800),
    ]
    mock_sdk = _make_mock_sdk(messages=messages)

    with patch.dict(sys.modules, {"claude_code_sdk": mock_sdk}):
        executor = _make_executor()
        wi = WorkItem(id="001", change_dir="c/001", title="", description="")
        task = TaskItem(id="task-1", title="JWT", description="")
        with patch.object(executor, "_get_changed_files", new_callable=AsyncMock, return_value=[]):
            with patch.object(executor, "_get_latest_commit", new_callable=AsyncMock, return_value=""):
                result = await executor.execute(wi, task, Path("/tmp"), "brief")

    assert result.turns_used == 3
    assert result.tokens_in == 4500
    assert result.tokens_out == 2800


@pytest.mark.asyncio
async def test_executor_captures_git_state():
    """Captures files_changed + commit_hash."""
    mock_sdk = _make_mock_sdk()

    with patch.dict(sys.modules, {"claude_code_sdk": mock_sdk}):
        executor = _make_executor()
        wi = WorkItem(id="001", change_dir="c/001", title="", description="")
        task = TaskItem(id="task-1", title="JWT", description="")
        with patch.object(executor, "_get_changed_files",
                          new_callable=AsyncMock,
                          return_value=["src/auth.py", "tests/test_auth.py"]):
            with patch.object(executor, "_get_latest_commit",
                              new_callable=AsyncMock, return_value="deadbeef"):
                result = await executor.execute(wi, task, Path("/tmp"), "brief")

    assert result.files_changed == ["src/auth.py", "tests/test_auth.py"]
    assert result.commit_hash == "deadbeef"


@pytest.mark.asyncio
async def test_executor_sdk_error_returns_failure():
    """SDK exception → success=False."""
    mock_sdk = _make_mock_sdk(raise_exc=RuntimeError("Claude Code crashed"))

    with patch.dict(sys.modules, {"claude_code_sdk": mock_sdk}):
        executor = _make_executor()
        wi = WorkItem(id="001", change_dir="c/001", title="", description="")
        task = TaskItem(id="task-1", title="JWT", description="")
        result = await executor.execute(wi, task, Path("/tmp"), "brief")

    assert result.success is False


@pytest.mark.asyncio
async def test_executor_system_prompt_has_commit_format():
    """System prompt includes commit format + change ID."""
    captured = {}
    mock_sdk = _make_mock_sdk()
    original_query = mock_sdk.query

    async def capturing_query(prompt, options):
        captured["system_prompt"] = options.system_prompt
        async for msg in original_query(prompt, options):
            yield msg

    mock_sdk.query = capturing_query

    with patch.dict(sys.modules, {"claude_code_sdk": mock_sdk}):
        executor = _make_executor()
        wi = WorkItem(id="001-auth", change_dir="c/001", title="", description="")
        task = TaskItem(id="task-1", title="JWT", description="")
        with patch.object(executor, "_get_changed_files", new_callable=AsyncMock, return_value=[]):
            with patch.object(executor, "_get_latest_commit", new_callable=AsyncMock, return_value=""):
                await executor.execute(wi, task, Path("/tmp"), "brief")

    assert "001-auth" in captured["system_prompt"]
    assert "commit" in captured["system_prompt"].lower()
