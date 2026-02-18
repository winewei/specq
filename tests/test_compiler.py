"""Tests for Context Compiler."""

import json
import pytest
from specq.compiler import LLMCompiler
from specq.models import TaskItem, Status


@pytest.mark.asyncio
async def test_compiler_sends_all_context(httpx_mock):
    """Prompt includes proposal + all tasks + project rules."""
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "brief"}]})

    compiler = LLMCompiler("anthropic", "claude-haiku-4-5", "key")
    await compiler.compile(
        proposal="# Add Auth\n实现 JWT 认证",
        all_tasks=["JWT Service", "Middleware"],
        current_task=TaskItem(id="task-1", title="JWT Service", description="实现 JWT"),
        prev_results=[],
        project_rules="使用 PyJWT 库\n遵循 REST 规范",
        retry_findings=None,
    )

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    user_msg = body["messages"][0]["content"]
    assert "Add Auth" in user_msg
    assert "JWT Service" in user_msg
    assert "Middleware" in user_msg
    assert "PyJWT" in user_msg


@pytest.mark.asyncio
async def test_compiler_includes_prev_task_results(httpx_mock):
    """Prompt includes previous task files_changed + commit."""
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "brief"}]})

    prev = TaskItem(id="task-1", title="JWT", description="jwt impl")
    prev.status = Status.ACCEPTED
    prev.files_changed = ["src/auth/jwt.py", "tests/test_jwt.py"]
    prev.commit_hash = "abc123"

    compiler = LLMCompiler("anthropic", "claude-haiku-4-5", "key")
    await compiler.compile(
        proposal="# Auth", all_tasks=["task-1", "task-2"],
        current_task=TaskItem(id="task-2", title="Middleware", description=""),
        prev_results=[prev],
        project_rules="",
        retry_findings=None,
    )

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    user_msg = body["messages"][0]["content"]
    assert "jwt.py" in user_msg


@pytest.mark.asyncio
async def test_compiler_includes_retry_findings(httpx_mock):
    """Retry prompt includes voter findings."""
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "fixed brief"}]})

    compiler = LLMCompiler("anthropic", "claude-haiku-4-5", "key")
    await compiler.compile(
        proposal="# Auth", all_tasks=["task-1"],
        current_task=TaskItem(id="task-1", title="JWT", description=""),
        prev_results=[], project_rules="",
        retry_findings=[
            {"severity": "critical", "category": "spec_compliance",
             "description": "JWT_SECRET 硬编码在源码中"},
            {"severity": "warning", "category": "regression_risk",
             "description": "未处理 token 过期"},
        ],
    )

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    user_msg = body["messages"][0]["content"]
    assert "JWT_SECRET" in user_msg or "硬编码" in user_msg


@pytest.mark.asyncio
async def test_compiler_no_findings_no_fix_section(httpx_mock):
    """First compile has no retry/fix content."""
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "brief"}]})

    compiler = LLMCompiler("anthropic", "claude-haiku-4-5", "key")
    await compiler.compile(
        proposal="# Auth", all_tasks=["t1"],
        current_task=TaskItem(id="t1", title="JWT", description=""),
        prev_results=[], project_rules="", retry_findings=None,
    )

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    user_msg = body["messages"][0]["content"]
    assert "修复" not in user_msg
    assert "finding" not in user_msg.lower()
