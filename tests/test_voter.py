"""Tests for multi-model voter."""

import json
import httpx as httpx_lib
import pytest
from specq.voter import ClaudeCodeVoter, LLMVoter, run_voters


# --- JSON vote parsing ---

@pytest.mark.asyncio
async def test_voter_parses_structured_result(httpx_mock):
    """Correctly parse LLM JSON vote response."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": json.dumps({
        "verdict": "fail",
        "confidence": 0.75,
        "findings": [
            {"severity": "critical", "category": "spec_compliance", "description": "缺少 refresh token"},
            {"severity": "warning", "category": "regression_risk", "description": "未处理并发"},
        ],
        "summary": "实现不完整"
    })}}]})

    voter = LLMVoter("openai", "gpt-4o", "key")
    result = await voter.review(
        diff="diff --git a/auth.py...",
        proposal="# Add JWT with refresh token",
        project_rules="必须支持 refresh token",
        checks=["spec_compliance", "regression_risk"],
    )

    assert result.verdict == "fail"
    assert result.confidence == 0.75
    assert len(result.findings) == 2
    assert result.findings[0]["severity"] == "critical"


@pytest.mark.asyncio
async def test_voter_malformed_json_not_pass(httpx_mock):
    """Non-JSON LLM response → not pass (safe default)."""
    httpx_mock.add_response(json={"choices": [{"message": {
        "content": "I think the code looks great! It passes all checks."
    }}]})

    voter = LLMVoter("openai", "gpt-4o", "key")
    result = await voter.review(diff="...", proposal="...", project_rules="", checks=[])
    assert result.verdict in ("fail", "error")


@pytest.mark.asyncio
async def test_voter_partial_json_defaults(httpx_mock):
    """JSON missing optional fields → defaults applied."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": json.dumps({
        "verdict": "pass",
    })}}]})

    voter = LLMVoter("openai", "gpt-4o", "key")
    result = await voter.review(diff="...", proposal="...", project_rules="", checks=[])
    assert result.verdict == "pass"
    assert result.findings is not None


# --- Parallel voting ---

@pytest.mark.asyncio
async def test_voters_parallel_all_collected(httpx_mock):
    """3 voters run in parallel, all results collected."""
    for verdict in ["pass", "fail", "pass"]:
        httpx_mock.add_response(json={"choices": [{"message": {"content": json.dumps({
            "verdict": verdict, "confidence": 0.8, "findings": [], "summary": "ok"
        })}}]})

    voters = [LLMVoter("openai", "gpt-4o", f"k{i}") for i in range(3)]
    results = await run_voters(voters, diff="...", proposal="...", project_rules="", checks=[])
    assert len(results) == 3
    assert sum(1 for r in results if r.verdict == "pass") == 2


@pytest.mark.asyncio
async def test_single_voter_failure_isolated():
    """One voter failure doesn't affect others — tested via mock."""
    from unittest.mock import AsyncMock, patch

    voter1 = LLMVoter("openai", "gpt-4o", "k1")
    voter2 = LLMVoter("openai", "gpt-4o", "k2")

    from specq.models import VoteResult

    pass_result = VoteResult(voter="openai/gpt-4o", verdict="pass",
                             confidence=0.9, findings=[], summary="ok")

    # voter1 raises, voter2 succeeds
    with patch.object(voter1, "review", side_effect=Exception("timeout")):
        with patch.object(voter2, "review", new_callable=AsyncMock, return_value=pass_result):
            results = await run_voters([voter1, voter2], diff="...", proposal="...",
                                       project_rules="", checks=[])

    assert len(results) == 2
    # One error, one pass
    verdicts = {r.verdict for r in results}
    assert "pass" in verdicts
    assert "error" in verdicts


@pytest.mark.asyncio
async def test_voter_passes_checks_to_prompt(httpx_mock):
    """Checks list appears in voter prompt."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": json.dumps({
        "verdict": "pass", "confidence": 0.9, "findings": [], "summary": ""
    })}}]})

    voter = LLMVoter("openai", "gpt-4o", "key")
    await voter.review(
        diff="...", proposal="...", project_rules="",
        checks=["spec_compliance", "regression_risk", "architecture"],
    )

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    user_msg = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "spec_compliance" in user_msg
    assert "architecture" in user_msg


# --- ClaudeCodeVoter ---

@pytest.mark.asyncio
async def test_claude_code_voter_no_api_call(httpx_mock):
    """ClaudeCodeVoter uses local CLI auth, not HTTP to LLM providers."""
    from unittest.mock import AsyncMock, patch

    raw = json.dumps({"verdict": "pass", "confidence": 0.9, "findings": [], "summary": "ok"})
    with patch("specq.voter._claude_code_chat", new=AsyncMock(return_value=raw)):
        voter = ClaudeCodeVoter(model="claude-sonnet-4-6")
        result = await voter.review(diff="...", proposal="...", project_rules="", checks=[])

    assert httpx_mock.get_requests() == []
    assert result.verdict == "pass"
    assert result.voter == "claude_code/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_claude_code_voter_sdk_error_returns_error_verdict():
    """ClaudeCodeVoter SDK failure returns error verdict (does not raise)."""
    from unittest.mock import AsyncMock, patch

    with patch("specq.voter._claude_code_chat", new=AsyncMock(side_effect=RuntimeError("no auth"))):
        voter = ClaudeCodeVoter(model="claude-sonnet-4-6")
        result = await voter.review(diff="...", proposal="...", project_rules="", checks=[])

    assert result.verdict == "error"
    assert "no auth" in result.summary


@pytest.mark.asyncio
async def test_pipeline_creates_claude_code_voter(tmp_project):
    """_create_voters returns ClaudeCodeVoter when provider is claude_code."""
    from specq.config import load_config
    from specq.pipeline import _create_voters

    (tmp_project / ".specq" / "config.yaml").write_text("""\
verification:
  voters:
    - provider: claude_code
      model: claude-sonnet-4-6
""")
    config = load_config(tmp_project)
    voters = _create_voters(config)
    assert len(voters) == 1
    assert isinstance(voters[0], ClaudeCodeVoter)
    assert voters[0].model == "claude-sonnet-4-6"
