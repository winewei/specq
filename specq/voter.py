"""Multi-model voter: parallel independent code review."""

from __future__ import annotations

import json

import anyio

from .models import VoteResult
from .providers import LLMProvider
from .compiler import _claude_code_chat

_SYSTEM_PROMPT = """你是代码审查员。对比 git diff 和原始 proposal，判断实现是否符合规范。

以 JSON 格式返回（不要包裹在 markdown 代码块中）：
{
  "verdict": "pass" 或 "fail",
  "confidence": 0.0-1.0,
  "findings": [
    {"severity": "info|warning|critical", "category": "spec_compliance|regression_risk|architecture", "description": "..."}
  ],
  "summary": "一句话总结"
}"""


def _parse_vote_response(raw: str, voter_name: str) -> VoteResult:
    """Parse LLM response into VoteResult. Default to fail on parse error."""
    # Try to extract JSON from response (may be wrapped in markdown)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last ``` lines
        json_lines = []
        inside = False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            elif line.strip() == "```" and inside:
                break
            elif inside:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return VoteResult(
            voter=voter_name,
            verdict="error",
            confidence=0.0,
            findings=[],
            summary=f"Failed to parse voter response as JSON",
        )

    verdict = data.get("verdict", "fail")
    if verdict not in ("pass", "fail"):
        verdict = "fail"

    return VoteResult(
        voter=voter_name,
        verdict=verdict,
        confidence=float(data.get("confidence", 0.0)),
        findings=data.get("findings", []),
        summary=data.get("summary", ""),
    )


class LLMVoter:
    """Single voter using an LLM provider."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.llm = LLMProvider(provider, model, api_key)

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    async def review(
        self,
        diff: str,
        proposal: str,
        project_rules: str,
        checks: list[str],
    ) -> VoteResult:
        parts = []
        parts.append("## Git Diff\n```\n")
        parts.append(diff[:50000])  # Truncate very large diffs
        parts.append("\n```\n\n")
        parts.append(f"## Original Proposal\n{proposal}\n\n")
        if project_rules:
            parts.append(f"## Project Rules\n{project_rules}\n\n")
        if checks:
            parts.append(f"## Required Checks\n")
            for c in checks:
                parts.append(f"- {c}\n")

        user_prompt = "".join(parts)
        raw = await self.llm.chat(_SYSTEM_PROMPT, user_prompt)
        return _parse_vote_response(raw, self.name)


class ClaudeCodeVoter:
    """Voter using local Claude Code CLI auth — no API key required.

    Uses claude_code_sdk.query() with max_turns=1 and no tools.
    Configure via: verification.voters[].provider: claude_code
    """

    def __init__(self, model: str):
        self.model = model

    @property
    def name(self) -> str:
        return f"claude_code/{self.model}"

    async def review(
        self,
        diff: str,
        proposal: str,
        project_rules: str,
        checks: list[str],
    ) -> VoteResult:
        parts = []
        parts.append("## Git Diff\n```\n")
        parts.append(diff[:50000])
        parts.append("\n```\n\n")
        parts.append(f"## Original Proposal\n{proposal}\n\n")
        if project_rules:
            parts.append(f"## Project Rules\n{project_rules}\n\n")
        if checks:
            parts.append("## Required Checks\n")
            for c in checks:
                parts.append(f"- {c}\n")

        user_prompt = "".join(parts)
        try:
            raw = await _claude_code_chat(_SYSTEM_PROMPT, user_prompt, self.model)
        except Exception as exc:
            return VoteResult(
                voter=self.name,
                verdict="error",
                confidence=0.0,
                findings=[],
                summary=f"ClaudeCodeVoter error: {exc}",
            )
        return _parse_vote_response(raw, self.name)


async def run_voters(
    voters: list[LLMVoter],
    diff: str,
    proposal: str,
    project_rules: str,
    checks: list[str],
) -> list[VoteResult]:
    """Run all voters in parallel. Isolate failures."""
    results: list[VoteResult] = []
    lock = anyio.Lock()

    async def _vote(voter: LLMVoter) -> None:
        try:
            result = await voter.review(diff, proposal, project_rules, checks)
        except Exception as exc:
            result = VoteResult(
                voter=voter.name,
                verdict="error",
                confidence=0.0,
                findings=[],
                summary=f"Voter error: {exc}",
            )
        async with lock:
            results.append(result)

    async with anyio.create_task_group() as tg:
        for voter in voters:
            tg.start_soon(_vote, voter)

    return results
