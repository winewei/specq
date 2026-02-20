"""CodeAgent: multi-turn coding agent capability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRun:
    """Raw result from a coding agent run."""

    success: bool
    output: str
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_sec: float = 0.0


class ClaudeCodeAgent:
    """Multi-turn coding agent via Claude Code SDK."""

    name = "claude_code"

    def __init__(
        self,
        model: str,
        max_turns: int,
        allowed_tools: list[str],
        system_prompt: str,
    ):
        self.model = model
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools
        self.system_prompt = system_prompt

    async def run(
        self,
        prompt: str,
        cwd: Path,
        system_prompt: str | None = None,
    ) -> AgentRun:
        """Run the agent.

        Args:
            prompt: The task description for the agent.
            cwd: Working directory for the agent.
            system_prompt: Override the instance-level system_prompt for this run.
                Useful when the prompt contains runtime data (e.g. work item id).
        """
        try:
            from claude_code_sdk import ClaudeCodeOptions, Message, query
        except ImportError:
            return AgentRun(success=False, output="claude-code-sdk not installed")

        options = ClaudeCodeOptions(
            model=self.model,
            max_turns=self.max_turns,
            cwd=str(cwd),
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            allowed_tools=self.allowed_tools,
        )

        output_parts: list[str] = []
        turns = 0
        tokens_in = 0
        tokens_out = 0
        start = time.monotonic()

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, Message):
                    turns += 1
                    for block in message.content:
                        if hasattr(block, "text"):
                            output_parts.append(block.text)
                    if hasattr(message, "usage") and message.usage:
                        tokens_in += getattr(message.usage, "input_tokens", 0)
                        tokens_out += getattr(message.usage, "output_tokens", 0)
        except Exception as exc:
            elapsed = time.monotonic() - start
            return AgentRun(
                success=False,
                output=f"Agent error: {exc}",
                turns=turns,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_sec=elapsed,
            )

        return AgentRun(
            success=True,
            output="\n".join(output_parts),
            turns=turns,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_sec=time.monotonic() - start,
        )
