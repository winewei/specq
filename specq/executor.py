"""Execution backend — Claude Code SDK integration."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol

from .models import ExecutionResult, TaskItem, WorkItem


class ExecutionBackend(Protocol):
    """v1 only has ClaudeCodeExecutor; Protocol reserved for future."""

    name: str

    async def execute(
        self,
        work_item: WorkItem,
        task: TaskItem,
        cwd: Path,
        brief: str,
    ) -> ExecutionResult: ...


class ClaudeCodeExecutor:
    """Claude Code SDK native integration."""

    name = "claude_code"

    def __init__(self, model: str, max_turns: int):
        self.model = model
        self.max_turns = max_turns

    async def execute(
        self,
        work_item: WorkItem,
        task: TaskItem,
        cwd: Path,
        brief: str,
    ) -> ExecutionResult:
        try:
            from claude_code_sdk import ClaudeCodeOptions, Message, query
        except ImportError:
            return ExecutionResult(
                success=False,
                output="claude-code-sdk not installed",
            )

        options = ClaudeCodeOptions(
            model=self.model,
            max_turns=self.max_turns,
            cwd=str(cwd),
            system_prompt=(
                f"完成后 commit 你的改动。"
                f"Commit message 格式：feat({work_item.id}): {{描述}}"
            ),
            allowed_tools=[
                "Bash", "Read", "Write", "Edit", "Glob", "Grep",
                "TodoRead", "TodoWrite",
            ],
        )

        output_parts: list[str] = []
        turns = 0
        tokens_in = 0
        tokens_out = 0
        start = time.monotonic()

        try:
            async for message in query(prompt=brief, options=options):
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
            return ExecutionResult(
                success=False,
                output=f"Executor error: {exc}",
                duration_sec=elapsed,
                turns_used=turns,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        elapsed = time.monotonic() - start

        files_changed = await self._get_changed_files(cwd)
        commit_hash = await self._get_latest_commit(cwd)

        return ExecutionResult(
            success=True,
            output="\n".join(output_parts),
            files_changed=files_changed,
            commit_hash=commit_hash,
            duration_sec=elapsed,
            turns_used=turns,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def _get_changed_files(self, cwd: Path) -> list[str]:
        from .git_ops import get_changed_files
        return await get_changed_files(cwd)

    async def _get_latest_commit(self, cwd: Path) -> str:
        from .git_ops import get_latest_commit
        return await get_latest_commit(cwd)
