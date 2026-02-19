"""Executor: domain wrapper around a CodeAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .git_ops import get_changed_files, get_latest_commit
from .models import ExecutionResult, TaskItem, WorkItem

_COMMIT_SYSTEM_PROMPT = (
    "完成后 commit 你的改动。"
    "Commit message 格式：feat({work_item_id}): {{描述}}"
)

_DEFAULT_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "TodoRead", "TodoWrite",
]


class Executor:
    """Runs a CodeAgent and collects git results.

    Separates agent execution (providers layer) from git post-processing
    (domain layer): changed files, commit hash.

    *agent* is any object that exposes::

        async def run(prompt, cwd, system_prompt=None) -> AgentRun
    """

    def __init__(self, agent: Any):
        self.agent = agent

    async def execute(
        self,
        work_item: WorkItem,
        task: TaskItem,
        cwd: Path,
        brief: str,
    ) -> ExecutionResult:
        system_prompt = _COMMIT_SYSTEM_PROMPT.format(work_item_id=work_item.id)
        run = await self.agent.run(prompt=brief, cwd=cwd, system_prompt=system_prompt)

        if not run.success:
            return ExecutionResult(
                success=False,
                output=run.output,
                duration_sec=run.duration_sec,
                turns_used=run.turns,
                tokens_in=run.tokens_in,
                tokens_out=run.tokens_out,
            )

        files_changed = await self._get_changed_files(cwd)
        commit_hash = await self._get_latest_commit(cwd)

        return ExecutionResult(
            success=True,
            output=run.output,
            files_changed=files_changed,
            commit_hash=commit_hash,
            duration_sec=run.duration_sec,
            turns_used=run.turns,
            tokens_in=run.tokens_in,
            tokens_out=run.tokens_out,
        )

    async def _get_changed_files(self, cwd: Path) -> list[str]:
        return await get_changed_files(cwd)

    async def _get_latest_commit(self, cwd: Path) -> str:
        return await get_latest_commit(cwd)
