"""Context Compiler: refine proposal + context into a focused task brief."""

from __future__ import annotations

from .models import TaskItem
from .providers import LLMProvider

_SYSTEM_PROMPT = """你是一个 Tech Lead，负责给开发者做任务 briefing。
根据提供的 proposal、task 列表和上下文，为当前 task 生成精准的执行指令。

输出格式：

## Task: {task 标题}
{一句话目标}

### 上下文
{前面做了什么，跟当前 task 的关系}

### 要求
{具体实现要求，从 proposal 中提取}

### 约束
{需要注意的规范和限制}

### 接口
{跟哪些模块交互}
"""

_RETRY_SECTION = """
### ⚠️ 上次验收反馈（需修复）
{findings}
"""


class LLMCompiler:
    """Context Compiler using a cheap LLM to refine task briefs."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.llm = LLMProvider(provider, model, api_key)

    async def compile(
        self,
        proposal: str,
        all_tasks: list[str],
        current_task: TaskItem,
        prev_results: list[TaskItem],
        project_rules: str,
        retry_findings: list[dict] | None,
    ) -> str:
        """Return a refined task brief for the executor."""
        parts = []

        parts.append("## Proposal\n")
        parts.append(proposal)
        parts.append("\n\n## All Tasks\n")
        for i, t in enumerate(all_tasks, 1):
            parts.append(f"{i}. {t}\n")

        parts.append(f"\n## Current Task\n")
        parts.append(f"ID: {current_task.id}\n")
        parts.append(f"Title: {current_task.title}\n")
        parts.append(f"Description: {current_task.description}\n")

        if prev_results:
            parts.append("\n## Previous Task Results\n")
            for prev in prev_results:
                files_str = ", ".join(prev.files_changed) if prev.files_changed else "none"
                parts.append(
                    f"- {prev.id} ({prev.title}): files={files_str}, commit={prev.commit_hash}\n"
                )

        if project_rules:
            parts.append(f"\n## Project Rules (CLAUDE.md)\n{project_rules}\n")

        if retry_findings:
            parts.append("\n## ⚠️ 上次验收反馈（需修复）\n")
            for f in retry_findings:
                parts.append(
                    f"- [{f.get('severity', 'info')}] {f.get('category', '')}: {f.get('description', '')}\n"
                )

        user_prompt = "".join(parts)
        return await self.llm.chat(_SYSTEM_PROMPT, user_prompt)
