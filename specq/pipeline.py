"""Core execution pipeline — the main orchestration loop."""

from __future__ import annotations

from pathlib import Path

from .aggregator import aggregate_votes
from .compiler import ClaudeCodeCompiler, LLMCompiler, PassthroughCompiler
from .config import Config, get_verification_strategy
from .dag import build_dag, check_cycle, update_blocked_ready
from .db import Database
from .executor import ClaudeCodeExecutor
from .git_ops import get_change_diff
from .models import Status, VoteResult
from .notifier import Notifier
from .scanner import scan_changes
from .scheduler import pick_next
from .voter import ClaudeCodeVoter, LLMVoter, run_voters


def _read_claude_md(config: Config) -> str:
    """Read CLAUDE.md project rules if it exists."""
    path = Path(config.project_root) / "CLAUDE.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _create_compiler(config: Config):
    provider = config.compiler.provider
    if not provider or provider == "none":
        return PassthroughCompiler()
    if provider == "claude_code":
        return ClaudeCodeCompiler(model=config.compiler.model)
    model = config.compiler.model
    api_key = _get_api_key(config, provider)
    return LLMCompiler(provider, model, api_key)


def _create_voters(config: Config) -> list:
    voters = []
    for v in config.verification.voters:
        provider = v.get("provider", "")
        model = v.get("model", "")
        if not provider or not model:
            continue
        if provider == "claude_code":
            voters.append(ClaudeCodeVoter(model=model))
        else:
            api_key = _get_api_key(config, provider)
            voters.append(LLMVoter(provider, model, api_key))
    return voters


def _get_api_key(config: Config, provider: str) -> str:
    if provider == "anthropic":
        return config.providers.anthropic.api_key
    elif provider == "openai":
        return config.providers.openai.api_key
    elif provider == "google":
        return config.providers.google.api_key
    elif provider == "glm":
        return config.providers.glm.api_key
    elif provider == "deepseek":
        return config.providers.deepseek.api_key
    return ""


async def run_pipeline(
    config: Config,
    db: Database,
    target_id: str | None = None,
) -> None:
    """Core execution loop."""
    compiler = _create_compiler(config)
    executor = ClaudeCodeExecutor(
        model=config.executor.model,
        max_turns=config.executor.max_turns,
    )
    notifier = Notifier(
        webhook_url=config.notify.webhook_url,
        events=config.notify.events,
    )
    project_rules = _read_claude_md(config)

    while True:
        # ① Scan
        work_items = scan_changes(config.project_root, config)

        # Sync to DB
        for wi in work_items:
            existing = await db.get_work_item(wi.id)
            if existing and existing.status in (
                Status.ACCEPTED, Status.FAILED, Status.SKIPPED,
            ):
                wi.status = existing.status
                wi.retry_count = existing.retry_count
            elif existing:
                wi.status = existing.status
                wi.retry_count = existing.retry_count

        # ② DAG + status update
        graph = build_dag(work_items)
        check_cycle(graph)
        update_blocked_ready(work_items)

        # Persist statuses
        for wi in work_items:
            await db.upsert_work_item(wi)
            for task in wi.tasks:
                await db.upsert_task(wi.id, task)

        # ③ Pick next
        next_item = pick_next(work_items, target_id)
        if next_item is None:
            break

        # ④ Execute all tasks in the change serially
        for task in next_item.tasks:
            # Compile
            await db.update_status(next_item.id, Status.COMPILING)
            await db.log_event(next_item.id, "compile", {"task": task.id})

            prev_results = [
                t for t in next_item.tasks
                if t.status == Status.ACCEPTED
            ]

            retry_findings = None
            if next_item.retry_count > 0:
                votes = await db.get_vote_results(
                    next_item.id, next_item.retry_count
                )
                retry_findings = [f for v in votes for f in v.findings]

            brief = await compiler.compile(
                proposal=next_item.description,
                all_tasks=[t.title for t in next_item.tasks],
                current_task=task,
                prev_results=prev_results,
                project_rules=project_rules,
                retry_findings=retry_findings,
            )
            next_item.compiled_brief = brief
            await db.update_compiled_brief(next_item.id, brief)

            # Execute
            await db.update_status(next_item.id, Status.RUNNING)
            await db.log_event(next_item.id, "execute", {"task": task.id})

            result = await executor.execute(
                work_item=next_item,
                task=task,
                cwd=Path(config.project_root),
                brief=brief,
            )

            # Update task results
            task.files_changed = result.files_changed
            task.commit_hash = result.commit_hash
            task.execution_output = result.output
            task.turns_used = result.turns_used
            task.tokens_in = result.tokens_in
            task.tokens_out = result.tokens_out
            task.duration_sec = result.duration_sec
            task.status = Status.ACCEPTED if result.success else Status.FAILED
            await db.upsert_task(next_item.id, task)

            if not result.success:
                break

        # ⑤ Verification (vote on the whole change diff)
        strategy = get_verification_strategy(next_item, config)

        if strategy != "skip":
            await db.update_status(next_item.id, Status.VERIFYING)
            diff = await get_change_diff(
                Path(config.project_root), config.base_branch
            )

            voters = _create_voters(config)
            vote_results = await run_voters(
                voters=voters,
                diff=diff,
                proposal=next_item.description,
                project_rules=project_rules,
                checks=config.verification.checks,
            )
            await db.save_vote_results(
                next_item.id,
                attempt=next_item.retry_count + 1,
                results=vote_results,
            )
            await db.log_event(next_item.id, "vote", {
                "attempt": next_item.retry_count + 1,
                "results": [
                    {"voter": v.voter, "verdict": v.verdict}
                    for v in vote_results
                ],
            })

            decision, findings = aggregate_votes(
                vote_results, strategy, next_item.risk
            )
        else:
            decision = "approved"
            findings = []

        # ⑥ Handle result
        if decision == "approved":
            await db.update_status(next_item.id, Status.ACCEPTED)
            await db.log_event(next_item.id, "approve", {})
            await notifier.notify("change.completed", next_item)
        elif decision == "needs_review":
            await db.update_status(next_item.id, Status.NEEDS_REVIEW)
            await db.log_event(next_item.id, "needs_review", {})
            await notifier.notify("change.needs_review", next_item)
            # In non-interactive mode, stop here for this item
            if target_id:
                break
        elif decision == "rejected":
            if next_item.retry_count < next_item.max_retries:
                next_item.retry_count += 1
                await db.update_retry_count(next_item.id, next_item.retry_count)
                await db.update_status(next_item.id, Status.READY)
                await db.log_event(next_item.id, "retry", {
                    "attempt": next_item.retry_count,
                    "findings": findings,
                })
                # Continue loop — will pick up the same item again
                continue
            else:
                await db.update_status(next_item.id, Status.FAILED)
                await db.log_event(next_item.id, "failed", {"reason": "max_retries_exceeded"})
                await notifier.notify("change.failed", next_item)

        # ⑦ Re-scan: loop back to top to discover new changes
        if target_id:
            break  # Only run the target
