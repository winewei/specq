"""specq CLI — typer-based command interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

app = typer.Typer(
    name="specq",
    help="specq — Spec-driven AI Agent Orchestrator",
    no_args_is_help=True,
)

# -------------------------------------------------------------------
# Templates
# -------------------------------------------------------------------

DEFAULT_CONFIG_TEMPLATE = """\
# .specq/config.yaml — team-shared configuration
# changes_dir is auto-detected: uses openspec/changes if present, falls back to changes/
# Uncomment to override:
# changes_dir: openspec/changes
base_branch: main

compiler:
  provider: anthropic
  model: claude-haiku-4-5

executor:
  type: claude_code
  model: claude-sonnet-4-5
  max_turns: 50

verification:
  voters:
    - provider: openai
      model: gpt-4o
    - provider: google
      model: gemini-2.5-pro
    - provider: anthropic
      model: claude-sonnet-4-5
  checks:
    - spec_compliance
    - regression_risk
    - architecture

risk_policy:
  low: skip
  medium:
    strategy: majority
  high:
    strategy: unanimous

budgets:
  max_retries: 3
  max_duration_sec: 600
  max_turns: 50
  daily_task_limit: 50

notify:
  webhook_url: ""
  events:
    - change.completed
    - change.failed
    - change.needs_review
    - quota.exceeded
"""

DEFAULT_LOCAL_CONFIG_TEMPLATE = """\
# .specq/local.config.yaml — personal overrides (DO NOT commit)
# providers:
#   anthropic:
#     api_key: sk-ant-xxx
#   openai:
#     api_key: sk-xxx
#   google:
#     api_key: AIza-xxx
"""

EXAMPLE_PROPOSAL = """\
---
depends_on: []
risk: low
---
# Example Change

This is an example change spec. Replace with your actual proposal.

## Goal
Describe what this change achieves.

## Approach
Describe how to implement it.
"""

EXAMPLE_TASKS = """\
# Tasks

## task-1: Example Task
Implement the example feature.
- Step 1
- Step 2
"""

GITIGNORE_ENTRIES = [
    ".specq/local.config.yaml",
    ".specq/state.db",
    ".specq/state.db-wal",
    ".specq/state.db-shm",
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_project_root() -> Path:
    return Path.cwd()


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def _get_db(project_root: Path):
    from .db import Database
    db_path = project_root / ".specq" / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))
    await db.init()
    return db


# -------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------

@app.command()
def init():
    """Initialize specq in the current project."""
    root = _get_project_root()

    # .specq/
    specq_dir = root / ".specq"
    specq_dir.mkdir(exist_ok=True)

    # config.yaml
    config_path = specq_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG_TEMPLATE)
        typer.echo(f"  Created {config_path.relative_to(root)}")
    else:
        typer.echo(f"  Exists  {config_path.relative_to(root)}")

    # local.config.yaml
    local_path = specq_dir / "local.config.yaml"
    if not local_path.exists():
        local_path.write_text(DEFAULT_LOCAL_CONFIG_TEMPLATE)
        typer.echo(f"  Created {local_path.relative_to(root)}")

    # changes/ — skip if openspec/changes already exists
    from .config import detect_changes_dir
    detected = detect_changes_dir(root)
    if detected == "openspec/changes":
        typer.echo(f"  Detected openspec/changes/ — using it as changes source")
    else:
        changes_dir = root / "changes"
        changes_dir.mkdir(exist_ok=True)

        # Example change
        example_dir = changes_dir / "000-example"
        if not example_dir.exists():
            example_dir.mkdir()
            (example_dir / "proposal.md").write_text(EXAMPLE_PROPOSAL)
            (example_dir / "tasks.md").write_text(EXAMPLE_TASKS)
            typer.echo(f"  Created {example_dir.relative_to(root)}/")

    # .gitignore
    gitignore_path = root / ".gitignore"
    existing = ""
    if gitignore_path.exists():
        existing = gitignore_path.read_text()
    additions = [e for e in GITIGNORE_ENTRIES if e not in existing]
    if additions:
        with open(gitignore_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("# specq\n")
            for entry in additions:
                f.write(f"{entry}\n")
        typer.echo(f"  Updated .gitignore")

    typer.echo("\n  specq initialized. Run `specq plan` to preview.")


@app.command()
def plan():
    """Show execution plan (dry-run)."""
    root = _get_project_root()

    from .config import load_config
    from .dag import build_dag, check_cycle, update_blocked_ready
    from .scanner import scan_changes

    config = load_config(root)
    items = scan_changes(root, config)

    if not items:
        typer.echo("  No changes found.")
        return

    graph = build_dag(items)
    try:
        check_cycle(graph)
    except Exception as e:
        typer.echo(f"  DAG Error: {e}", err=True)
        raise typer.Exit(1)

    update_blocked_ready(items)

    # Header
    typer.echo("")
    typer.echo("  specq — Spec-driven Orchestrator")
    typer.echo(f"  Changes: {len(items)} · DAG: valid")
    typer.echo("")

    # Table
    typer.echo(f"  {'#':<3} {'ID':<25} {'Status':<10} {'Deps':<15} {'Risk':<8} {'Verify'}")
    typer.echo(f"  {'---':<3} {'---':<25} {'---':<10} {'---':<15} {'---':<8} {'---'}")

    for i, item in enumerate(items, 1):
        deps_str = ", ".join(item.deps) if item.deps else "—"
        verify = item.verification_strategy
        status = item.status.value
        typer.echo(
            f"  {i:<3} {item.id:<25} {status:<10} {deps_str:<15} {item.risk:<8} {verify}"
        )

    # Executor info
    typer.echo("")
    typer.echo(f"  Executor: claude-code ({config.executor.model})")
    typer.echo(f"  Compiler: {config.compiler.model}")
    voters_str = ", ".join(
        v.get("model", "?") for v in config.verification.voters
    )
    if voters_str:
        typer.echo(f"  Voters: {voters_str}")
    typer.echo("")


@app.command()
def status(
    change_id: str = typer.Argument(None, help="Show details for a specific change"),
):
    """Show status overview."""
    root = _get_project_root()

    async def _status():
        from .config import load_config
        config = load_config(root)
        db = await _get_db(root)
        try:
            items = await db.list_all_work_items()
            if not items:
                # Try scanning
                from .scanner import scan_changes
                scanned = scan_changes(root, config)
                if not scanned:
                    typer.echo("  No changes found.")
                    return
                for wi in scanned:
                    await db.upsert_work_item(wi)
                items = scanned

            if change_id:
                wi = await db.get_work_item(change_id)
                if not wi:
                    typer.echo(f"  Change '{change_id}' not found.")
                    return
                typer.echo(f"\n  {wi.id}")
                typer.echo(f"  Status: {wi.status.value}")
                typer.echo(f"  Risk: {wi.risk}")
                typer.echo(f"  Deps: {', '.join(wi.deps) if wi.deps else '—'}")
                typer.echo(f"  Retries: {wi.retry_count}/{wi.max_retries}")
                tasks = await db.get_tasks(wi.id)
                if tasks:
                    typer.echo(f"\n  Tasks:")
                    for t in tasks:
                        typer.echo(f"    {t.id}: {t.title} [{t.status.value}]")
                return

            typer.echo("\n  specq — Status Overview")
            typer.echo("  " + "─" * 50)

            status_icons = {
                "accepted": "✅", "running": "🔄", "compiling": "🔄",
                "verifying": "🔄", "blocked": "⏳", "ready": "⏳",
                "pending": "⏳", "failed": "❌", "needs_review": "⚠️",
                "rejected": "❌", "skipped": "⏭️", "approved": "✅",
            }

            for wi in items:
                icon = status_icons.get(wi.status.value, "  ")
                typer.echo(f"  {icon} {wi.id:<25} {wi.status.value}")

        finally:
            await db.close()

    _run_async(_status())


@app.command()
def deps():
    """Show DAG dependency graph."""
    root = _get_project_root()

    from .config import load_config
    from .dag import build_dag, topological_order
    from .scanner import scan_changes

    config = load_config(root)
    items = scan_changes(root, config)
    if not items:
        typer.echo("  No changes found.")
        return

    graph = build_dag(items)
    order = topological_order(graph)

    typer.echo("\n  DAG Dependency Graph")
    typer.echo("  " + "─" * 40)
    for node in order:
        deps_set = graph.get(node, set())
        if deps_set:
            typer.echo(f"  {node} ← {', '.join(sorted(deps_set))}")
        else:
            typer.echo(f"  {node} (root)")
    typer.echo("")


@app.command()
def run(
    change_id: str = typer.Argument(None, help="Run a specific change"),
    all_changes: bool = typer.Option(False, "--all", help="Run all changes"),
):
    """Execute ready WorkItems."""
    root = _get_project_root()

    async def _run():
        from .config import load_config
        from .pipeline import run_pipeline

        config = load_config(root)
        db = await _get_db(root)
        try:
            target = change_id if not all_changes else None
            typer.echo("  specq — Starting pipeline...")
            await run_pipeline(config, db, target_id=target)
            typer.echo("  specq — Pipeline complete.")
        finally:
            await db.close()

    _run_async(_run())


@app.command()
def logs(change_id: str = typer.Argument(..., help="Change ID")):
    """Show execution logs."""
    root = _get_project_root()

    async def _logs():
        db = await _get_db(root)
        try:
            entries = await db.get_logs(change_id)
            if not entries:
                typer.echo(f"  No logs for '{change_id}'.")
                return
            typer.echo(f"\n  Logs — {change_id}")
            typer.echo("  " + "─" * 50)
            for entry in entries:
                typer.echo(f"  [{entry['created_at']}] {entry['event']}")
                if entry.get("detail"):
                    import json
                    typer.echo(f"    {json.dumps(entry['detail'], ensure_ascii=False)}")
        finally:
            await db.close()

    _run_async(_logs())


@app.command()
def votes(change_id: str = typer.Argument(..., help="Change ID")):
    """Show voter results."""
    root = _get_project_root()

    async def _votes():
        db = await _get_db(root)
        try:
            wi = await db.get_work_item(change_id)
            if not wi:
                typer.echo(f"  Change '{change_id}' not found.")
                return

            # Show latest attempt
            attempt = wi.retry_count + 1
            results = await db.get_vote_results(change_id, attempt)
            if not results:
                # Try attempt 1
                results = await db.get_vote_results(change_id, 1)
            if not results:
                typer.echo(f"  No vote results for '{change_id}'.")
                return

            typer.echo(f"\n  Verification — {change_id}")
            typer.echo(f"  {'Voter':<25} {'Verdict':<10} {'Conf.':<8} {'Findings'}")
            typer.echo(f"  {'─' * 25} {'─' * 10} {'─' * 8} {'─' * 20}")

            for vr in results:
                verdict_str = "✅ pass" if vr.verdict == "pass" else "❌ fail"
                findings_str = _summarize_findings(vr.findings)
                typer.echo(
                    f"  {vr.voter:<25} {verdict_str:<10} {vr.confidence:<8.2f} {findings_str}"
                )
        finally:
            await db.close()

    _run_async(_votes())


def _summarize_findings(findings: list[dict]) -> str:
    if not findings:
        return "—"
    counts: dict[str, int] = {}
    for f in findings:
        sev = f.get("severity", "info")
        counts[sev] = counts.get(sev, 0) + 1
    parts = [f"{c} {s}" for s, c in counts.items()]
    return ", ".join(parts)


@app.command()
def accept(change_id: str = typer.Argument(..., help="Change ID to accept")):
    """Accept a needs_review change."""
    root = _get_project_root()

    async def _accept():
        from .models import Status
        db = await _get_db(root)
        try:
            wi = await db.get_work_item(change_id)
            if not wi:
                typer.echo(f"  Change '{change_id}' not found.")
                return
            if wi.status != Status.NEEDS_REVIEW:
                typer.echo(f"  Change '{change_id}' is {wi.status.value}, not needs_review.")
                return
            await db.update_status(change_id, Status.ACCEPTED)
            await db.log_event(change_id, "accept", {"manual": True})
            typer.echo(f"  ✅ {change_id} accepted.")
        finally:
            await db.close()

    _run_async(_accept())


@app.command()
def reject(change_id: str = typer.Argument(..., help="Change ID to reject")):
    """Reject a needs_review change → failed."""
    root = _get_project_root()

    async def _reject():
        from .models import Status
        db = await _get_db(root)
        try:
            wi = await db.get_work_item(change_id)
            if not wi:
                typer.echo(f"  Change '{change_id}' not found.")
                return
            await db.update_status(change_id, Status.FAILED)
            await db.log_event(change_id, "reject", {"manual": True})
            typer.echo(f"  ❌ {change_id} rejected → failed.")
        finally:
            await db.close()

    _run_async(_reject())


@app.command()
def retry(change_id: str = typer.Argument(..., help="Change ID to retry")):
    """Retry a failed change."""
    root = _get_project_root()

    async def _retry():
        from .models import Status
        db = await _get_db(root)
        try:
            wi = await db.get_work_item(change_id)
            if not wi:
                typer.echo(f"  Change '{change_id}' not found.")
                return
            if wi.status != Status.FAILED:
                typer.echo(f"  Change '{change_id}' is {wi.status.value}, not failed.")
                return
            await db.update_status(change_id, Status.READY)
            await db.log_event(change_id, "retry", {"manual": True})
            typer.echo(f"  🔄 {change_id} set to ready for retry.")
        finally:
            await db.close()

    _run_async(_retry())


@app.command()
def skip(change_id: str = typer.Argument(..., help="Change ID to skip")):
    """Skip a change and unblock downstream."""
    root = _get_project_root()

    async def _skip():
        from .models import Status
        db = await _get_db(root)
        try:
            wi = await db.get_work_item(change_id)
            if not wi:
                typer.echo(f"  Change '{change_id}' not found.")
                return
            await db.update_status(change_id, Status.SKIPPED)
            await db.log_event(change_id, "skip", {"manual": True})
            typer.echo(f"  ⏭️  {change_id} skipped.")
        finally:
            await db.close()

    _run_async(_skip())


@app.command("config")
def config_show():
    """Show merged configuration with source annotations."""
    root = _get_project_root()

    from .config import load_config
    import yaml

    config = load_config(root)

    # Convert to dict for display
    from dataclasses import asdict
    data = asdict(config)
    # Remove sensitive keys
    if "providers" in data:
        for p in data["providers"].values():
            if isinstance(p, dict) and "api_key" in p and p["api_key"]:
                p["api_key"] = p["api_key"][:8] + "..."

    typer.echo("\n  specq — Merged Configuration")
    typer.echo("  " + "─" * 40)
    typer.echo(yaml.dump(data, default_flow_style=False, allow_unicode=True))


@app.command()
def scan():
    """Manually trigger a scan of changes/ directory."""
    root = _get_project_root()

    from .config import load_config
    from .scanner import scan_changes

    config = load_config(root)
    items = scan_changes(root, config)

    if not items:
        typer.echo("  No changes found.")
        return

    typer.echo(f"\n  Found {len(items)} change(s):")
    for item in items:
        deps_str = f" (deps: {', '.join(item.deps)})" if item.deps else ""
        typer.echo(f"  - {item.id}: {item.title}{deps_str}")
    typer.echo("")


if __name__ == "__main__":
    app()
