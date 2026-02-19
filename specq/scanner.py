"""Scan changes/ directory → WorkItem list."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .config import Config
from .models import Status, TaskItem, WorkItem


# -------------------------------------------------------------------
# Frontmatter parsing
# -------------------------------------------------------------------

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_text).
    """
    match = _FM_RE.match(content)
    if match:
        raw_yaml = match.group(1)
        body = match.group(2)
        meta = yaml.safe_load(raw_yaml)
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        return meta, body
    return {}, content


# -------------------------------------------------------------------
# Tasks parsing
# -------------------------------------------------------------------

_TASK_HEADING_RE = re.compile(r"^##\s+(task-\S+):\s*(.+)$", re.IGNORECASE)


def parse_tasks(content: str) -> list[TaskItem]:
    """Parse tasks.md content → list of TaskItem."""
    tasks: list[TaskItem] = []
    current_id: str | None = None
    current_title: str = ""
    current_lines: list[str] = []

    def _flush():
        nonlocal current_id, current_title, current_lines
        if current_id is not None:
            desc = "\n".join(current_lines).strip()
            tasks.append(TaskItem(id=current_id, title=current_title, description=desc))
        current_id = None
        current_title = ""
        current_lines = []

    for line in content.splitlines():
        m = _TASK_HEADING_RE.match(line)
        if m:
            _flush()
            current_id = m.group(1)
            current_title = m.group(2).strip()
        elif current_id is not None:
            current_lines.append(line)

    _flush()
    return tasks


# -------------------------------------------------------------------
# Directory scanning
# -------------------------------------------------------------------

def _parse_change_dir(
    change_dir: Path, config: Config
) -> WorkItem:
    """Parse a single change directory → WorkItem."""
    proposal_path = change_dir / "proposal.md"
    tasks_path = change_dir / "tasks.md"

    proposal_content = proposal_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(proposal_content)

    tasks: list[TaskItem] = []
    if tasks_path.exists():
        tasks_content = tasks_path.read_text(encoding="utf-8")
        tasks = parse_tasks(tasks_content)

    # Extract title from first markdown heading in body
    title = change_dir.name
    for line in body.splitlines():
        line_s = line.strip()
        if line_s.startswith("# "):
            title = line_s[2:].strip()
            break

    # Build WorkItem with frontmatter overrides
    deps = meta.get("depends_on", [])
    if deps is None:
        deps = []

    wi = WorkItem(
        id=change_dir.name,
        change_dir=str(change_dir.relative_to(Path(config.project_root))),
        title=title,
        description=body.strip(),
        deps=deps,
        priority=meta.get("priority", 0),
        risk=meta.get("risk", "medium"),
        tasks=tasks,
    )

    # Per-change overrides from frontmatter
    if "executor_type" in meta:
        wi.executor_type = meta["executor_type"]
    if "executor_model" in meta:
        wi.executor_model = meta["executor_model"]
    if "max_turns" in meta:
        wi.executor_max_turns = meta["max_turns"]
    if "executor_tools" in meta and isinstance(meta["executor_tools"], list):
        wi.executor_tools = meta["executor_tools"]
    if "verification" in meta and isinstance(meta["verification"], dict):
        wi.verification_strategy = meta["verification"].get(
            "strategy", wi.verification_strategy
        )

    return wi


def scan_changes(project_root: str | Path, config: Config | None = None) -> list[WorkItem]:
    """Scan the changes/ directory and return sorted WorkItems.

    Skips: archive/, dirs without proposal.md, regular files.
    """
    project_root = Path(project_root)

    if config is None:
        from .config import load_config
        config = load_config(project_root)

    changes_dir = project_root / config.changes_dir

    if not changes_dir.exists():
        return []

    items: list[WorkItem] = []
    for entry in sorted(changes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "archive":
            continue
        if not (entry / "proposal.md").exists():
            continue

        wi = _parse_change_dir(entry, config)
        items.append(wi)

    return items
