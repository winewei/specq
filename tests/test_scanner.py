"""Tests for scanner: directory scanning + frontmatter/tasks parsing."""

import pytest
from specq.scanner import scan_changes, parse_frontmatter, parse_tasks
from specq.config import load_config
from specq.models import Status


# --- Directory scanning ---

def test_scan_single_change(tmp_project, sample_change):
    """Scan single change → correct ID, risk, priority, tasks."""
    config = load_config(tmp_project)
    items = scan_changes(tmp_project, config)
    assert len(items) == 1
    wi = items[0]
    assert wi.id == "001-add-auth"
    assert wi.risk == "medium"
    assert wi.priority == 5
    assert len(wi.tasks) == 2
    assert wi.tasks[0].id == "task-1"
    assert wi.tasks[0].title == "JWT Service"


def test_scan_multiple_sorted(multi_changes):
    """Multiple changes sorted by directory name."""
    config = load_config(multi_changes)
    items = scan_changes(multi_changes, config)
    assert len(items) == 4
    assert [i.id for i in items] == [
        "001-user-model", "002-add-auth", "003-add-api", "004-rate-limit"
    ]


def test_scan_ignores_archive(tmp_project):
    """archive/ directory is skipped."""
    config = load_config(tmp_project)
    archive = tmp_project / "changes" / "archive" / "done-change"
    archive.mkdir(parents=True)
    (archive / "proposal.md").write_text("# Done")
    (archive / "tasks.md").write_text("## task-1: done\n")
    items = scan_changes(tmp_project, config)
    assert len(items) == 0


def test_scan_ignores_no_proposal(tmp_project):
    """Directories without proposal.md are skipped."""
    config = load_config(tmp_project)
    (tmp_project / "changes" / "bad-dir").mkdir()
    (tmp_project / "changes" / "bad-dir" / "tasks.md").write_text("## task-1: x\n")
    items = scan_changes(tmp_project, config)
    assert len(items) == 0


def test_scan_ignores_files_not_dirs(tmp_project):
    """Regular files in changes/ are skipped."""
    config = load_config(tmp_project)
    (tmp_project / "changes" / "README.md").write_text("# Changes")
    items = scan_changes(tmp_project, config)
    assert len(items) == 0


# --- Frontmatter parsing ---

def test_parse_frontmatter_complete():
    """Complete frontmatter with deps, risk, priority, executor config."""
    content = """\
---
depends_on:
  - 001-user-model
  - 002-add-auth
risk: high
priority: 10
executor_model: claude-sonnet-4-5
max_turns: 80
verification:
  strategy: unanimous
---
# Rate Limiting
Body text here.
"""
    meta, body = parse_frontmatter(content)
    assert meta["depends_on"] == ["001-user-model", "002-add-auth"]
    assert meta["risk"] == "high"
    assert meta["priority"] == 10
    assert meta["executor_model"] == "claude-sonnet-4-5"
    assert meta["verification"]["strategy"] == "unanimous"
    assert "# Rate Limiting" in body
    assert "---" not in body


def test_parse_frontmatter_empty():
    """No frontmatter → empty meta + full body."""
    content = "# Just a title\n\nBody."
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert "# Just a title" in body


def test_parse_frontmatter_empty_yaml():
    """Frontmatter exists but empty YAML."""
    content = "---\n---\n# Title"
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert "# Title" in body


# --- Tasks parsing ---

def test_parse_tasks_standard():
    """Standard tasks.md with multiple tasks + multi-line descriptions."""
    content = """\
# Tasks

## task-1: JWT Service
实现 JWT token 签发。
- create_access_token()
- verify_token()

## task-2: Auth Middleware
实现认证中间件。
拦截需要认证的路由。

## task-3: Login API
实现登录端点。
"""
    tasks = parse_tasks(content)
    assert len(tasks) == 3
    assert tasks[0].id == "task-1"
    assert tasks[0].title == "JWT Service"
    assert "create_access_token" in tasks[0].description
    assert tasks[1].id == "task-2"
    assert "拦截" in tasks[1].description


def test_parse_tasks_preserves_order():
    """Task order matches file order (not sorted by ID)."""
    content = "## task-3: C\ndesc\n## task-1: A\ndesc\n## task-2: B\ndesc\n"
    tasks = parse_tasks(content)
    assert [t.id for t in tasks] == ["task-3", "task-1", "task-2"]


def test_parse_tasks_empty():
    """Empty tasks.md → empty list."""
    tasks = parse_tasks("# Tasks\n")
    assert tasks == []


# --- OpenSpec auto-detection ---

def test_scan_auto_detects_openspec_changes(tmp_path):
    """scan_changes auto-detects openspec/changes when no changes_dir configured."""
    from specq.config import load_config

    (tmp_path / ".specq").mkdir()
    change_dir = tmp_path / "openspec" / "changes" / "my-feature"
    change_dir.mkdir(parents=True)
    (change_dir / "proposal.md").write_text("""\
---
risk: low
---
# My Feature
Add a new feature.
""")
    (change_dir / "tasks.md").write_text("## task-1: Implement\nDo it.\n")

    config = load_config(tmp_path)
    assert config.changes_dir == "openspec/changes"

    items = scan_changes(tmp_path, config)
    assert len(items) == 1
    assert items[0].id == "my-feature"
    assert items[0].risk == "low"


def test_scan_openspec_ignores_archive(tmp_path):
    """archive/ inside openspec/changes is skipped."""
    from specq.config import load_config

    (tmp_path / ".specq").mkdir()
    changes_root = tmp_path / "openspec" / "changes"
    changes_root.mkdir(parents=True)

    active = changes_root / "active-change"
    active.mkdir()
    (active / "proposal.md").write_text("# Active\n")

    archived = changes_root / "archive" / "old-change"
    archived.mkdir(parents=True)
    (archived / "proposal.md").write_text("# Old\n")

    config = load_config(tmp_path)
    items = scan_changes(tmp_path, config)
    assert len(items) == 1
    assert items[0].id == "active-change"
