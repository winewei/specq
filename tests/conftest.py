"""Shared fixtures for specq tests."""

import pytest
import pytest_asyncio
from pathlib import Path


@pytest.fixture
def tmp_project(tmp_path):
    """Create a complete temporary project: .specq/ + changes/ + CLAUDE.md"""
    specq_dir = tmp_path / ".specq"
    specq_dir.mkdir()

    changes_dir = tmp_path / "changes"
    changes_dir.mkdir()

    (tmp_path / "CLAUDE.md").write_text("# Project Rules\nUse PyJWT for auth.\n")

    (specq_dir / "config.yaml").write_text("""\
changes_dir: changes
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
notify:
  webhook_url: ""
  events:
    - change.completed
    - change.failed
""")
    return tmp_path


@pytest.fixture
def sample_change(tmp_project):
    """Single change with frontmatter."""
    change_dir = tmp_project / "changes" / "001-add-auth"
    change_dir.mkdir()
    (change_dir / "proposal.md").write_text("""\
---
depends_on: []
risk: medium
priority: 5
---
# Add JWT Authentication

实现 JWT 认证模块，支持 access 和 refresh token。
""")
    (change_dir / "tasks.md").write_text("""\
# Tasks

## task-1: JWT Service
实现 JWT token 签发与验证。
- create_access_token(user_id, role)
- verify_token(token) → payload

## task-2: Auth Middleware
实现认证中间件，拦截需要认证的路由。
""")
    return change_dir


@pytest.fixture
def multi_changes(tmp_project):
    """4 changes forming a diamond DAG:
    001 → 002 ─┐
    001 → 003 ─┤→ 004
    """
    for cid, deps, risk in [
        ("001-user-model", [], "low"),
        ("002-add-auth", ["001-user-model"], "medium"),
        ("003-add-api", ["001-user-model"], "low"),
        ("004-rate-limit", ["002-add-auth", "003-add-api"], "high"),
    ]:
        d = tmp_project / "changes" / cid
        d.mkdir()
        if deps:
            deps_yaml = "\n  - ".join(deps)
            deps_str = f"\n  - {deps_yaml}"
        else:
            deps_str = " []"
        (d / "proposal.md").write_text(f"""\
---
depends_on:{deps_str}
risk: {risk}
---
# {cid}
""")
        (d / "tasks.md").write_text(f"## task-1: Implement {cid}\nDo the thing.\n")
    return tmp_project


@pytest.fixture
def cycle_changes(tmp_project):
    """3 changes forming a cycle: a → b → c → a"""
    for cid, deps in [
        ("a-feat", ["c-feat"]),
        ("b-feat", ["a-feat"]),
        ("c-feat", ["b-feat"]),
    ]:
        d = tmp_project / "changes" / cid
        d.mkdir()
        deps_yaml = "\n  - ".join(deps)
        (d / "proposal.md").write_text(f"""\
---
depends_on:
  - {deps_yaml}
risk: low
---
# {cid}
""")
        (d / "tasks.md").write_text(f"## task-1: {cid}\nwork\n")
    return tmp_project


@pytest_asyncio.fixture
async def db(tmp_path):
    """Real SQLite file database (WAL mode)."""
    from specq.db import Database
    db = Database(str(tmp_path / "test.db"))
    await db.init()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def memory_db():
    """In-memory database for fast unit tests."""
    from specq.db import Database
    db = Database(":memory:")
    await db.init()
    yield db
    await db.close()
