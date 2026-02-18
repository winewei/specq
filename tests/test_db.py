"""Tests for SQLite database layer."""

import asyncio
import pytest
from specq.models import Status, WorkItem, TaskItem, VoteResult
from specq.db import Database


# --- Basic CRUD ---

@pytest.mark.asyncio
async def test_create_and_get_work_item(memory_db):
    """Create WorkItem → read back with matching fields."""
    wi = WorkItem(
        id="001-auth", change_dir="changes/001-auth",
        title="Auth", description="Add auth",
        deps=["000-user"], risk="high", priority=10,
    )
    await memory_db.upsert_work_item(wi)
    loaded = await memory_db.get_work_item("001-auth")
    assert loaded.id == "001-auth"
    assert loaded.deps == ["000-user"]
    assert loaded.risk == "high"
    assert loaded.priority == 10
    assert loaded.status == Status.PENDING


@pytest.mark.asyncio
async def test_upsert_idempotent(memory_db):
    """Repeated upsert of same ID → update, not error."""
    wi = WorkItem(id="001", change_dir="c/001", title="v1", description="")
    await memory_db.upsert_work_item(wi)
    wi.title = "v2"
    wi.status = Status.READY
    await memory_db.upsert_work_item(wi)
    loaded = await memory_db.get_work_item("001")
    assert loaded.title == "v2"
    assert loaded.status == Status.READY


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(memory_db):
    """Query non-existent ID → None."""
    result = await memory_db.get_work_item("nonexistent")
    assert result is None


# --- Status updates ---

@pytest.mark.asyncio
async def test_update_status_changes_updated_at(memory_db):
    """Updating status also updates updated_at."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    await memory_db.upsert_work_item(wi)
    original = await memory_db.get_work_item("001")

    await asyncio.sleep(0.01)

    await memory_db.update_status("001", Status.RUNNING)
    updated = await memory_db.get_work_item("001")
    assert updated.status == Status.RUNNING
    assert updated.updated_at >= original.updated_at


@pytest.mark.asyncio
async def test_list_by_status(memory_db):
    """Query by status returns correct items."""
    for i, status in enumerate([Status.READY, Status.READY, Status.RUNNING, Status.BLOCKED]):
        wi = WorkItem(id=f"00{i}", change_dir=f"c/00{i}", title=f"t{i}", description="")
        wi.status = status
        await memory_db.upsert_work_item(wi)

    ready = await memory_db.list_by_status(Status.READY)
    assert len(ready) == 2
    running = await memory_db.list_by_status(Status.RUNNING)
    assert len(running) == 1
    blocked = await memory_db.list_by_status(Status.BLOCKED)
    assert len(blocked) == 1


# --- Task CRUD ---

@pytest.mark.asyncio
async def test_task_full_lifecycle(memory_db):
    """Task lifecycle: create → update execution results."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    await memory_db.upsert_work_item(wi)

    task = TaskItem(id="task-1", title="JWT", description="impl jwt")
    await memory_db.upsert_task("001", task)

    tasks = await memory_db.get_tasks("001")
    assert len(tasks) == 1
    assert tasks[0].title == "JWT"
    assert tasks[0].turns_used == 0

    # Update
    task.status = Status.ACCEPTED
    task.turns_used = 12
    task.tokens_in = 5000
    task.tokens_out = 3000
    task.files_changed = ["auth.py", "test_auth.py"]
    task.commit_hash = "abc123"
    task.duration_sec = 45.2
    await memory_db.upsert_task("001", task)

    tasks = await memory_db.get_tasks("001")
    assert tasks[0].turns_used == 12
    assert tasks[0].tokens_in == 5000
    assert tasks[0].files_changed == ["auth.py", "test_auth.py"]
    assert tasks[0].duration_sec == 45.2


# --- Vote Results ---

@pytest.mark.asyncio
async def test_save_and_get_votes(memory_db):
    """Vote results stored and retrieved correctly."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    await memory_db.upsert_work_item(wi)

    votes = [
        VoteResult(voter="openai/gpt-4o", verdict="pass", confidence=0.9, findings=[], summary="ok"),
        VoteResult(voter="google/gemini", verdict="fail", confidence=0.7,
                   findings=[{"severity": "critical", "description": "bug"}], summary="fail"),
    ]
    await memory_db.save_vote_results("001", attempt=1, results=votes)

    loaded = await memory_db.get_vote_results("001", attempt=1)
    assert len(loaded) == 2
    assert loaded[0].verdict == "pass"
    assert loaded[1].findings[0]["severity"] == "critical"


@pytest.mark.asyncio
async def test_multiple_vote_attempts_isolated(memory_db):
    """Multiple vote attempts stored separately by attempt number."""
    wi = WorkItem(id="001", change_dir="c/001", title="t", description="")
    await memory_db.upsert_work_item(wi)

    v1 = [VoteResult(voter="a", verdict="fail", confidence=0.6, findings=[], summary="")]
    v2 = [VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary="")]
    await memory_db.save_vote_results("001", attempt=1, results=v1)
    await memory_db.save_vote_results("001", attempt=2, results=v2)

    r1 = await memory_db.get_vote_results("001", attempt=1)
    r2 = await memory_db.get_vote_results("001", attempt=2)
    assert r1[0].verdict == "fail"
    assert r2[0].verdict == "pass"


# --- Run Log ---

@pytest.mark.asyncio
async def test_run_log_ordered(memory_db):
    """Event logs recorded in order."""
    await memory_db.log_event("001", "scan", {"source": "changes/001"})
    await memory_db.log_event("001", "compile", {"model": "haiku"})
    await memory_db.log_event("001", "execute", {"turns": 12})

    logs = await memory_db.get_logs("001")
    assert len(logs) == 3
    assert [l["event"] for l in logs] == ["scan", "compile", "execute"]


# --- JSON roundtrip ---

@pytest.mark.asyncio
async def test_json_fields_roundtrip(memory_db):
    """JSON fields (deps, files_changed) survive serialization."""
    wi = WorkItem(
        id="001", change_dir="c/001", title="t", description="",
        deps=["a", "b", "c"],
    )
    await memory_db.upsert_work_item(wi)
    loaded = await memory_db.get_work_item("001")
    assert loaded.deps == ["a", "b", "c"]
    assert isinstance(loaded.deps, list)


# --- WAL mode ---

@pytest.mark.asyncio
async def test_wal_mode_enabled(db):
    """File database has WAL mode enabled."""
    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        cursor = await conn.execute("PRAGMA journal_mode")
        mode = (await cursor.fetchone())[0]
        assert mode == "wal"
