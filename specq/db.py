"""SQLite state persistence with WAL mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from .models import Status, TaskItem, VoteResult, WorkItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
    id TEXT PRIMARY KEY,
    change_dir TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    risk TEXT NOT NULL DEFAULT 'medium',
    priority INTEGER DEFAULT 0,
    deps TEXT DEFAULT '[]',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    compiled_brief TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT NOT NULL,
    work_item_id TEXT NOT NULL REFERENCES work_items(id),
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    files_changed TEXT DEFAULT '[]',
    commit_hash TEXT DEFAULT '',
    execution_output TEXT DEFAULT '',
    turns_used INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    duration_sec REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (work_item_id, id)
);

CREATE TABLE IF NOT EXISTS vote_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_item_id TEXT NOT NULL REFERENCES work_items(id),
    attempt INTEGER NOT NULL,
    voter TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL,
    findings TEXT DEFAULT '[]',
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_item_id TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_work_items_status ON work_items(status);
CREATE INDEX IF NOT EXISTS idx_tasks_work_item ON tasks(work_item_id);
CREATE INDEX IF NOT EXISTS idx_votes_work_item ON vote_results(work_item_id);
CREATE INDEX IF NOT EXISTS idx_run_log_work_item ON run_log(work_item_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        if self.db_path != ":memory:":
            await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ---------------------------------------------------------------
    # WorkItem CRUD
    # ---------------------------------------------------------------

    async def upsert_work_item(self, wi: WorkItem) -> None:
        now = _now()
        if not wi.created_at:
            wi.created_at = now
        wi.updated_at = now
        await self._conn.execute(
            """INSERT INTO work_items
               (id, change_dir, title, description, status, risk, priority,
                deps, retry_count, max_retries, compiled_brief, error_message,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 change_dir=excluded.change_dir,
                 title=excluded.title,
                 description=excluded.description,
                 status=excluded.status,
                 risk=excluded.risk,
                 priority=excluded.priority,
                 deps=excluded.deps,
                 retry_count=excluded.retry_count,
                 max_retries=excluded.max_retries,
                 compiled_brief=excluded.compiled_brief,
                 error_message=excluded.error_message,
                 updated_at=excluded.updated_at
            """,
            (
                wi.id, wi.change_dir, wi.title, wi.description,
                wi.status.value if isinstance(wi.status, Status) else wi.status,
                wi.risk, wi.priority,
                json.dumps(wi.deps), wi.retry_count, wi.max_retries,
                wi.compiled_brief, wi.error_message,
                wi.created_at, wi.updated_at,
            ),
        )
        await self._conn.commit()

    async def get_work_item(self, work_item_id: str) -> WorkItem | None:
        cursor = await self._conn.execute(
            "SELECT * FROM work_items WHERE id = ?", (work_item_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_work_item(row)

    async def list_all_work_items(self) -> list[WorkItem]:
        cursor = await self._conn.execute("SELECT * FROM work_items ORDER BY id")
        rows = await cursor.fetchall()
        return [self._row_to_work_item(r) for r in rows]

    async def list_by_status(self, status: Status) -> list[WorkItem]:
        cursor = await self._conn.execute(
            "SELECT * FROM work_items WHERE status = ? ORDER BY id",
            (status.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work_item(r) for r in rows]

    async def update_status(self, work_item_id: str, status: Status) -> None:
        await self._conn.execute(
            "UPDATE work_items SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, _now(), work_item_id),
        )
        await self._conn.commit()

    async def update_retry_count(self, work_item_id: str, count: int) -> None:
        await self._conn.execute(
            "UPDATE work_items SET retry_count = ?, updated_at = ? WHERE id = ?",
            (count, _now(), work_item_id),
        )
        await self._conn.commit()

    async def update_compiled_brief(self, work_item_id: str, brief: str) -> None:
        await self._conn.execute(
            "UPDATE work_items SET compiled_brief = ?, updated_at = ? WHERE id = ?",
            (brief, _now(), work_item_id),
        )
        await self._conn.commit()

    async def update_error_message(self, work_item_id: str, msg: str) -> None:
        await self._conn.execute(
            "UPDATE work_items SET error_message = ?, updated_at = ? WHERE id = ?",
            (msg, _now(), work_item_id),
        )
        await self._conn.commit()

    # ---------------------------------------------------------------
    # Task CRUD
    # ---------------------------------------------------------------

    async def upsert_task(self, work_item_id: str, task: TaskItem) -> None:
        now = _now()
        await self._conn.execute(
            """INSERT INTO tasks
               (id, work_item_id, title, description, status,
                files_changed, commit_hash, execution_output,
                turns_used, tokens_in, tokens_out, duration_sec,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(work_item_id, id) DO UPDATE SET
                 title=excluded.title,
                 description=excluded.description,
                 status=excluded.status,
                 files_changed=excluded.files_changed,
                 commit_hash=excluded.commit_hash,
                 execution_output=excluded.execution_output,
                 turns_used=excluded.turns_used,
                 tokens_in=excluded.tokens_in,
                 tokens_out=excluded.tokens_out,
                 duration_sec=excluded.duration_sec,
                 updated_at=excluded.updated_at
            """,
            (
                task.id, work_item_id, task.title, task.description,
                task.status.value if isinstance(task.status, Status) else task.status,
                json.dumps(task.files_changed), task.commit_hash,
                task.execution_output, task.turns_used,
                task.tokens_in, task.tokens_out, task.duration_sec,
                now, now,
            ),
        )
        await self._conn.commit()

    async def get_tasks(self, work_item_id: str) -> list[TaskItem]:
        cursor = await self._conn.execute(
            "SELECT * FROM tasks WHERE work_item_id = ? ORDER BY id",
            (work_item_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(r) for r in rows]

    # ---------------------------------------------------------------
    # Vote Results
    # ---------------------------------------------------------------

    async def save_vote_results(
        self, work_item_id: str, attempt: int, results: list[VoteResult]
    ) -> None:
        now = _now()
        for vr in results:
            await self._conn.execute(
                """INSERT INTO vote_results
                   (work_item_id, attempt, voter, verdict, confidence, findings, summary, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    work_item_id, attempt, vr.voter, vr.verdict,
                    vr.confidence, json.dumps(vr.findings), vr.summary, now,
                ),
            )
        await self._conn.commit()

    async def get_vote_results(
        self, work_item_id: str, attempt: int
    ) -> list[VoteResult]:
        cursor = await self._conn.execute(
            "SELECT * FROM vote_results WHERE work_item_id = ? AND attempt = ? ORDER BY id",
            (work_item_id, attempt),
        )
        rows = await cursor.fetchall()
        return [
            VoteResult(
                voter=r["voter"],
                verdict=r["verdict"],
                confidence=r["confidence"],
                findings=json.loads(r["findings"]) if r["findings"] else [],
                summary=r["summary"] or "",
            )
            for r in rows
        ]

    # ---------------------------------------------------------------
    # Run Log
    # ---------------------------------------------------------------

    async def log_event(
        self, work_item_id: str, event: str, detail: dict | None = None
    ) -> None:
        await self._conn.execute(
            "INSERT INTO run_log (work_item_id, event, detail, created_at) VALUES (?,?,?,?)",
            (work_item_id, event, json.dumps(detail) if detail else None, _now()),
        )
        await self._conn.commit()

    async def get_logs(self, work_item_id: str) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM run_log WHERE work_item_id = ? ORDER BY id",
            (work_item_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "event": r["event"],
                "detail": json.loads(r["detail"]) if r["detail"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _row_to_work_item(row) -> WorkItem:
        return WorkItem(
            id=row["id"],
            change_dir=row["change_dir"],
            title=row["title"],
            description=row["description"] or "",
            deps=json.loads(row["deps"]) if row["deps"] else [],
            priority=row["priority"],
            risk=row["risk"],
            status=Status(row["status"]),
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            compiled_brief=row["compiled_brief"] or "",
            error_message=row["error_message"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_task(row) -> TaskItem:
        return TaskItem(
            id=row["id"],
            title=row["title"],
            description=row["description"] or "",
            status=Status(row["status"]),
            files_changed=json.loads(row["files_changed"]) if row["files_changed"] else [],
            commit_hash=row["commit_hash"] or "",
            execution_output=row["execution_output"] or "",
            turns_used=row["turns_used"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            duration_sec=row["duration_sec"],
        )
