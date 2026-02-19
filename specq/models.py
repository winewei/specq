"""Core data models for specq."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    COMPILING = "compiling"
    RUNNING = "running"
    VERIFYING = "verifying"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskItem:
    """A single task within a change."""

    id: str
    title: str
    description: str
    status: Status = Status.PENDING
    files_changed: list[str] = field(default_factory=list)
    commit_hash: str = ""
    execution_output: str = ""
    turns_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_sec: float = 0.0


@dataclass
class VoteResult:
    """Result from a single voter."""

    voter: str
    verdict: str  # "pass" | "fail" | "error"
    confidence: float
    findings: list[dict] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a task."""

    success: bool
    output: str
    files_changed: list[str] = field(default_factory=list)
    commit_hash: str = ""
    duration_sec: float = 0.0
    turns_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class WorkItem:
    """Complete representation of a change."""

    # Identity
    id: str
    change_dir: str
    title: str
    description: str

    # Dependencies & scheduling
    deps: list[str] = field(default_factory=list)
    priority: int = 0
    risk: str = "medium"

    # Config (merged from 3 layers)
    compiler_provider: str = "anthropic"
    compiler_model: str = "claude-haiku-4-5"
    executor_type: str = ""        # "" → use global config; "gemini_cli" | "codex" | "claude_code"
    executor_model: str = ""       # "" → use global config; non-empty → per-change override
    executor_max_turns: int = 0    # 0  → use global config; non-zero → per-change override
    executor_tools: list[str] = field(default_factory=list)  # [] → use global allowed_tools
    verification_strategy: str = "" # "" → use risk_policy; non-empty → per-change override
    voters: list[dict] = field(default_factory=list)

    # Budget
    max_retries: int = 3
    max_duration_sec: int = 600

    # Runtime
    status: Status = Status.PENDING
    tasks: list[TaskItem] = field(default_factory=list)
    retry_count: int = 0
    vote_results: list[VoteResult] = field(default_factory=list)
    compiled_brief: str = ""
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""
