"""Git operations wrapper using subprocess / anyio."""

from __future__ import annotations

import subprocess
from pathlib import Path

import anyio


async def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout."""
    result = await anyio.to_thread.run_sync(
        lambda: subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
    )
    return result.stdout.strip()


async def get_changed_files(cwd: Path) -> list[str]:
    """Get list of changed files (staged + unstaged + untracked)."""
    try:
        # Modified / staged
        diff_output = await _run_git(["diff", "--name-only", "HEAD"], cwd)
        files = [f for f in diff_output.splitlines() if f]
    except subprocess.CalledProcessError:
        files = []

    # Untracked
    try:
        untracked = await _run_git(
            ["ls-files", "--others", "--exclude-standard"], cwd
        )
        files.extend(f for f in untracked.splitlines() if f)
    except subprocess.CalledProcessError:
        pass

    return sorted(set(files))


async def get_latest_commit(cwd: Path) -> str:
    """Get the short hash of the latest commit."""
    try:
        return await _run_git(["rev-parse", "--short", "HEAD"], cwd)
    except subprocess.CalledProcessError:
        return ""


async def get_diff_from_base(cwd: Path, base_branch: str) -> str:
    """Get diff from base branch to HEAD."""
    try:
        return await _run_git(["diff", f"{base_branch}...HEAD"], cwd)
    except subprocess.CalledProcessError:
        # Fallback: diff against HEAD
        try:
            return await _run_git(["diff", "HEAD"], cwd)
        except subprocess.CalledProcessError:
            return ""


async def get_change_diff(cwd: Path, base_branch: str) -> str:
    """Get the full diff for voting/review."""
    return await get_diff_from_base(cwd, base_branch)
