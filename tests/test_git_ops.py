"""Tests for git operations."""

import subprocess
import pytest
from specq.git_ops import get_changed_files, get_latest_commit, get_diff_from_base


@pytest.fixture
def git_repo(tmp_path):
    """Initialize a real git repo with initial branch 'main'."""
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"],
                   cwd=tmp_path, check=True, capture_output=True)
    # Disable commit signing for tests
    subprocess.run(["git", "config", "commit.gpgsign", "false"],
                   cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--no-gpg-sign", "-m", "init"],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.mark.asyncio
async def test_no_changes_empty_list(git_repo):
    """No changes → empty list."""
    files = await get_changed_files(git_repo)
    assert files == []


@pytest.mark.asyncio
async def test_detects_modified_files(git_repo):
    """Modified file is detected."""
    (git_repo / "README.md").write_text("# Updated")
    files = await get_changed_files(git_repo)
    assert "README.md" in files


@pytest.mark.asyncio
async def test_detects_new_files(git_repo):
    """New untracked file is detected."""
    (git_repo / "new.py").write_text("print('hello')")
    files = await get_changed_files(git_repo)
    assert "new.py" in files


@pytest.mark.asyncio
async def test_latest_commit_hash(git_repo):
    """Returns valid short commit hash."""
    commit = await get_latest_commit(git_repo)
    assert len(commit) >= 7
    assert all(c in "0123456789abcdef" for c in commit)


@pytest.mark.asyncio
async def test_diff_from_base_branch(git_repo):
    """Cross-branch diff includes new file content."""
    subprocess.run(["git", "checkout", "-b", "feature"],
                   cwd=git_repo, check=True, capture_output=True)
    (git_repo / "new.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--no-gpg-sign", "-m", "add new"],
                   cwd=git_repo, check=True, capture_output=True)

    diff = await get_diff_from_base(git_repo, "main")
    assert "new.py" in diff
    assert "hello" in diff
