"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner
from specq.cli import app

runner = CliRunner()


def test_init_creates_structure(tmp_path, monkeypatch):
    """specq init creates .specq/ + config.yaml + changes/ + .gitignore."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / ".specq" / "config.yaml").exists()
    assert (tmp_path / ".specq" / "local.config.yaml").exists()
    assert (tmp_path / "changes").exists()
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".specq/local.config.yaml" in gitignore
    assert ".specq/state.db" in gitignore


def test_init_idempotent(tmp_path, monkeypatch):
    """specq init repeated does not overwrite existing config."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    (tmp_path / ".specq" / "config.yaml").write_text("custom: true")
    runner.invoke(app, ["init"])
    assert "custom: true" in (tmp_path / ".specq" / "config.yaml").read_text()


def test_plan_shows_dag(tmp_project, multi_changes, monkeypatch):
    """specq plan shows DAG with all changes."""
    monkeypatch.chdir(tmp_project)
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    assert "001-user-model" in result.output
    assert "004-rate-limit" in result.output


def test_status_empty_project(tmp_project, monkeypatch):
    """Empty project status doesn't crash."""
    monkeypatch.chdir(tmp_project)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


def test_scan_finds_changes(tmp_project, sample_change, monkeypatch):
    """specq scan finds and lists changes."""
    monkeypatch.chdir(tmp_project)
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "001-add-auth" in result.output


def test_config_show(tmp_project, monkeypatch):
    """specq config shows merged config."""
    monkeypatch.chdir(tmp_project)
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "changes_dir" in result.output


def test_init_detects_openspec(tmp_path, monkeypatch):
    """specq init detects openspec/changes/ and skips creating changes/."""
    (tmp_path / "openspec" / "changes").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert not (tmp_path / "changes").exists()
    assert "openspec/changes" in result.output


def test_init_no_openspec_creates_changes(tmp_path, monkeypatch):
    """specq init creates changes/ when no openspec directory exists."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / "changes").exists()
