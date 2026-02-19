"""Tests for three-layer config loading and merging."""

import pytest
from specq.config import load_config, deep_merge, detect_changes_dir


# --- Three-layer merge ---

def test_load_default_config(tmp_project):
    """Load config.yaml correctly."""
    config = load_config(tmp_project)
    assert config.compiler.provider == "anthropic"
    assert config.compiler.model == "claude-haiku-4-5"
    assert config.executor.model == "claude-sonnet-4-5"
    assert config.executor.max_turns == 50


def test_local_overrides_base(tmp_project):
    """local.config.yaml overrides config.yaml field-by-field."""
    (tmp_project / ".specq" / "local.config.yaml").write_text("""\
executor:
  model: claude-sonnet-4-5
  max_turns: 30
compiler:
  provider: google
  model: gemini-2.5-flash
""")
    config = load_config(tmp_project)
    assert config.executor.max_turns == 30
    assert config.compiler.provider == "google"
    assert config.compiler.model == "gemini-2.5-flash"
    # Un-overridden fields keep original values
    assert config.executor.type == "claude_code"


def test_env_var_overrides_all(tmp_project, monkeypatch):
    """Environment variables have highest priority."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
    config = load_config(tmp_project)
    assert config.providers.anthropic.api_key == "env-key-123"


def test_env_var_beats_local_config(tmp_project, monkeypatch):
    """Env var overrides local.config.yaml API key."""
    (tmp_project / ".specq" / "local.config.yaml").write_text("""\
providers:
  anthropic:
    api_key: local-key-456
""")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-789")
    config = load_config(tmp_project)
    assert config.providers.anthropic.api_key == "env-key-789"


# --- deep merge edge cases ---

def test_deep_merge_nested_dicts():
    """deep merge: nested dicts are merged field-by-field."""
    base = {"a": {"x": 1, "y": 2}, "b": 10}
    override = {"a": {"y": 99, "z": 3}}
    result = deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}


def test_deep_merge_list_replaces():
    """deep merge: lists are replaced entirely (not appended)."""
    base = {"voters": [{"provider": "openai"}]}
    override = {"voters": [{"provider": "google"}, {"provider": "anthropic"}]}
    result = deep_merge(base, override)
    assert len(result["voters"]) == 2
    assert result["voters"][0]["provider"] == "google"


def test_deep_merge_none_ignored():
    """deep merge: None values do not override."""
    base = {"model": "haiku"}
    override = {"model": None}
    result = deep_merge(base, override)
    assert result["model"] == "haiku"


# --- Config validation ---

def test_missing_config_yaml_uses_defaults(tmp_path):
    """No config.yaml → all defaults."""
    (tmp_path / ".specq").mkdir()
    config = load_config(tmp_path)
    assert config.changes_dir == "changes"


def test_invalid_yaml_raises(tmp_project):
    """Invalid YAML syntax → clear error."""
    (tmp_project / ".specq" / "config.yaml").write_text("{{invalid yaml")
    with pytest.raises(Exception):
        load_config(tmp_project)


def test_unknown_fields_ignored(tmp_project):
    """Unknown fields in config don't raise errors."""
    (tmp_project / ".specq" / "config.yaml").write_text("""\
changes_dir: changes
some_future_field: true
""")
    config = load_config(tmp_project)
    assert config.changes_dir == "changes"


# --- OpenSpec auto-detection ---

def test_detect_changes_dir_openspec(tmp_path):
    """detect_changes_dir returns openspec/changes when directory exists."""
    (tmp_path / "openspec" / "changes").mkdir(parents=True)
    assert detect_changes_dir(tmp_path) == "openspec/changes"


def test_detect_changes_dir_fallback(tmp_path):
    """detect_changes_dir falls back to 'changes' when no openspec dir."""
    assert detect_changes_dir(tmp_path) == "changes"


def test_load_config_auto_detects_openspec(tmp_path):
    """load_config auto-detects openspec/changes when no changes_dir configured."""
    (tmp_path / ".specq").mkdir()
    (tmp_path / "openspec" / "changes").mkdir(parents=True)
    config = load_config(tmp_path)
    assert config.changes_dir == "openspec/changes"


def test_load_config_explicit_changes_dir_overrides_openspec(tmp_path):
    """Explicit changes_dir in config takes priority over openspec auto-detection."""
    (tmp_path / ".specq").mkdir()
    (tmp_path / "openspec" / "changes").mkdir(parents=True)
    (tmp_path / ".specq" / "config.yaml").write_text("changes_dir: custom/path\n")
    config = load_config(tmp_path)
    assert config.changes_dir == "custom/path"
