"""Three-layer config loading and merging."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProviderCreds:
    api_key: str = ""


@dataclass
class ProvidersConfig:
    anthropic: ProviderCreds = field(default_factory=ProviderCreds)
    openai: ProviderCreds = field(default_factory=ProviderCreds)
    google: ProviderCreds = field(default_factory=ProviderCreds)


@dataclass
class CompilerConfig:
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"


@dataclass
class ExecutorConfig:
    type: str = "claude_code"
    model: str = "claude-sonnet-4-5"
    max_turns: int = 50


@dataclass
class VoterEntry:
    provider: str = ""
    model: str = ""


@dataclass
class VerificationConfig:
    voters: list[dict] = field(default_factory=list)
    checks: list[str] = field(default_factory=lambda: [
        "spec_compliance", "regression_risk", "architecture",
    ])


@dataclass
class RiskStrategyConfig:
    strategy: str = "majority"


@dataclass
class RiskPolicyConfig:
    low: str | RiskStrategyConfig = "skip"
    medium: RiskStrategyConfig = field(
        default_factory=lambda: RiskStrategyConfig(strategy="majority")
    )
    high: RiskStrategyConfig = field(
        default_factory=lambda: RiskStrategyConfig(strategy="unanimous")
    )


@dataclass
class BudgetsConfig:
    max_retries: int = 3
    max_duration_sec: int = 600
    max_turns: int = 50
    daily_task_limit: int = 50


@dataclass
class NotifyConfig:
    webhook_url: str = ""
    events: list[str] = field(default_factory=lambda: [
        "change.completed", "change.failed", "change.needs_review",
    ])


@dataclass
class Config:
    changes_dir: str = ""  # empty = auto-detect (prefers openspec/changes if present)
    base_branch: str = "main"
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    risk_policy: RiskPolicyConfig = field(default_factory=RiskPolicyConfig)
    budgets: BudgetsConfig = field(default_factory=BudgetsConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    project_root: str = ""


# ---------------------------------------------------------------------------
# OpenSpec auto-detection
# ---------------------------------------------------------------------------

def detect_changes_dir(project_root: Path) -> str:
    """Return the changes directory path, preferring openspec/changes if it exists."""
    openspec_changes = project_root / "openspec" / "changes"
    if openspec_changes.is_dir():
        return "openspec/changes"
    return "changes"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Field-level deep merge. Lists are replaced, None values ignored."""
    result = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Dict → Config mapping
# ---------------------------------------------------------------------------

def _build_risk_policy(data: dict) -> RiskPolicyConfig:
    rp = RiskPolicyConfig()
    if "low" in data:
        rp.low = data["low"] if isinstance(data["low"], str) else RiskStrategyConfig(
            strategy=data["low"].get("strategy", "skip"))
    if "medium" in data:
        val = data["medium"]
        rp.medium = RiskStrategyConfig(
            strategy=val if isinstance(val, str) else val.get("strategy", "majority"))
    if "high" in data:
        val = data["high"]
        rp.high = RiskStrategyConfig(
            strategy=val if isinstance(val, str) else val.get("strategy", "unanimous"))
    return rp


def _dict_to_config(data: dict, project_root: str) -> Config:
    cfg = Config(project_root=project_root)

    if "changes_dir" in data:
        cfg.changes_dir = data["changes_dir"]
    if "base_branch" in data:
        cfg.base_branch = data["base_branch"]

    if "compiler" in data and isinstance(data["compiler"], dict):
        c = data["compiler"]
        cfg.compiler = CompilerConfig(
            provider=c.get("provider", cfg.compiler.provider),
            model=c.get("model", cfg.compiler.model),
        )

    if "executor" in data and isinstance(data["executor"], dict):
        e = data["executor"]
        cfg.executor = ExecutorConfig(
            type=e.get("type", cfg.executor.type),
            model=e.get("model", cfg.executor.model),
            max_turns=e.get("max_turns", cfg.executor.max_turns),
        )

    if "verification" in data and isinstance(data["verification"], dict):
        v = data["verification"]
        cfg.verification = VerificationConfig(
            voters=v.get("voters", []),
            checks=v.get("checks", cfg.verification.checks),
        )

    if "risk_policy" in data and isinstance(data["risk_policy"], dict):
        cfg.risk_policy = _build_risk_policy(data["risk_policy"])

    if "budgets" in data and isinstance(data["budgets"], dict):
        b = data["budgets"]
        cfg.budgets = BudgetsConfig(
            max_retries=b.get("max_retries", cfg.budgets.max_retries),
            max_duration_sec=b.get("max_duration_sec", cfg.budgets.max_duration_sec),
            max_turns=b.get("max_turns", cfg.budgets.max_turns),
            daily_task_limit=b.get("daily_task_limit", cfg.budgets.daily_task_limit),
        )

    if "notify" in data and isinstance(data["notify"], dict):
        n = data["notify"]
        cfg.notify = NotifyConfig(
            webhook_url=n.get("webhook_url", ""),
            events=n.get("events", cfg.notify.events),
        )

    if "providers" in data and isinstance(data["providers"], dict):
        p = data["providers"]
        cfg.providers = ProvidersConfig(
            anthropic=ProviderCreds(api_key=p.get("anthropic", {}).get("api_key", "")),
            openai=ProviderCreds(api_key=p.get("openai", {}).get("api_key", "")),
            google=ProviderCreds(api_key=p.get("google", {}).get("api_key", "")),
        )

    return cfg


# ---------------------------------------------------------------------------
# Load config (3-layer)
# ---------------------------------------------------------------------------

def load_config(project_root: str | Path) -> Config:
    """Load and merge config from up to 3 layers.

    Priority (highest first):
      1. Environment variables (API keys)
      2. .specq/local.config.yaml
      3. .specq/config.yaml
    """
    project_root = Path(project_root)
    specq_dir = project_root / ".specq"

    # Layer 1: base config
    base_path = specq_dir / "config.yaml"
    base_data: dict = {}
    if base_path.exists():
        raw = base_path.read_text()
        parsed = yaml.safe_load(raw)
        if parsed is None:
            base_data = {}
        elif not isinstance(parsed, dict):
            raise ValueError(f"Invalid config.yaml: expected mapping, got {type(parsed).__name__}")
        else:
            base_data = parsed

    # Layer 2: local override
    local_path = specq_dir / "local.config.yaml"
    local_data: dict = {}
    if local_path.exists():
        raw = local_path.read_text()
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            local_data = parsed

    # Merge
    merged = deep_merge(base_data, local_data)
    cfg = _dict_to_config(merged, str(project_root))

    # Auto-detect changes_dir if not explicitly configured
    if not cfg.changes_dir:
        cfg.changes_dir = detect_changes_dir(project_root)

    # Layer 3: env vars override API keys (highest priority)
    env_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    if env_anthropic:
        cfg.providers.anthropic.api_key = env_anthropic

    env_openai = os.environ.get("OPENAI_API_KEY")
    if env_openai:
        cfg.providers.openai.api_key = env_openai

    env_google = os.environ.get("GOOGLE_API_KEY")
    if env_google:
        cfg.providers.google.api_key = env_google

    return cfg


def get_verification_strategy(work_item, config: Config) -> str:
    """Resolve verification strategy for a work item based on risk policy."""
    risk = work_item.risk
    if work_item.verification_strategy and work_item.verification_strategy != "majority":
        return work_item.verification_strategy

    rp = config.risk_policy
    if risk == "low":
        return rp.low if isinstance(rp.low, str) else rp.low.strategy
    elif risk == "high":
        return rp.high if isinstance(rp.high, str) else rp.high.strategy
    else:
        return rp.medium if isinstance(rp.medium, str) else rp.medium.strategy
