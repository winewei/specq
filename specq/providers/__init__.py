"""LLM provider abstraction — unified AI capabilities.

Two capability types:
  - TextGen  (HttpTextGen, ClaudeCodeTextGen, ACPTextGen …)  — single-turn text generation
  - CodeAgent                                                — multi-turn coding agent

Each CLI-based provider supports BOTH capabilities:

  ┌──────────────┬────────────────────┬──────────────────────┐
  │ CLI          │ TextGen (Compiler/ │ CodeAgent (Executor) │
  │              │ Voters)            │                      │
  ├──────────────┼────────────────────┼──────────────────────┤
  │ Claude Code  │ ClaudeCodeTextGen  │ ClaudeCodeAgent      │
  │ Gemini CLI   │ GeminiCLITextGen   │ GeminiCLIAgent       │
  │ Codex CLI    │ CodexTextGen       │ CodexAgent           │
  └──────────────┴────────────────────┴──────────────────────┘

All CLI providers authenticate via their own login mechanism — no API key needed.
"""

from .text_gen import (
    ENDPOINTS,
    ClaudeCodeTextGen,
    HttpTextGen,
    ACPTextGen,
    GeminiCLITextGen,
    CodexTextGen,
)
from .code_agent import AgentRun, ClaudeCodeAgent
from .acp_agent import ACPSubprocessAgent, GeminiCLIAgent, CodexAgent

__all__ = [
    # Text generation
    "HttpTextGen",
    "ClaudeCodeTextGen",
    "ACPTextGen",
    "GeminiCLITextGen",
    "CodexTextGen",
    "ENDPOINTS",
    # Coding agents
    "AgentRun",
    "ClaudeCodeAgent",
    "ACPSubprocessAgent",
    "GeminiCLIAgent",
    "CodexAgent",
]
