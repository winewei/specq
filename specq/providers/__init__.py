"""LLM provider abstraction — unified AI capabilities.

Two capability types:
  - TextGen  (HttpTextGen, ClaudeCodeTextGen)  — single-turn text generation
  - CodeAgent                                  — multi-turn coding agent

Available coding agents:
  - ClaudeCodeAgent  — Claude Code SDK (claude --mode acp)
  - GeminiCLIAgent   — Gemini CLI ACP  (gemini --experimental-acp)
  - CodexAgent       — Codex CLI ACP   (codex --mode acp)
"""

from .text_gen import ENDPOINTS, ClaudeCodeTextGen, HttpTextGen
from .code_agent import AgentRun, ClaudeCodeAgent
from .acp_agent import ACPSubprocessAgent, GeminiCLIAgent, CodexAgent

__all__ = [
    # Text generation
    "HttpTextGen",
    "ClaudeCodeTextGen",
    "ENDPOINTS",
    # Coding agents
    "AgentRun",
    "ClaudeCodeAgent",
    "ACPSubprocessAgent",
    "GeminiCLIAgent",
    "CodexAgent",
]
