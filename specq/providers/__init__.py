"""LLM provider abstraction — unified AI capabilities.

Two capability types:
  - TextGen  (HttpTextGen, ClaudeCodeTextGen)  — single-turn text generation
  - CodeAgent (ClaudeCodeAgent)                — multi-turn coding agent
"""

from .text_gen import ENDPOINTS, ClaudeCodeTextGen, HttpTextGen
from .code_agent import AgentRun, ClaudeCodeAgent

__all__ = [
    # Text generation
    "HttpTextGen",
    "ClaudeCodeTextGen",
    "ENDPOINTS",
    # Coding agent
    "ClaudeCodeAgent",
    "AgentRun",
]
