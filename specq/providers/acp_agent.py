"""ACP (Agent Client Protocol) subprocess agents — Gemini CLI and Codex CLI.

ACP is a JSON-RPC 2.0 protocol over stdin/stdout used by Claude Code,
Codex CLI, Gemini CLI, and other coding agent CLIs.
See: https://agentclientprotocol.com
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .code_agent import AgentRun


class ACPSubprocessAgent:
    """Generic coding agent that drives a CLI subprocess via ACP.

    Message flow:
      1. Spawn subprocess with ACP flags
      2. Send  ``initialize`` request, wait for response
      3. Send  ``initialized`` notification
      4. Send  ``agents/run`` request with prompt
      5. Read streaming notifications (text deltas, permission requests …)
      6. Auto-approve any ``permissions/requested`` from the agent
      7. Stop on ``agents/done`` notification or final ``agents/run`` response
    """

    name = "acp"

    def __init__(
        self,
        cmd: list[str],
        max_turns: int = 50,
        system_prompt: str = "",
    ):
        self._cmd = cmd
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    async def run(
        self,
        prompt: str,
        cwd: Path,
        system_prompt: str | None = None,
    ) -> AgentRun:
        """Spawn the CLI, send *prompt*, collect output, return AgentRun."""
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        start = time.monotonic()
        output_parts: list[str] = []
        turns = 0
        proc = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )

            _id = 0

            def next_id() -> int:
                nonlocal _id
                _id += 1
                return _id

            async def send(msg: dict) -> None:
                proc.stdin.write((json.dumps(msg) + "\n").encode())
                await proc.stdin.drain()

            # ── 1. Initialize ──────────────────────────────────────────────
            init_id = next_id()
            await send({
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1",
                    "capabilities": {},
                    "clientInfo": {"name": "specq", "version": "0.1.0"},
                },
            })

            # Wait for the server's initialize response (one line)
            await proc.stdout.readline()

            # Send initialized notification
            await send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

            # ── 2. Build input messages ────────────────────────────────────
            input_msgs: list[dict] = []
            if effective_system:
                input_msgs.append({
                    "role": "system",
                    "content": [{"type": "text", "text": effective_system}],
                })
            input_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            })

            run_req_id = next_id()
            await send({
                "jsonrpc": "2.0",
                "id": run_req_id,
                "method": "agents/run",
                "params": {"input": input_msgs},
            })

            # ── 3. Collect streaming output ────────────────────────────────
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break  # EOF — subprocess exited

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                method = msg.get("method", "")

                # Auto-approve any permission request so the agent can use tools
                if method == "permissions/requested":
                    perm_id = msg.get("params", {}).get("permissionsRequestId", "")
                    await send({
                        "jsonrpc": "2.0",
                        "method": "permissions/granted",
                        "params": {"permissionsRequestId": perm_id},
                    })
                    continue

                # Streaming text from the agent
                if method == "agents/textDelta":
                    delta = msg.get("params", {}).get("delta", {})
                    if delta.get("type") == "text":
                        output_parts.append(delta["text"])
                    continue

                # Agent completed one turn
                if method == "agents/turnDone":
                    turns += 1
                    continue

                # Agent signals it is fully done
                if method == "agents/done":
                    break

                # Final JSON-RPC response to our agents/run request
                if "id" in msg and msg.get("id") == run_req_id:
                    if "result" in msg:
                        for out in msg["result"].get("output", []):
                            for blk in out.get("content", []):
                                if blk.get("type") == "text":
                                    text = blk["text"]
                                    if text not in output_parts:
                                        output_parts.append(text)
                    elif "error" in msg:
                        err = msg["error"]
                        return AgentRun(
                            success=False,
                            output=(
                                f"ACP error {err.get('code')}: {err.get('message')}"
                            ),
                            turns=turns,
                            duration_sec=time.monotonic() - start,
                        )
                    break

        except FileNotFoundError:
            cmd_name = self._cmd[0] if self._cmd else "unknown"
            return AgentRun(
                success=False,
                output=(
                    f"CLI not found: '{cmd_name}'. "
                    "Please install it and ensure it is on PATH."
                ),
                duration_sec=time.monotonic() - start,
            )
        except Exception as exc:
            return AgentRun(
                success=False,
                output=f"ACP agent error: {exc}",
                turns=turns,
                duration_sec=time.monotonic() - start,
            )
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                    await asyncio.wait_for(proc.wait(), timeout=10.0)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass

        return AgentRun(
            success=True,
            output="\n".join(output_parts),
            turns=turns,
            duration_sec=time.monotonic() - start,
        )


class GeminiCLIAgent(ACPSubprocessAgent):
    """Gemini CLI coding agent via ACP protocol.

    Requires: ``gemini`` CLI installed and authenticated.

    Invokes::

        gemini --experimental-acp [--model <model>]

    Configure in ``.specq/config.yaml``::

        executor:
          type: gemini_cli
          model: gemini-2.5-pro   # optional; CLI default if omitted
    """

    name = "gemini_cli"

    def __init__(self, model: str = "", max_turns: int = 50, system_prompt: str = ""):
        self.model = model
        cmd = ["gemini", "--experimental-acp"]
        if model:
            cmd += ["--model", model]
        super().__init__(cmd=cmd, max_turns=max_turns, system_prompt=system_prompt)


class CodexAgent(ACPSubprocessAgent):
    """OpenAI Codex CLI coding agent via ACP protocol.

    Requires: ``codex`` CLI installed and ``OPENAI_API_KEY`` set.

    Invokes::

        codex --mode acp [--model <model>]

    Configure in ``.specq/config.yaml``::

        executor:
          type: codex
          model: o3              # optional; CLI default if omitted
    """

    name = "codex"

    def __init__(self, model: str = "", max_turns: int = 50, system_prompt: str = ""):
        self.model = model
        cmd = ["codex", "--mode", "acp"]
        if model:
            cmd += ["--model", model]
        super().__init__(cmd=cmd, max_turns=max_turns, system_prompt=system_prompt)
