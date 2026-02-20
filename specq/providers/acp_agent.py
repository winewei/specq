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

# Seconds to wait for the CLI to respond to the initialize request.
_INIT_TIMEOUT = 30.0


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

    Stderr is drained concurrently to prevent OS pipe buffer fill-up
    and the resulting deadlock that would occur for long-running agents.
    """

    name = "acp"

    def __init__(
        self,
        cmd: list[str],
        max_turns: int = 50,
        system_prompt: str = "",
        auto_approve_permissions: bool = True,
    ):
        self._cmd = cmd
        # max_turns is stored for interface compatibility but is NOT passed to the
        # ACP subprocess — the ACP protocol does not define a max_turns parameter
        # in agents/run.  Turn limiting is the responsibility of the CLI itself.
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        # When True, all permissions/requested notifications are automatically
        # granted so the agent can use tools without user interaction.  Set to
        # False in security-sensitive environments where tool use should be
        # restricted at the orchestrator layer rather than the CLI layer.
        self.auto_approve_permissions = auto_approve_permissions

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
        stderr_task: asyncio.Task | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )

            # Drain stderr concurrently so the OS pipe buffer never fills up,
            # which would block the subprocess and deadlock the event loop.
            async def _drain_stderr() -> None:
                try:
                    await proc.stderr.read()
                except Exception:
                    pass

            stderr_task = asyncio.create_task(_drain_stderr())

            _id = 0

            def next_id() -> int:
                nonlocal _id
                _id += 1
                return _id

            async def send(msg: dict) -> None:
                if proc.stdin is None:
                    raise RuntimeError("subprocess stdin is not available")
                proc.stdin.write((json.dumps(msg) + "\n").encode())
                await proc.stdin.drain()

            # ── 1. Initialize ──────────────────────────────────────────────
            await send({
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1",
                    "capabilities": {},
                    "clientInfo": {"name": "specq", "version": "0.1.0"},
                },
            })

            # Wait for the server's initialize response with a timeout so a
            # hanging CLI does not block indefinitely.  Parse the response and
            # fail fast if the server reports an error — this is far more
            # debuggable than letting the failure surface later in agents/run.
            try:
                init_line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=_INIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"ACP initialize timed out after {_INIT_TIMEOUT:.0f} s"
                )
            if init_line:
                try:
                    init_resp = json.loads(init_line.decode())
                    if "error" in init_resp:
                        err = init_resp["error"]
                        raise RuntimeError(
                            f"ACP initialize failed: "
                            f"{err.get('code')} {err.get('message')}"
                        )
                except json.JSONDecodeError:
                    pass  # Non-JSON banner line — proceed and let agents/run reveal any issue

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
            if proc.stdout is None:
                raise RuntimeError("subprocess stdout is not available")

            done_received = False
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break  # EOF — subprocess exited

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                method = msg.get("method", "")

                # Grant permission requests only when auto_approve_permissions is
                # enabled (the default for coding agent use-cases).  When disabled,
                # permission requests are silently ignored, which causes the CLI to
                # block or timeout — set it to False only if you restrict tool use
                # at a higher layer.
                if method == "permissions/requested":
                    if self.auto_approve_permissions:
                        perm_id = msg.get("params", {}).get("permissionsRequestId", "")
                        await send({
                            "jsonrpc": "2.0",
                            "method": "permissions/granted",
                            "params": {"permissionsRequestId": perm_id},
                        })
                    continue

                # Streaming text from the agent — deltas are character sequences,
                # not lines; concatenate them directly without any separator.
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
                    done_received = True
                    break

                # Final JSON-RPC response to our agents/run request.
                # Only use the result text when we received no streaming deltas —
                # when deltas ARE present they form the complete live output and
                # the final result is typically a summarised copy that would
                # produce duplicate or overlapping content if appended.
                if "id" in msg and msg.get("id") == run_req_id:
                    if "result" in msg:
                        if not output_parts:
                            for out in msg["result"].get("output", []):
                                for blk in out.get("content", []):
                                    if blk.get("type") == "text":
                                        output_parts.append(blk["text"])
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
            if stderr_task is not None:
                stderr_task.cancel()
                try:
                    await stderr_task
                except (asyncio.CancelledError, Exception):
                    pass
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                    await asyncio.wait_for(proc.wait(), timeout=10.0)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass

        output = "".join(output_parts)

        # If the subprocess exited (EOF) without sending agents/done, treat it as
        # a failure when the process returncode is a non-zero integer — this
        # distinguishes a clean early-EOF (returncode 0 or unknown) from a crash.
        if not done_received:
            returncode = getattr(proc, "returncode", None)
            if isinstance(returncode, int) and returncode != 0:
                return AgentRun(
                    success=False,
                    output=(
                        f"Subprocess exited with code {returncode} without completing."
                        + (f" Partial output: {output}" if output else "")
                    ),
                    turns=turns,
                    duration_sec=time.monotonic() - start,
                )

        # Text deltas are sequential character sequences — join with no separator.
        return AgentRun(
            success=True,
            output=output,
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

    def __init__(
        self,
        model: str = "",
        max_turns: int = 50,
        system_prompt: str = "",
        auto_approve_permissions: bool = True,
    ):
        self.model = model
        cmd = ["gemini", "--experimental-acp"]
        if model:
            cmd += ["--model", model]
        super().__init__(
            cmd=cmd,
            max_turns=max_turns,
            system_prompt=system_prompt,
            auto_approve_permissions=auto_approve_permissions,
        )


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

    def __init__(
        self,
        model: str = "",
        max_turns: int = 50,
        system_prompt: str = "",
        auto_approve_permissions: bool = True,
    ):
        self.model = model
        cmd = ["codex", "--mode", "acp"]
        if model:
            cmd += ["--model", model]
        super().__init__(
            cmd=cmd,
            max_turns=max_turns,
            system_prompt=system_prompt,
            auto_approve_permissions=auto_approve_permissions,
        )
