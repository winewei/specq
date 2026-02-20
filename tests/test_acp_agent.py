"""Tests for ACPSubprocessAgent, GeminiCLIAgent, CodexAgent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from specq.providers.acp_agent import (
    ACPSubprocessAgent,
    CodexAgent,
    GeminiCLIAgent,
    _INIT_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rpc(method: str, params: dict | None = None) -> bytes:
    """Build a newline-terminated JSON-RPC notification line."""
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


def _rpc_result(req_id: int, result: dict) -> bytes:
    """Build a newline-terminated JSON-RPC response line."""
    msg = {"jsonrpc": "2.0", "id": req_id, "result": result}
    return (json.dumps(msg) + "\n").encode()


def _rpc_error(req_id: int, code: int, message: str) -> bytes:
    """Build a newline-terminated JSON-RPC error response."""
    msg = {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
    return (json.dumps(msg) + "\n").encode()


def _init_response() -> bytes:
    """Minimal initialize response."""
    return _rpc_result(1, {
        "protocolVersion": "0.1",
        "capabilities": {},
        "serverInfo": {"name": "test-agent", "version": "0.1.0"},
    })


def _make_proc(stdout_lines: list[bytes]) -> MagicMock:
    """Return a mock asyncio subprocess with the given stdout lines."""
    proc = MagicMock()

    # stdin
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    # stdout — readline() pops from the list, returns b"" at the end (EOF)
    lines_iter = iter(stdout_lines + [b""])

    async def _readline():
        return next(lines_iter, b"")

    proc.stdout = MagicMock()
    proc.stdout.readline = _readline

    # stderr — returns immediately (nothing to drain)
    async def _stderr_read():
        return b""

    proc.stderr = MagicMock()
    proc.stderr.read = _stderr_read

    # wait / terminate
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()

    return proc


# ---------------------------------------------------------------------------
# Basic happy-path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_deltas_concatenated_without_separator(tmp_path):
    """Text deltas must be joined with '' not '\\n'."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "Hello"}}),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": ", "}}),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "world!"}}),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="do something", cwd=tmp_path)

    assert result.success is True
    assert result.output == "Hello, world!"


@pytest.mark.asyncio
async def test_agents_done_stops_loop(tmp_path):
    """agents/done notification terminates the read loop."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "done"}}),
        _rpc("agents/done"),
        # These lines should never be reached
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": " EXTRA"}}),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is True
    assert result.output == "done"


@pytest.mark.asyncio
async def test_final_rpc_response_extracts_text(tmp_path):
    """Final agents/run JSON-RPC response also yields output text."""
    final_resp = _rpc_result(2, {
        "agentsRunId": "r1",
        "output": [
            {"role": "assistant", "content": [{"type": "text", "text": "final answer"}]}
        ],
    })
    stdout = [_init_response(), final_resp]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="q", cwd=tmp_path)

    assert result.success is True
    assert result.output == "final answer"


@pytest.mark.asyncio
async def test_turn_count_from_turn_done(tmp_path):
    """Each agents/turnDone notification increments the turn counter."""
    stdout = [
        _init_response(),
        _rpc("agents/turnDone"),
        _rpc("agents/turnDone"),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.turns == 2


# ---------------------------------------------------------------------------
# Permission handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_request_auto_approved(tmp_path):
    """permissions/requested triggers a permissions/granted response."""
    written: list[str] = []

    proc = _make_proc([
        _init_response(),
        _rpc("permissions/requested", {"permissionsRequestId": "perm-42"}),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "ok"}}),
        _rpc("agents/done"),
    ])

    # Capture what gets written to stdin
    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is True
    assert result.output == "ok"

    # Verify a permissions/granted was sent
    granted = [m for m in written if m.get("method") == "permissions/granted"]
    assert len(granted) == 1
    assert granted[0]["params"]["permissionsRequestId"] == "perm-42"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_prompt_included_in_agents_run(tmp_path):
    """System prompt is sent as a 'system' role message in agents/run."""
    written: list[dict] = []

    proc = _make_proc([_init_response(), _rpc("agents/done")])

    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        await agent.run(prompt="task", cwd=tmp_path, system_prompt="Be concise.")

    run_msgs = [m for m in written if m.get("method") == "agents/run"]
    assert len(run_msgs) == 1
    input_msgs = run_msgs[0]["params"]["input"]
    system_msg = next((m for m in input_msgs if m["role"] == "system"), None)
    assert system_msg is not None
    assert "Be concise." in system_msg["content"][0]["text"]


@pytest.mark.asyncio
async def test_no_system_role_when_prompt_empty(tmp_path):
    """Empty system prompt → no 'system' role message sent."""
    written: list[dict] = []

    proc = _make_proc([_init_response(), _rpc("agents/done")])

    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"], system_prompt="")
        await agent.run(prompt="task", cwd=tmp_path)

    run_msgs = [m for m in written if m.get("method") == "agents/run"]
    input_msgs = run_msgs[0]["params"]["input"]
    assert not any(m["role"] == "system" for m in input_msgs)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_not_found_returns_failure(tmp_path):
    """FileNotFoundError → success=False with helpful message."""
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError()),
    ):
        agent = ACPSubprocessAgent(cmd=["nonexistent-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is False
    assert "nonexistent-cli" in result.output
    assert "PATH" in result.output


@pytest.mark.asyncio
async def test_acp_error_response_returns_failure(tmp_path):
    """ACP JSON-RPC error response → success=False."""
    stdout = [_init_response(), _rpc_error(2, -32600, "Invalid request")]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is False
    assert "-32600" in result.output
    assert "Invalid request" in result.output


@pytest.mark.asyncio
async def test_eof_before_done_still_returns_output(tmp_path):
    """EOF (subprocess exits cleanly) before agents/done → still returns collected output."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "partial"}}),
        # No agents/done — subprocess just exits
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is True
    assert result.output == "partial"


@pytest.mark.asyncio
async def test_eof_with_nonzero_returncode_returns_failure(tmp_path):
    """EOF + non-zero returncode → success=False (subprocess crashed)."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "partial"}}),
        # Subprocess crashes — no agents/done
    ]
    proc = _make_proc(stdout)
    proc.returncode = 1  # non-zero exit code

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is False
    assert "1" in result.output  # exit code present


@pytest.mark.asyncio
async def test_initialize_timeout_returns_failure(tmp_path):
    """If CLI hangs on initialize, timeout fires and returns failure."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()

    # readline() blocks forever — simulate by never resolving
    hang = asyncio.get_event_loop().create_future()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(return_value=(await asyncio.sleep(0) or hang))

    async def _hanging_readline():
        await asyncio.sleep(9999)
        return b""  # never reached

    proc.stdout.readline = _hanging_readline

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
         patch("specq.providers.acp_agent._INIT_TIMEOUT", 0.05):
        agent = ACPSubprocessAgent(cmd=["slow-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is False
    assert "timed out" in result.output.lower()


@pytest.mark.asyncio
async def test_malformed_json_line_skipped(tmp_path):
    """Non-JSON lines from stdout are silently skipped."""
    stdout = [
        _init_response(),
        b"not valid json\n",
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "good"}}),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is True
    assert result.output == "good"


# ---------------------------------------------------------------------------
# Subclass command-line construction
# ---------------------------------------------------------------------------

def test_gemini_cli_cmd_no_model():
    """GeminiCLIAgent: without model, uses --experimental-acp only."""
    agent = GeminiCLIAgent()
    assert agent._cmd == ["gemini", "--experimental-acp"]


def test_gemini_cli_cmd_with_model():
    """GeminiCLIAgent: model is appended with --model flag."""
    agent = GeminiCLIAgent(model="gemini-2.5-pro")
    assert agent._cmd == ["gemini", "--experimental-acp", "--model", "gemini-2.5-pro"]


def test_codex_agent_cmd_no_model():
    """CodexAgent: without model, uses --mode acp only."""
    agent = CodexAgent()
    assert agent._cmd == ["codex", "--mode", "acp"]


def test_codex_agent_cmd_with_model():
    """CodexAgent: model is appended with --model flag."""
    agent = CodexAgent(model="o3")
    assert agent._cmd == ["codex", "--mode", "acp", "--model", "o3"]


# ---------------------------------------------------------------------------
# Initialize response validation (issue #9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_error_response_returns_failure(tmp_path):
    """If the CLI returns a JSON-RPC error for initialize, run() returns failure."""
    init_error = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32600, "message": "Unsupported protocol version"},
    }).encode() + b"\n"
    proc = _make_proc([init_error, _rpc("agents/done")])

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    assert result.success is False
    assert "initialize failed" in result.output.lower()
    assert "Unsupported protocol version" in result.output


# ---------------------------------------------------------------------------
# Permission auto-approval gate (issue #10)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_not_granted_when_disabled(tmp_path):
    """auto_approve_permissions=False → permissions/granted is NOT sent."""
    written: list[dict] = []

    proc = _make_proc([
        _init_response(),
        _rpc("permissions/requested", {"permissionsRequestId": "perm-99"}),
        _rpc("agents/done"),
    ])

    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"], auto_approve_permissions=False)
        await agent.run(prompt="task", cwd=tmp_path)

    granted = [m for m in written if m.get("method") == "permissions/granted"]
    assert len(granted) == 0


# ---------------------------------------------------------------------------
# Output deduplication strategy (issue #11)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_result_ignored_when_deltas_present(tmp_path):
    """When streaming deltas were received, the final agents/run result text is ignored."""
    final_resp = _rpc_result(2, {
        "output": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hello, world! (duplicate summary)"}
            ]}
        ],
    })
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "Hello,"}}),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": " world!"}}),
        final_resp,
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        agent = ACPSubprocessAgent(cmd=["fake-cli"])
        result = await agent.run(prompt="task", cwd=tmp_path)

    # Output should come from streaming deltas only — no duplication
    assert result.output == "Hello, world!"
