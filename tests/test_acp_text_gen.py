"""Tests for ACPTextGen, GeminiCLITextGen, CodexTextGen."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from specq.providers.text_gen import ACPTextGen, CodexTextGen, GeminiCLITextGen


# ---------------------------------------------------------------------------
# Helpers — reuse the ACP mock pattern from test_acp_agent.py
# ---------------------------------------------------------------------------

def _rpc(method: str, params: dict | None = None) -> bytes:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


def _rpc_result(req_id: int, result: dict) -> bytes:
    msg = {"jsonrpc": "2.0", "id": req_id, "result": result}
    return (json.dumps(msg) + "\n").encode()


def _init_response() -> bytes:
    return _rpc_result(1, {
        "protocolVersion": "0.1",
        "capabilities": {},
        "serverInfo": {"name": "test-agent", "version": "0.1.0"},
    })


def _make_proc(stdout_lines: list[bytes]) -> MagicMock:
    proc = MagicMock()

    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    lines_iter = iter(stdout_lines + [b""])

    async def _readline():
        return next(lines_iter, b"")

    proc.stdout = MagicMock()
    proc.stdout.readline = _readline

    async def _stderr_read():
        return b""

    proc.stderr = MagicMock()
    proc.stderr.read = _stderr_read

    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()

    return proc


# ---------------------------------------------------------------------------
# ACPTextGen — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acp_text_gen_returns_text():
    """ACPTextGen.chat() returns the agent's text output."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "Hello "}}),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "world"}}),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        from specq.providers.acp_agent import ACPSubprocessAgent
        agent = ACPSubprocessAgent(cmd=["fake-cli"], auto_approve_permissions=False)
        text_gen = ACPTextGen(agent)
        result = await text_gen.chat("You are helpful.", "Say hello")

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_acp_text_gen_passes_system_prompt():
    """System prompt is forwarded as a system role message via ACP."""
    written: list[dict] = []

    proc = _make_proc([_init_response(), _rpc("agents/done")])

    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        from specq.providers.acp_agent import ACPSubprocessAgent
        agent = ACPSubprocessAgent(cmd=["fake-cli"], auto_approve_permissions=False)
        text_gen = ACPTextGen(agent)
        await text_gen.chat("Be concise.", "Summarize this")

    run_msgs = [m for m in written if m.get("method") == "agents/run"]
    assert len(run_msgs) == 1
    input_msgs = run_msgs[0]["params"]["input"]
    system_msg = next((m for m in input_msgs if m["role"] == "system"), None)
    assert system_msg is not None
    assert "Be concise." in system_msg["content"][0]["text"]


# ---------------------------------------------------------------------------
# ACPTextGen — error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acp_text_gen_raises_on_failure():
    """ACPTextGen.chat() raises RuntimeError when the agent fails."""
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError())):
        from specq.providers.acp_agent import ACPSubprocessAgent
        agent = ACPSubprocessAgent(cmd=["nonexistent-cli"], auto_approve_permissions=False)
        text_gen = ACPTextGen(agent)

        with pytest.raises(RuntimeError, match="ACP text generation failed"):
            await text_gen.chat("sys", "usr")


@pytest.mark.asyncio
async def test_acp_text_gen_no_permissions_granted():
    """ACPTextGen does NOT auto-approve tool permissions."""
    written: list[dict] = []

    proc = _make_proc([
        _init_response(),
        _rpc("permissions/requested", {"permissionsRequestId": "perm-1"}),
        _rpc("agents/done"),
    ])

    def _capture_write(data: bytes):
        written.append(json.loads(data.decode().strip()))

    proc.stdin.write = _capture_write

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        from specq.providers.acp_agent import ACPSubprocessAgent
        agent = ACPSubprocessAgent(cmd=["fake-cli"], auto_approve_permissions=False)
        text_gen = ACPTextGen(agent)
        await text_gen.chat("sys", "usr")

    granted = [m for m in written if m.get("method") == "permissions/granted"]
    assert len(granted) == 0


# ---------------------------------------------------------------------------
# GeminiCLITextGen
# ---------------------------------------------------------------------------

def test_gemini_cli_text_gen_uses_correct_cmd():
    """GeminiCLITextGen wraps GeminiCLIAgent with correct CLI flags."""
    tg = GeminiCLITextGen(model="gemini-2.5-pro")
    assert tg._agent._cmd == ["gemini", "--experimental-acp", "--model", "gemini-2.5-pro"]
    assert tg._agent.auto_approve_permissions is False


def test_gemini_cli_text_gen_no_model():
    """GeminiCLITextGen without model uses CLI default."""
    tg = GeminiCLITextGen()
    assert tg._agent._cmd == ["gemini", "--experimental-acp"]


@pytest.mark.asyncio
async def test_gemini_cli_text_gen_chat():
    """GeminiCLITextGen.chat() returns text from Gemini CLI."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "Gemini says hi"}}),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        tg = GeminiCLITextGen(model="gemini-2.5-pro")
        result = await tg.chat("system", "user prompt")

    assert result == "Gemini says hi"


# ---------------------------------------------------------------------------
# CodexTextGen
# ---------------------------------------------------------------------------

def test_codex_text_gen_uses_correct_cmd():
    """CodexTextGen wraps CodexAgent with correct CLI flags."""
    tg = CodexTextGen(model="gpt-5.3")
    assert tg._agent._cmd == ["codex", "--mode", "acp", "--model", "gpt-5.3"]
    assert tg._agent.auto_approve_permissions is False


def test_codex_text_gen_no_model():
    """CodexTextGen without model uses CLI default."""
    tg = CodexTextGen()
    assert tg._agent._cmd == ["codex", "--mode", "acp"]


@pytest.mark.asyncio
async def test_codex_text_gen_chat():
    """CodexTextGen.chat() returns text from Codex CLI."""
    stdout = [
        _init_response(),
        _rpc("agents/textDelta", {"delta": {"type": "text", "text": "Codex says hi"}}),
        _rpc("agents/done"),
    ]
    proc = _make_proc(stdout)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        tg = CodexTextGen(model="gpt-5.3")
        result = await tg.chat("system", "user prompt")

    assert result == "Codex says hi"


# ---------------------------------------------------------------------------
# Pipeline factory integration — _make_text_gen
# ---------------------------------------------------------------------------

def test_make_text_gen_gemini_cli(tmp_path):
    """_make_text_gen('gemini_cli') returns GeminiCLITextGen."""
    from specq.config import load_config
    from specq.pipeline import _make_text_gen

    (tmp_path / ".specq").mkdir()
    (tmp_path / ".specq" / "config.yaml").write_text("verification:\n  voters: []\n")
    config = load_config(tmp_path)

    tg = _make_text_gen(config, "gemini_cli", "gemini-2.5-pro")
    assert isinstance(tg, GeminiCLITextGen)


def test_make_text_gen_codex(tmp_path):
    """_make_text_gen('codex') returns CodexTextGen."""
    from specq.config import load_config
    from specq.pipeline import _make_text_gen

    (tmp_path / ".specq").mkdir()
    (tmp_path / ".specq" / "config.yaml").write_text("verification:\n  voters: []\n")
    config = load_config(tmp_path)

    tg = _make_text_gen(config, "codex", "gpt-5.3")
    assert isinstance(tg, CodexTextGen)
