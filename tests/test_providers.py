"""Tests for LLM provider API wrappers."""

import json
import httpx
import pytest
from specq.providers import LLMProvider


# --- API request format ---

@pytest.mark.asyncio
async def test_anthropic_request_format(httpx_mock):
    """Anthropic: correct headers + body."""
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "response"}]})
    p = LLMProvider("anthropic", "claude-haiku-4-5", "sk-test")
    result = await p.chat("system prompt", "user prompt")

    assert result == "response"
    req = httpx_mock.get_request()
    assert req.headers["x-api-key"] == "sk-test"
    assert req.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(req.content)
    assert body["model"] == "claude-haiku-4-5"
    assert body["system"] == "system prompt"
    assert body["messages"][0]["content"] == "user prompt"


@pytest.mark.asyncio
async def test_openai_request_format(httpx_mock):
    """OpenAI: Bearer token + system/user messages."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": "resp"}}]})
    p = LLMProvider("openai", "gpt-4o", "sk-oai")
    result = await p.chat("sys", "usr")

    assert result == "resp"
    req = httpx_mock.get_request()
    assert "Bearer sk-oai" in req.headers["authorization"]
    body = json.loads(req.content)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_google_request_format(httpx_mock):
    """Google: URL contains model + api key."""
    httpx_mock.add_response(json={"candidates": [{"content": {"parts": [{"text": "resp"}]}}]})
    p = LLMProvider("google", "gemini-2.5-flash", "AIza-key")
    result = await p.chat("sys", "usr")

    assert result == "resp"
    req = httpx_mock.get_request()
    assert "gemini-2.5-flash" in str(req.url)
    assert "AIza-key" in str(req.url)


@pytest.mark.asyncio
async def test_glm_request_format(httpx_mock):
    """GLM: hits bigmodel.cn endpoint with Bearer token, OpenAI-compatible format."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": "glm resp"}}]})
    p = LLMProvider("glm", "glm-4-air", "glm-key")
    result = await p.chat("sys", "usr")

    assert result == "glm resp"
    req = httpx_mock.get_request()
    assert "bigmodel.cn" in str(req.url)
    assert "Bearer glm-key" in req.headers["authorization"]
    body = json.loads(req.content)
    assert body["model"] == "glm-4-air"
    assert body["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_deepseek_request_format(httpx_mock):
    """DeepSeek: hits api.deepseek.com with Bearer token, OpenAI-compatible format."""
    httpx_mock.add_response(json={"choices": [{"message": {"content": "ds resp"}}]})
    p = LLMProvider("deepseek", "deepseek-chat", "ds-key")
    result = await p.chat("sys", "usr")

    assert result == "ds resp"
    req = httpx_mock.get_request()
    assert "deepseek.com" in str(req.url)
    assert "Bearer ds-key" in req.headers["authorization"]


def test_glm_api_key_from_env(tmp_path, monkeypatch):
    """GLM_API_KEY env var is picked up by load_config."""
    from specq.config import load_config
    (tmp_path / ".specq").mkdir()
    (tmp_path / ".specq" / "config.yaml").write_text("verification:\n  voters: []\n")
    monkeypatch.setenv("GLM_API_KEY", "glm-env-key")
    cfg = load_config(tmp_path)
    assert cfg.providers.glm.api_key == "glm-env-key"


def test_deepseek_api_key_from_env(tmp_path, monkeypatch):
    """DEEPSEEK_API_KEY env var is picked up by load_config."""
    from specq.config import load_config
    (tmp_path / ".specq").mkdir()
    (tmp_path / ".specq" / "config.yaml").write_text("verification:\n  voters: []\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-env-key")
    cfg = load_config(tmp_path)
    assert cfg.providers.deepseek.api_key == "ds-env-key"


# --- Error handling ---

@pytest.mark.asyncio
async def test_unknown_provider_raises():
    """Unknown provider → ValueError."""
    p = LLMProvider("azure", "model", "key")
    with pytest.raises(ValueError, match="[Uu]nknown"):
        await p.chat("sys", "usr")


@pytest.mark.asyncio
async def test_retry_on_timeout(httpx_mock):
    """Timeout retry: first 2 timeout, 3rd succeeds."""
    httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
    httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "ok"}]})

    p = LLMProvider("anthropic", "claude-haiku-4-5", "key")
    result = await p.chat("sys", "usr")
    assert result == "ok"


@pytest.mark.asyncio
async def test_retry_exhausted_raises(httpx_mock):
    """All retries exhausted → raise. MAX_RETRIES=3 → 4 total attempts."""
    for _ in range(4):
        httpx_mock.add_exception(httpx.ReadTimeout("timeout"))

    p = LLMProvider("anthropic", "claude-haiku-4-5", "key")
    with pytest.raises(Exception):
        await p.chat("sys", "usr")


@pytest.mark.asyncio
async def test_429_rate_limit_retry(httpx_mock):
    """429 triggers retry."""
    httpx_mock.add_response(status_code=429, json={"error": "rate limited"})
    httpx_mock.add_response(json={"content": [{"type": "text", "text": "ok"}]})

    p = LLMProvider("anthropic", "claude-haiku-4-5", "key")
    result = await p.chat("sys", "usr")
    assert result == "ok"


@pytest.mark.asyncio
async def test_500_raises_after_retry(httpx_mock):
    """500 after all retries → raise. MAX_RETRIES=3 → 4 total attempts."""
    for _ in range(4):
        httpx_mock.add_response(status_code=500, json={"error": "internal"})

    p = LLMProvider("anthropic", "claude-haiku-4-5", "key")
    with pytest.raises(Exception):
        await p.chat("sys", "usr")
