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
