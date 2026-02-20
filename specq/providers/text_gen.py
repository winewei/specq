"""TextGen: single-turn text generation — HTTP providers and Claude Code SDK."""

from __future__ import annotations

import anyio
import httpx

ENDPOINTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    # OpenAI-compatible providers
    "glm": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 500, 502, 503, 529}


class HttpTextGen:
    """HTTP-based text generation: Anthropic, OpenAI-compatible, Google."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    async def chat(self, system: str, user: str) -> str:
        """Send prompt, return text response."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system, user)
        elif self.provider == "google":
            return await self._call_google(system, user)
        elif self.provider in ENDPOINTS:
            return await self._call_openai_compat(system, user, ENDPOINTS[self.provider])
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _call_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Run one request with exponential-backoff retry. Creates a fresh
        httpx.AsyncClient per call so there is no shared state to clean up."""
        last_exc = None
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    resp = await client.request(method, url, **kwargs)
                    if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                        await resp.aclose()  # Release connection before sleeping
                        await anyio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    return resp
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        await anyio.sleep(2 ** attempt)
                        continue
                    raise
        raise last_exc  # type: ignore[misc]

    async def _call_anthropic(self, system: str, user: str) -> str:
        resp = await self._call_with_retry(
            "POST",
            ENDPOINTS["anthropic"],
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        return resp.json()["content"][0]["text"]

    async def _call_openai_compat(self, system: str, user: str, url: str) -> str:
        resp = await self._call_with_retry(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return resp.json()["choices"][0]["message"]["content"]

    async def _call_google(self, system: str, user: str) -> str:
        url = ENDPOINTS["google"].format(model=self.model)
        resp = await self._call_with_retry(
            "POST",
            f"{url}?key={self.api_key}",
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
            },
        )
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class ClaudeCodeTextGen:
    """Text generation via local Claude Code CLI auth — no API key needed.

    Uses claude_code_sdk.query() with max_turns=1 and no tools, so it behaves
    as a single-turn text generator authenticated via ``claude login``.
    """

    def __init__(self, model: str):
        self.model = model

    async def chat(self, system: str, user: str) -> str:
        from claude_code_sdk import ClaudeCodeOptions, Message, query

        options = ClaudeCodeOptions(
            model=self.model,
            max_turns=1,
            allowed_tools=[],
            system_prompt=system,
        )
        parts: list[str] = []
        async for message in query(prompt=user, options=options):
            if isinstance(message, Message):
                for block in message.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
        return "".join(parts)
