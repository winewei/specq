"""LLM provider abstraction — unified httpx-based API calls."""

from __future__ import annotations

import json

import anyio
import httpx

ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}

_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 500, 502, 503, 529}


class LLMProvider:
    """Unified LLM API wrapper using httpx."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120)

    async def chat(self, system: str, user: str) -> str:
        """Send prompt, return text response."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system, user)
        elif self.provider == "openai":
            return await self._call_openai(system, user)
        elif self.provider == "google":
            return await self._call_google(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def close(self) -> None:
        await self.client.aclose()

    async def _call_with_retry(self, method, url, **kwargs) -> httpx.Response:
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self.client.request(method, url, **kwargs)
                if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
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

    async def _call_openai(self, system: str, user: str) -> str:
        resp = await self._call_with_retry(
            "POST",
            ENDPOINTS["openai"],
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
