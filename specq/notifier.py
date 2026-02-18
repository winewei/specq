"""Webhook notifications."""

from __future__ import annotations

import httpx

from .models import WorkItem


class Notifier:
    """Send webhook notifications for pipeline events."""

    def __init__(self, webhook_url: str = "", events: list[str] | None = None):
        self.webhook_url = webhook_url
        self.events = events or []
        self.client = httpx.AsyncClient()

    async def notify(self, event: str, work_item: WorkItem) -> None:
        if not self.webhook_url or event not in self.events:
            return

        payload = {
            "event": event,
            "change_id": work_item.id,
            "title": work_item.title,
            "status": work_item.status.value if hasattr(work_item.status, "value") else str(work_item.status),
            "retry_count": work_item.retry_count,
        }

        try:
            await self.client.post(self.webhook_url, json=payload, timeout=10)
        except Exception:
            pass  # Notification failure should not affect pipeline

    async def close(self) -> None:
        await self.client.aclose()
