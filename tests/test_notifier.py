"""Tests for webhook notifications."""

import json
import httpx as httpx_lib
import pytest
from specq.notifier import Notifier
from specq.models import WorkItem, Status


@pytest.mark.asyncio
async def test_sends_webhook(httpx_mock):
    """Sends correct webhook POST."""
    httpx_mock.add_response(status_code=200)

    notifier = Notifier(webhook_url="https://hook.example.com/cb",
                        events=["change.completed"])
    wi = WorkItem(id="001", change_dir="c/001", title="Auth",
                  description="", status=Status.ACCEPTED)
    await notifier.notify("change.completed", wi)

    req = httpx_mock.get_request()
    body = json.loads(req.content)
    assert body["event"] == "change.completed"
    assert body["change_id"] == "001"


@pytest.mark.asyncio
async def test_filters_unsubscribed_events():
    """Events not in list → no request sent (no httpx_mock needed since no request)."""
    notifier = Notifier(webhook_url="https://hook.example.com/cb",
                        events=["change.failed"])
    wi = WorkItem(id="001", change_dir="c/001", title="", description="")
    await notifier.notify("change.completed", wi)
    # No crash, no request sent


@pytest.mark.asyncio
async def test_no_webhook_noop():
    """No webhook configured → silent noop."""
    notifier = Notifier(webhook_url="", events=["change.completed"])
    wi = WorkItem(id="001", change_dir="c/001", title="", description="")
    await notifier.notify("change.completed", wi)  # no crash


@pytest.mark.asyncio
async def test_webhook_failure_silent(httpx_mock):
    """Webhook failure → no crash."""
    httpx_mock.add_exception(httpx_lib.ConnectError("unreachable"))
    notifier = Notifier(webhook_url="https://hook.example.com/cb",
                        events=["change.failed"])
    wi = WorkItem(id="001", change_dir="c/001", title="", description="")
    await notifier.notify("change.failed", wi)  # no exception
