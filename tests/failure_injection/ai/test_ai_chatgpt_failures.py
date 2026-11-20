"""Failure injection for the Sign-in-with-ChatGPT runtime: network partitions, credential-store failures, garbage
on the loopback listener, malformed streams, cancellation mid-turn. Every case fails closed (NO_TRADE or a clear
sign-in error) and nothing is retried in a loop."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

from packages.contracts.enums import DecisionAction, ModelRunStatus
from services.ai.chatgpt_fake import fake_browser_sign_in
from services.ai.chatgpt_responses import ChatGptConfig
from services.ai.token_store import ACCOUNT_CLIENT_ID, ACCOUNT_REFRESH_TOKEN, MemoryTokenStore, TokenStoreError

pytestmark = pytest.mark.timeout(60)


class _Switch:
    def __init__(self) -> None:
        self.server: Any = None
        self.cut_api = False
        self.cut_discovery = False
        self.garbage_stream = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.cut_discovery and request.url.path.endswith("openid-configuration"):
            raise httpx.ConnectError("partition", request=request)
        if self.cut_api and request.url.path.endswith("/responses"):
            raise httpx.ConnectError("partition", request=request)
        if self.garbage_stream and request.url.path.endswith("/responses"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=b"event: response.created\ndata: {not json\n\n")
        return self.server(request)


def _partitioned(chatgpt_factory: Any, **kw: Any) -> tuple[Any, _Switch]:
    sw = _Switch()
    h = chatgpt_factory(http_client=httpx.AsyncClient(transport=httpx.MockTransport(sw), follow_redirects=False), **kw)
    sw.server = h.server
    return h, sw


async def test_network_partition_mid_research_backs_off_and_keeps_credentials(chatgpt_factory: Any) -> None:
    h, sw = _partitioned(chatgpt_factory)
    await h.sign_in()
    sw.cut_api = True
    r = await h.research()
    assert r.proposal.action == DecisionAction.NO_TRADE
    assert r.model_runs[0].status == ModelRunStatus.ERROR and "STREAM_INTERRUPTED" in (r.model_runs[0].error or "")
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is not None and h.rt.auth.signed_in()  # never erased on a blip
    sw.cut_api = False
    again = await h.research(causal_id="cz_2")
    assert again.model_runs[0].status == ModelRunStatus.SKIPPED  # bounded backoff, no tight retry
    assert "backing off" in (again.proposal.no_trade_reason or "")


async def test_malformed_stream_is_schema_safe(chatgpt_factory: Any) -> None:
    h, sw = _partitioned(chatgpt_factory)
    await h.sign_in()
    sw.garbage_stream = True
    r = await h.research()
    assert r.proposal.action == DecisionAction.NO_TRADE and "BAD_RESPONSE" in (r.model_runs[0].error or "")


async def test_discovery_unreachable_uses_documented_endpoints_and_cannot_revoke(chatgpt_factory: Any) -> None:
    h, sw = _partitioned(chatgpt_factory)
    sw.cut_discovery = True
    await h.sign_in()  # documented fallback endpoints (the fake config's)
    out = await h.rt.chatgpt_logout()
    assert out["revoked"] is False and "revocation_endpoint" in out["detail"]
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None
    assert "not confirmed" in (h.rt.status()["chatgpt"]["last_error"] or "")


class _FailingStore(MemoryTokenStore):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on

    def set(self, account: str, value: str) -> None:
        if account == self.fail_on:
            raise TokenStoreError("Keychain write failed (simulated)")
        super().set(account, value)


async def test_keychain_failure_for_client_id_aborts_sign_in(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(store=_FailingStore(ACCOUNT_CLIENT_ID))
    info = await h.rt.chatgpt_login()
    await fake_browser_sign_in(h.server, info["authorize_url"])
    assert "could not save the issued client id" in (await h.rt.auth.wait_login(5) or "")
    assert h.rt.status()["chatgpt"]["signed_in"] is False


async def test_keychain_failure_for_refresh_token_is_reported(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(store=_FailingStore(ACCOUNT_REFRESH_TOKEN))
    await h.sign_in()
    st = h.rt.status()["chatgpt"]
    assert st["signed_in"] is True and "memory only" in (st["last_error"] or "")  # honest: lost on restart
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None


async def test_token_endpoint_outage_during_sign_in(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    h.server.token_status_override = 500
    info = await h.rt.chatgpt_login()
    await fake_browser_sign_in(h.server, info["authorize_url"])
    err = await h.rt.auth.wait_login(5) or ""
    assert "code exchange" in err and h.store.get(ACCOUNT_REFRESH_TOKEN) is None


async def test_listener_survives_garbage_and_slow_clients(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    info = await h.rt.chatgpt_login()
    port = urlsplit(info["callback"]).port
    assert port is not None
    for payload in (b"\x00\xff garbage\r\n\r\n", b"POST /auth/callback HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
                    b"GET /auth/callback HTTP/1.1\r\n" + b"X-Pad: " + b"a" * 70_000 + b"\r\n\r\n"):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(payload)
        await writer.drain()
        head = await asyncio.wait_for(reader.read(64), timeout=15)
        writer.close()
        assert head.startswith(b"HTTP/1.1 4"), head
    assert h.rt.auth.login_pending
    ok = await fake_browser_sign_in(h.server, info["authorize_url"])
    assert ok.status_code == 200


async def test_cancellation_mid_turn_releases_the_gate(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(config=ChatGptConfig())
    await h.sign_in()
    gate_release = asyncio.Event()
    original = h.server._responses  # pyright: ignore[reportPrivateUsage]

    def slow(request: httpx.Request, body: Any) -> httpx.Response:
        gate_release.set()
        raise httpx.ReadTimeout("never answers", request=request)

    h.server._responses = slow  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(h.research())
    await gate_release.wait()
    r = await task
    assert r.model_runs[0].status == ModelRunStatus.TIMEOUT
    assert h.rt.researcher.gate.status()["in_flight"] == []
    h.server._responses = original  # pyright: ignore[reportPrivateUsage]
    task2 = asyncio.create_task(h.research(causal_id="cz_c2"))
    await asyncio.sleep(0)
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2
    assert h.rt.researcher.gate.status()["in_flight"] == []  # settled in finally even when cancelled
