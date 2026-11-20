"""Two local processes share ONE ChatGPT session (the API process and the ``services.ai.chat`` CLI): the
refresh token in the Keychain rotates on every refresh, so a process that refreshed with a stale cached token
would trip reuse detection (``refresh_token_reused``) and sign BOTH out. Each refresh therefore re-reads the
store under an exclusive cross-process file lock. Simulated here with two ``AiRuntime`` objects over one shared
credential store, one keys dir and one fake auth server (reuse detection enabled)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from services.ai.chatgpt_oauth import ChatGptError, cross_process_lock
from services.ai.runtime_modes import AiRuntime
from services.ai.token_store import ACCOUNT_REFRESH_TOKEN

pytestmark = pytest.mark.timeout(60)


def _second_process(h: Any) -> AiRuntime:
    """Another 'process': its own ChatGptAuth (own memory), same Keychain store, same keys dir, same server."""
    return AiRuntime(h.settings, h.secrets, clock=h.clock, wall_clock=h.clock, token_store=h.store,
                     fake_server=h.server, oauth_config=h.server.config, auto_complete_fake_login=False)


async def test_rotation_by_another_process_is_picked_up_not_reused(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    other = _second_process(h)
    try:
        first_rt = h.store.get(ACCOUNT_REFRESH_TOKEN)
        assert await other.auth.access_token()  # the chat CLI refreshes: the stored refresh token rotates
        rotated = h.store.get(ACCOUNT_REFRESH_TOKEN)
        assert rotated and rotated != first_rt
        clock.advance(h.server.access_ttl_s + 1)  # the API process' access token expires
        assert await h.rt.auth.access_token()  # refreshes with the ROTATED token read from the store
        assert h.rt.auth.signed_in() and other.auth.signed_in()
        assert h.store.get(ACCOUNT_REFRESH_TOKEN) not in (first_rt, rotated)
        assert not [r for r in h.server.recorded if r.status == 400 and r.path.endswith("/oauth/token")]
    finally:
        await other.aclose()


class _SlowTokenEndpoint:
    """Async MockTransport handler that yields to the event loop before answering a token request (as real
    network I/O does), so two unserialized refreshes would interleave and reuse the same refresh token."""

    def __init__(self) -> None:
        self.server: Any = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            await asyncio.sleep(0.05)
        return self.server(request)


async def test_concurrent_refreshes_in_two_processes_are_serialized(chatgpt_factory: Any, clock: Any) -> None:
    slow = _SlowTokenEndpoint()
    h = chatgpt_factory(http_client=httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    slow.server = h.server
    await h.sign_in()
    other = _second_process(h)
    try:
        assert other.auth.signed_in()  # both processes have read (and cached) the same refresh token
        clock.advance(h.server.access_ttl_s + 1)
        await asyncio.gather(h.rt.auth.refresh(), other.auth.refresh())
        tokens = [r for r in h.server.recorded if r.path.endswith("/oauth/token")]
        assert all(r.status == 200 for r in tokens), [r.status for r in tokens]
        assert h.rt.auth.signed_in() and other.auth.signed_in()
    finally:
        await other.aclose()


async def test_logout_in_one_process_ends_the_shared_session(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    other = _second_process(h)
    try:
        res = await other.chatgpt_logout()
        assert res["revoked"] is True
        assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None
        h.rt.auth.invalidate_access_token()
        with pytest.raises(ChatGptError) as err:
            await h.rt.auth.access_token()  # re-reads the store: nothing to refresh with -> sign in again
        assert err.value.kind == "NOT_SIGNED_IN"
    finally:
        await other.aclose()


async def test_cross_process_lock_times_out_fail_closed(tmp_path: Any) -> None:
    path = tmp_path / "chatgpt_refresh.lock"
    async with cross_process_lock(path):
        with pytest.raises(ChatGptError) as err:
            async with cross_process_lock(path, timeout_s=0.2):
                pass
        assert err.value.kind == "TRANSIENT"
    async with cross_process_lock(path, timeout_s=0.2):  # released again
        pass
