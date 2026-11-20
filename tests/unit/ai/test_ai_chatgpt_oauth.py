"""Sign in with ChatGPT: authorize request, loopback callback, code exchange, ID-token checks, scope gate, refresh
and logout -- all against the explicit FakeChatGptServer (no network; the loopback listener is a real 127.0.0.1
socket)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from services.ai.chatgpt_fake import fake_browser_sign_in
from services.ai.chatgpt_oauth import (
    DYNAMIC_CLIENT_ID,
    PLAN_USAGE_SCOPE,
    SCOPES,
    ChatGptError,
    ChatGptOAuthConfig,
    mask_email,
    pkce_pair,
)
from services.ai.token_store import ACCOUNT_CLIENT_ID, ACCOUNT_REFRESH_TOKEN

pytestmark = pytest.mark.timeout(60)


def _q(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


async def _get(url: str, **headers: str) -> httpx.Response:
    async with httpx.AsyncClient(trust_env=False, timeout=5.0) as c:
        return await c.get(url, headers=headers or None)


def test_pkce_and_masking() -> None:
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert pkce_pair()[0] != verifier
    assert mask_email("trader@example.com") == "t***@e***.com" and mask_email(None) is None


async def test_first_registration_authorize_request(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    info = await h.rt.chatgpt_login()
    q = _q(info["authorize_url"])
    parts = urlsplit(info["authorize_url"])
    assert f"{parts.scheme}://{parts.hostname}{parts.path}" == h.server.config.authorization_endpoint
    assert q["client_id"] == DYNAMIC_CLIENT_ID and q["agent_name_hint"] == "local-ai-trader"
    assert q["response_type"] == "code" and q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
    assert q["scope"].split() == list(SCOPES) and PLAN_USAGE_SCOPE in q["scope"]
    assert q["resource"] == "https://api.openai.com/v1"
    assert q["ext_agent_host_id"].startswith("urn:uuid:") and len(q["state"]) >= 32 and len(q["nonce"]) >= 32
    redirect = urlsplit(q["redirect_uri"])
    assert (redirect.scheme, redirect.hostname, redirect.path) == ("http", "127.0.0.1", "/auth/callback")
    assert q["redirect_uri"] == info["callback"] and "localhost" not in info["authorize_url"]
    assert "id_token_hint" not in q and "code_verifier" not in q  # the verifier never leaves the process
    # the listener is already accepting connections when the URL is handed out
    resp = await _get(f"http://127.0.0.1:{redirect.port}/not-the-callback", Host=f"127.0.0.1:{redirect.port}")
    assert resp.status_code == 404
    # a fresh state/nonce/verifier per attempt
    again = _q((await h.rt.chatgpt_login())["authorize_url"])
    assert again["state"] != q["state"] and again["nonce"] != q["nonce"]
    assert again["code_challenge"] != q["code_challenge"]


async def test_full_sign_in_stores_refresh_token_and_client_id_only(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    st = h.rt.status()["chatgpt"]
    assert st["signed_in"] and st["plan_usage_granted"] and st["email_masked"] == "t***@e***.com"
    assert st["transport"] == "FAKE" and st["credential_store"] == "memory" and st["login_pending"] is False
    client_id = h.store.get(ACCOUNT_CLIENT_ID)
    assert client_id and client_id.startswith("oaiapp_") and client_id in h.server.issued_clients
    grant = h.server.grants[client_id]
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) == grant.refresh_token
    # the access token is kept in memory only: never in the store, never in the metadata file
    stored = {h.store.get(ACCOUNT_CLIENT_ID), h.store.get(ACCOUNT_REFRESH_TOKEN)}
    assert grant.access_token not in stored
    meta = (h.settings.keys_dir / "chatgpt_account.fake.json").read_text()
    assert grant.access_token not in meta and grant.refresh_token not in meta and "trader@example.com" not in meta
    assert oct((h.settings.keys_dir / "chatgpt_account.fake.json").stat().st_mode & 0o777) == "0o600"
    host = (h.settings.keys_dir / "chatgpt_host_id").read_text().strip()
    assert host.startswith("urn:uuid:")
    # code exchange was form-encoded with PKCE and the issued client id
    token_calls = [r for r in h.server.recorded if r.path.endswith("/oauth/token")]
    assert token_calls[0].body["grant_type"] == "authorization_code"
    assert token_calls[0].body["client_id"] == client_id and token_calls[0].body["code_verifier"]
    assert token_calls[0].body["resource"] == "https://api.openai.com/v1"
    assert h.server.violations == []


async def test_reauthorization_uses_issued_client_without_name_hint(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    client_id = h.store.get(ACCOUNT_CLIENT_ID)
    info = await h.rt.chatgpt_login()
    q = _q(info["authorize_url"])
    assert q["client_id"] == client_id and "agent_name_hint" not in q and q.get("id_token_hint")
    host_first = q["ext_agent_host_id"]
    resp = await fake_browser_sign_in(h.server, info["authorize_url"])
    assert resp.status_code == 200 and await h.rt.auth.wait_login(5) is None
    assert _q((await h.rt.chatgpt_login())["authorize_url"])["ext_agent_host_id"] == host_first  # stable per host
    assert h.server.violations == []


async def test_forged_state_is_ignored_and_listener_keeps_waiting(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    info = await h.rt.chatgpt_login()
    port = urlsplit(info["callback"]).port
    forged = await _get(f"{info['callback']}?code=evil&state=forged", Host=f"127.0.0.1:{port}")
    assert forged.status_code == 400 and "State mismatch" in forged.text
    wrong_host = await _get(f"{info['callback']}?code=x&state=y", Host="evil.example:80")
    assert wrong_host.status_code == 400
    dup = await _get(f"{info['callback']}?state=a&state=b", Host=f"127.0.0.1:{port}")
    assert dup.status_code == 400
    assert h.rt.auth.login_pending  # still waiting for the genuine callback
    assert not [r for r in h.server.recorded if r.path.endswith("/oauth/token")]  # nothing was exchanged
    ok = await fake_browser_sign_in(h.server, info["authorize_url"])
    assert ok.status_code == 200 and "<script" not in ok.text
    assert not h.rt.auth.login_pending
    # single use: the listener is closed after the successful callback
    with pytest.raises(httpx.ConnectError):
        await _get(f"{info['callback']}?code=again&state=x", Host=f"127.0.0.1:{port}")


async def test_access_denied_exchanges_nothing(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    h.server.deny_authorization = True
    info = await h.rt.chatgpt_login()
    resp = await fake_browser_sign_in(h.server, info["authorize_url"])
    assert resp.status_code == 400
    assert "access_denied" in (await h.rt.auth.wait_login(5) or "")
    assert not [r for r in h.server.recorded if r.path.endswith("/oauth/token")]
    assert h.rt.status()["chatgpt"]["signed_in"] is False


async def test_registration_without_issued_client_id_is_incomplete(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    h.server.omit_callback_client_id = True
    info = await h.rt.chatgpt_login()
    await fake_browser_sign_in(h.server, info["authorize_url"])
    assert "registration incomplete" in (await h.rt.auth.wait_login(5) or "")
    assert h.store.get(ACCOUNT_CLIENT_ID) is None and h.store.get(ACCOUNT_REFRESH_TOKEN) is None


async def test_reauthorization_with_different_client_id_is_rejected(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    saved = h.store.get(ACCOUNT_CLIENT_ID)
    info = await h.rt.chatgpt_login()
    q = _q(info["authorize_url"])
    port = urlsplit(info["callback"]).port
    resp = await _get(f"{info['callback']}?code=c&state={q['state']}&client_id=oaiapp_someoneelse",
                      Host=f"127.0.0.1:{port}")
    assert resp.status_code == 400 and "different client id" in resp.text
    assert h.store.get(ACCOUNT_CLIENT_ID) == saved


@pytest.mark.parametrize("knob", ["tamper_id_token", "wrong_nonce"])
async def test_invalid_id_token_never_signs_in(chatgpt_factory: Any, knob: str) -> None:
    h = chatgpt_factory()
    setattr(h.server, knob, True)
    info = await h.rt.chatgpt_login()
    await fake_browser_sign_in(h.server, info["authorize_url"])
    err = await h.rt.auth.wait_login(5) or ""
    assert "ID token rejected" in err
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None and h.rt.status()["chatgpt"]["signed_in"] is False


async def test_identity_only_sign_in_is_never_used_for_inference(chatgpt_factory: Any) -> None:
    h = chatgpt_factory(grant_plan_usage=False)
    await h.sign_in()
    st = h.rt.status()["chatgpt"]
    assert st["signed_in"] is True and st["plan_usage_granted"] is False
    with pytest.raises(ChatGptError) as exc:
        await h.rt.auth.access_token()
    assert exc.value.kind == "PLAN_USAGE_NOT_GRANTED"
    result = await h.research()
    assert result.proposal.action.value == "NO_TRADE" and PLAN_USAGE_SCOPE in (result.proposal.no_trade_reason or "")
    assert h.server.responses_bodies == [] and not [r for r in h.server.recorded if r.path.endswith("/models")]


@pytest.mark.parametrize("override", [
    {"issuer": "https://auth.evil.example"},
    {"token_endpoint": "https://auth.evil.example/token"},
    {"authorization_endpoint": "http://auth.openai-fake.invalid/api/accounts/authorize"},
    {"revocation_endpoint": "https://evil.example/revoke"},
])
async def test_discovery_document_is_validated(chatgpt_factory: Any, override: dict[str, str]) -> None:
    h = chatgpt_factory()
    h.server.discovery_override = override
    with pytest.raises(ChatGptError) as exc:
        await h.rt.chatgpt_login()
    assert exc.value.kind == "DISCOVERY_INVALID"
    assert not h.rt.auth.login_pending


async def test_refresh_rotates_and_omits_scope(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    client_id = h.store.get(ACCOUNT_CLIENT_ID)
    first_refresh = h.store.get(ACCOUNT_REFRESH_TOKEN)
    first_access = await h.rt.auth.access_token()
    assert await h.rt.auth.access_token() == first_access  # still fresh: no refresh
    clock.advance(minutes=59)  # inside the refresh margin
    second_access = await h.rt.auth.access_token()
    assert second_access != first_access
    refresh_calls = [r.body for r in h.server.recorded if r.path.endswith("/oauth/token")
                     and r.body.get("grant_type") == "refresh_token"]
    assert len(refresh_calls) == 1
    assert set(refresh_calls[0]) == {"grant_type", "client_id", "refresh_token", "resource"}  # scope omitted
    assert refresh_calls[0]["client_id"] == client_id and refresh_calls[0]["refresh_token"] == first_refresh
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) != first_refresh  # rotated and replaced together
    assert h.server.violations == []


async def test_concurrent_refreshes_are_serialized(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    clock.advance(hours=2)  # expired
    tokens = await asyncio.gather(*(h.rt.auth.access_token() for _ in range(10)))
    assert len(set(tokens)) == 1
    refreshes = [r for r in h.server.recorded if r.path.endswith("/oauth/token")
                 and r.body.get("grant_type") == "refresh_token"]
    assert len(refreshes) == 1  # no race on the rotating refresh token


async def test_earliest_refresh_at_is_respected(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    acc = h.rt.auth._access  # pyright: ignore[reportPrivateUsage]
    assert acc is not None
    acc.earliest_refresh_at = clock.now() + timedelta(minutes=59, seconds=30)
    clock.advance(minutes=59)  # near expiry but before earliest_refresh_at: keep the token
    assert await h.rt.auth.access_token() == acc.token
    clock.advance(seconds=40)
    assert await h.rt.auth.access_token() != acc.token


@pytest.mark.parametrize("code", ["invalid_grant", "refresh_token_reused", "refresh_token_expired",
                                  "refresh_token_invalidated", "token_expired", "invalid_refresh_token"])
async def test_terminal_refresh_errors_clear_tokens(chatgpt_factory: Any, clock: Any, code: str) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    client_id = h.store.get(ACCOUNT_CLIENT_ID)
    h.server.refresh_error = code
    clock.advance(hours=2)
    with pytest.raises(ChatGptError) as exc:
        await h.rt.auth.access_token()
    assert exc.value.kind == "SIGN_IN_REQUIRED" and exc.value.code == code
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None
    assert h.store.get(ACCOUNT_CLIENT_ID) == client_id  # kept for the next sign-in
    st = h.rt.status()["chatgpt"]
    assert st["signed_in"] is False and "sign in again" in (st["last_error"] or "")


async def test_transient_refresh_failure_keeps_credentials(chatgpt_factory: Any, clock: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    rt_before = h.store.get(ACCOUNT_REFRESH_TOKEN)
    h.server.token_status_override = 503
    clock.advance(hours=2)
    with pytest.raises(ChatGptError) as exc:
        await h.rt.auth.access_token()
    assert exc.value.kind == "TRANSIENT"
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) == rt_before and h.rt.auth.signed_in()


async def test_logout_revokes_then_deletes(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    client_id = h.store.get(ACCOUNT_CLIENT_ID)
    out = await h.rt.chatgpt_logout()
    assert out == {"signed_in": False, "revoked": True, "detail": "refresh token revoked"}
    assert h.server.revocations == [{"token": "[REDACTED]", "token_type_hint": "refresh_token",
                                     "client_id": str(client_id)}]
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None and h.store.get(ACCOUNT_CLIENT_ID) == client_id
    assert h.server.grants[str(client_id)].revoked
    st = h.rt.status()["chatgpt"]
    assert st["signed_in"] is False and st["plan_usage_granted"] is False and st["last_error"] is None
    with pytest.raises(ChatGptError) as exc:
        await h.rt.auth.access_token()
    assert exc.value.kind == "NOT_SIGNED_IN"


async def test_logout_with_unreachable_revocation_reports_unconfirmed(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    await h.sign_in()
    h.server.revoke_status = 503
    out = await h.rt.chatgpt_logout()
    assert out["revoked"] is False
    assert len(h.server.revocations) == ChatGptOAuthConfig.fake().revoke_attempts  # bounded retries
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) is None  # cleared locally anyway
    assert "not confirmed" in (h.rt.status()["chatgpt"]["last_error"] or "")


async def test_listener_times_out(chatgpt_factory: Any) -> None:
    from dataclasses import replace

    h = chatgpt_factory(oauth_config=replace(ChatGptOAuthConfig.fake(), callback_timeout_s=0.2))
    await h.rt.chatgpt_login()
    assert "timed out" in (await h.rt.auth.wait_login(5) or "")
    await asyncio.sleep(0.05)
    assert not h.rt.auth.login_pending
