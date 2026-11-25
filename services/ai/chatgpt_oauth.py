"""Official "Sign in with ChatGPT" plan-usage flow for an open-source, locally hosted app.

Implemented exactly as documented at developers.openai.com/siwc (sign-in, profiles-and-sessions, token reference,
errors-and-recovery, models-and-inference, preview-limitations; read 2026-11-07/08). No other endpoint is used.

Sign-in (Authorization Code + PKCE S256, public client, no secret):

* Discovery ``https://auth.openai.com/.well-known/openid-configuration`` is preferred at runtime (issuer must be
  ``https://auth.openai.com`` and every endpoint must be https on an allowed auth host, else fail closed); the
  documented endpoints are only a fallback when discovery is unreachable (then remote revocation is unavailable).
* First sign-in uses ``client_id=dynamic_agent_client`` + ``agent_name_hint=local-ai-trader``; the callback returns
  the issued client id (``oaiapp_...``), which is saved and used afterwards (re-authorization omits
  ``agent_name_hint``; a callback carrying a *different* client id is rejected; a registration callback without
  one is incomplete). ``ext_agent_host_id`` is a stable ``urn:uuid:<uuid4>`` kept in ``var/<ns>/keys/``.
* ``scope=openid profile email offline_access resource.invoke chatgpt.tokens.use.direct``,
  ``resource=https://api.openai.com/v1``, fresh ``state`` + ``nonce`` + PKCE verifier per attempt,
  ``redirect_uri=http://127.0.0.1:<port>/auth/callback`` (never ``localhost``). The loopback listener (127.0.0.1
  only, exact path, ``Host`` checked, single use, short timeout) is started **before** the URL is returned.
* The code is exchanged form-encoded (``grant_type=authorization_code, client_id, code, code_verifier,
  redirect_uri, resource``). The ID token is verified (RS256/ES256 against the discovery JWKS; ``iss``, ``aud`` =
  issued client id, ``exp``, ``nonce``). ``chatgpt.tokens.use.direct`` must appear in the *token response* scopes
  before any inference: an identity-only sign-in is never used for inference.

Credentials: the refresh token and issued client id go to the macOS Keychain (service ``local-ai-trader``,
accounts ``CHATGPT_REFRESH_TOKEN`` / ``CHATGPT_CLIENT_ID``); the access token and ID token stay in memory only.
Refresh (``grant_type=refresh_token, client_id, refresh_token, resource``; ``scope`` omitted) is serialized with a
lock and replaces access token, expiry, scopes and the rotating refresh token together; an unusable refresh token
(``invalid_grant``, ``refresh_token_*`` ...) clears the tokens and requires a new sign-in (client id and host id are
kept). Logout POSTs ``token, token_type_hint=refresh_token, client_id`` to the discovery ``revocation_endpoint``
(bounded retries on network/5xx), then deletes the refresh token from the Keychain.

Eligibility: plan usage is for eligible ChatGPT plan users of open-source / locally hosted apps. The docs do not
address financial or automated use, so the operator must confirm eligibility and OpenAI's terms; the app invokes it
only on an explicit operator action or the gated research cycle, never per market tick.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx

from packages.contracts.enums import ModelTransport
from packages.core.clock import Clock, SystemClock
from packages.core.config import Settings
from packages.core.logging import log_event

from .jws import JwsError, validate_id_token
from .token_store import ACCOUNT_CLIENT_ID, ACCOUNT_REFRESH_TOKEN, TokenStore, TokenStoreError

log = logging.getLogger("lat.ai.chatgpt")

REAL_ISSUER: Final = "https://auth.openai.com"
DISCOVERY_PATH: Final = "/.well-known/openid-configuration"
SCOPES: Final[tuple[str, ...]] = (
    "openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct",
)
PLAN_USAGE_SCOPE: Final = "chatgpt.tokens.use.direct"
API_RESOURCE: Final = "https://api.openai.com/v1"
USAGE_URL: Final = "https://chatgpt.com/settings/usage"
DYNAMIC_CLIENT_ID: Final = "dynamic_agent_client"
AGENT_NAME_HINT: Final = "local-ai-trader"
CALLBACK_PATH: Final = "/auth/callback"
PLAN_PRICING_SOURCE: Final = "CHATGPT_PLAN (no per-token charge to the app)"
ELIGIBILITY_NOTE: Final = (
    "user must confirm eligibility and OpenAI terms: ChatGPT plan usage is for eligible ChatGPT plan users of "
    "open-source/locally hosted apps; OpenAI's docs do not address financial or automated use"
)
INVOCATION_POLICY: Final = (
    "invoked only on an explicit operator action or the gated research cycle (selected candidate, per-symbol "
    "interval, daily run limit); never per market tick"
)
REFRESH_TERMINAL_ERRORS: Final[frozenset[str]] = frozenset({
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired", "refresh_token_invalidated",
    "refresh_token_reused",
})
HOST_ID_FILE: Final = "chatgpt_host_id"
REFRESH_LOCK_FILE: Final = "chatgpt_refresh"
STORE_RECHECK_S: Final = 30.0  # how often a cached "no refresh token" answer is re-read from the store
_CLIENT_ID_RE: Final = re.compile(r"^[A-Za-z0-9_.:-]{3,200}$")
_MAX_CALLBACK_REQUEST: Final = 8192


# ------------------------------------------------------------------------------------------------
# configuration / errors
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatGptOAuthConfig:
    issuer: str = REAL_ISSUER
    discovery_url: str = REAL_ISSUER + DISCOVERY_PATH
    # Documented fallbacks, used only when discovery is unreachable.
    authorization_endpoint: str = "https://auth.openai.com/api/accounts/authorize"
    token_endpoint: str = "https://auth.openai.com/api/accounts/oauth/token"  # noqa: S105 - an endpoint URL
    jwks_uri: str = "https://auth.openai.com/.well-known/jwks.json"
    revocation_endpoint: str | None = None  # discovery only
    allowed_auth_hosts: frozenset[str] = frozenset({"auth.openai.com"})
    api_base: str = "https://api.openai.com/v1"
    resource: str = API_RESOURCE
    scope: str = " ".join(SCOPES)
    agent_name_hint: str = AGENT_NAME_HINT
    callback_timeout_s: float = 300.0
    http_timeout_s: float = 30.0
    refresh_margin_s: float = 120.0
    revoke_attempts: int = 3
    revoke_backoff_s: float = 0.5
    id_token_leeway_s: float = 60.0
    simulated: bool = False

    @classmethod
    def fake(cls) -> ChatGptOAuthConfig:
        """FAKE transport endpoints on the reserved ``.invalid`` TLD: a human who opens a simulated authorize URL
        can never reach (or register an app with) the real OpenAI service."""
        issuer = "https://auth.openai-fake.invalid"
        return cls(
            issuer=issuer, discovery_url=issuer + DISCOVERY_PATH,
            authorization_endpoint=issuer + "/api/accounts/authorize", token_endpoint=issuer + "/api/accounts/oauth/token",
            jwks_uri=issuer + "/.well-known/jwks.json", allowed_auth_hosts=frozenset({"auth.openai-fake.invalid"}),
            api_base="https://api.openai-fake.invalid/v1", simulated=True, revoke_backoff_s=0.0,
        )


class ChatGptError(RuntimeError):
    """A sign-in / token / inference failure with a stable ``kind`` and an operator-facing message.

    ``kind`` values: NOT_SIGNED_IN, SIGN_IN_REQUIRED, PLAN_USAGE_NOT_GRANTED, LOGIN_FAILED, DISCOVERY_INVALID,
    TOKEN_ERROR, TRANSIENT, USAGE_LIMIT, NOT_ELIGIBLE, USAGE_UNAVAILABLE, UNSUPPORTED_CAPABILITY,
    ROUTE_NOT_SUPPORTED, INVALID_USER, SCOPE_NOT_AUTHORIZED, ADMISSION_DENIED, INCOMPLETE, STREAM_INTERRUPTED,
    TIMEOUT, PAUSED, BAD_RESPONSE.
    """

    def __init__(self, kind: str, message: str, *, code: str | None = None, status: int | None = None,
                 param: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = code
        self.status = status
        self.param = param

    def describe(self) -> str:
        bits = [self.kind]
        if self.code:
            bits.append(self.code)
        if self.status:
            bits.append(f"HTTP {self.status}")
        return f"{' '.join(bits)}: {self}"


@dataclass(frozen=True)
class Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    revocation_endpoint: str | None
    source: str  # "discovery" | "documented-defaults"


@dataclass
class _AccessToken:
    token: str
    expires_at: datetime
    earliest_refresh_at: datetime | None
    scopes: frozenset[str]


@dataclass
class _PendingLogin:
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    client_id: str  # dynamic_agent_client for a first registration, else the issued id
    registration: bool
    started_at: datetime
    discovery: Discovery
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def pkce_pair() -> tuple[str, str]:
    """``(verifier, S256 challenge)``; the challenge is base64url(SHA-256(verifier)) without padding."""
    verifier = secrets.token_urlsafe(64)  # 86 chars from the unreserved set (RFC 7636: 43..128)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
    return verifier, challenge


def mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    local, _, domain = email.partition("@")
    host, dot, tld = domain.rpartition(".")
    host = host or domain
    masked_domain = f"{host[:1]}***{dot}{tld}" if dot else f"{domain[:1]}***"
    return f"{local[:1]}***@{masked_domain}"


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any, now: datetime) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), UTC) if value > 1_000_000_000 else now + timedelta(seconds=value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def _scopes(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(value.split())
    if isinstance(value, list):
        return frozenset(str(v) for v in cast(list[Any], value))
    return frozenset()


def _https_on(url: Any, hosts: frozenset[str]) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme == "https" and (parts.hostname or "") in hosts and not parts.username and not parts.password


def ensure_host_id(keys_dir: Path) -> str:
    """Stable, opaque per-host id (``urn:uuid:<uuid4>``) persisted 0600 in ``var/<ns>/keys/``."""
    path = keys_dir / HOST_ID_FILE
    with contextlib.suppress(FileNotFoundError):
        value = path.read_text().strip()
        if re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", value):
            return value
    keys_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        keys_dir.chmod(0o700)
    value = f"urn:uuid:{uuid.uuid4()}"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(value + "\n")
    os.replace(tmp, path)
    return value


def _write_private_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(dict(data), fh, sort_keys=True)
    os.replace(tmp, path)


@contextlib.asynccontextmanager
async def cross_process_lock(path: Path, *, timeout_s: float = 30.0) -> AsyncIterator[None]:
    """Exclusive ``flock`` on ``path`` shared by every local process of this namespace (the API process and the
    ``services.ai.chat`` CLI both hold a ChatGPT session backed by the ONE rotating refresh token in the store).
    Polls without blocking the event loop; fails closed (``TRANSIENT``) after ``timeout_s``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ChatGptError("TRANSIENT", "another local process is refreshing the ChatGPT session; "
                                                    "try again shortly") from None
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _oauth_error(resp: httpx.Response) -> tuple[str | None, str]:
    try:
        body = resp.json()
    except ValueError:
        return None, f"HTTP {resp.status_code}"
    if isinstance(body, dict):
        b = cast(dict[str, Any], body)
        err = b.get("error")
        if isinstance(err, dict):
            e = cast(dict[str, Any], err)
            return (str(e.get("code") or e.get("type") or "") or None), str(e.get("message") or "")[:300]
        if isinstance(err, str):
            return err, str(b.get("error_description") or "")[:300]
        if isinstance(b.get("detail"), str):
            return None, str(b["detail"])[:300]
    return None, f"HTTP {resp.status_code}"


# ------------------------------------------------------------------------------------------------
# loopback callback listener
# ------------------------------------------------------------------------------------------------

CallbackHandler = Callable[[dict[str, str]], Awaitable[tuple[int, str, bool]]]

_PAGE = (
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>local-ai-trader</title></head>"
    "<body style=\"font-family:system-ui;margin:3em\"><h1>{title}</h1><p>{body}</p></body></html>"
)


class LoopbackCallbackServer:
    """Single-use HTTP listener on ``127.0.0.1`` for ``GET /auth/callback`` (exact path, ``Host`` checked).

    Requests with another method/path/host are answered 404/400 and ignored; the first request the handler accepts
    as final (``done=True``) closes the listener. The listener also closes after ``timeout_s``.
    """

    def __init__(self, port: int, handler: CallbackHandler, *, timeout_s: float, path: str = CALLBACK_PATH,
                 on_timeout: Callable[[], None] | None = None) -> None:
        self._requested_port = port
        self._handler = handler
        self._timeout_s = timeout_s
        self._path = path
        self._on_timeout = on_timeout
        self._server: asyncio.Server | None = None
        self._timer: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()
        self.port: int | None = None

    async def start(self) -> int:
        try:
            self._server = await asyncio.start_server(self._client, host="127.0.0.1", port=self._requested_port,
                                                      reuse_address=False)
        except OSError as exc:
            raise ChatGptError("LOGIN_FAILED", f"cannot listen on 127.0.0.1:{self._requested_port} for the sign-in "
                                               f"callback ({exc.strerror or type(exc).__name__}); free the port or "
                                               "set LAT_CHATGPT_OAUTH_PORT") from None
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])
        self._timer = asyncio.create_task(self._expire(), name="chatgpt-callback-timeout")
        return self.port

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def _expire(self) -> None:
        await asyncio.sleep(self._timeout_s)
        if not self._closed.is_set():
            if self._on_timeout is not None:
                self._on_timeout()
            await self.close()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
        timer = self._timer
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, title, body, done = 400, "Bad request", "This listener only accepts the ChatGPT sign-in callback.", False
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
            if len(raw) > _MAX_CALLBACK_REQUEST:
                raise ValueError("request too large")
            head = raw.decode("latin-1").split("\r\n")
            method, target, _version = head[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, _, v in (h.partition(":") for h in head[1:] if h)}
            parts = urlsplit(target)
            if method != "GET" or parts.path != self._path:
                status, title, body = 404, "Not found", "Unknown path."
            elif headers.get("host") != f"127.0.0.1:{self.port}":
                status, title, body = 400, "Bad request", "Unexpected Host header."
            else:
                qs = parse_qs(parts.query, keep_blank_values=True, max_num_fields=20)
                if any(len(v) != 1 for v in qs.values()):
                    status, title, body = 400, "Bad request", "Duplicate callback parameters."
                else:
                    status, message, done = await self._handler({k: v[0] for k, v in qs.items()})
                    title = "Signed in" if status == 200 else "Sign-in not completed"
                    body = message
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError, ValueError):
            status, title, body = 400, "Bad request", "Malformed request."
        except Exception as exc:  # never let the listener crash the API process
            log_event(log, "chatgpt callback handler failed", logging.WARNING, error=type(exc).__name__)
            status, title, body = 500, "Sign-in failed", "Internal error while completing sign-in."
        page = _PAGE.format(title=html.escape(title), body=html.escape(body)).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status, "Error")
        head_out = (
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(page)}\r\n"
            "Cache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nX-Content-Type-Options: nosniff\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        with contextlib.suppress(Exception):
            writer.write(head_out + page)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        if done:
            await self.close()


# ------------------------------------------------------------------------------------------------
# the auth session
# ------------------------------------------------------------------------------------------------


class ChatGptAuth:
    """One Sign-in-with-ChatGPT session for this host (access token in memory, refresh token in the store)."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: ModelTransport,
        token_store: TokenStore,
        http_client: httpx.AsyncClient,
        config: ChatGptOAuthConfig | None = None,
        wall_clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.config = config or (ChatGptOAuthConfig.fake() if transport == ModelTransport.FAKE else ChatGptOAuthConfig())
        self.store = token_store
        self.http = http_client
        self.clock: Clock = wall_clock or SystemClock()
        self._access: _AccessToken | None = None
        self._refresh_token: str | None = None
        self._refresh_loaded = False
        self._client_id: str | None = None
        self._client_loaded = False
        self._id_token: str | None = None  # memory only, used as id_token_hint on re-authorization
        self._discovery: Discovery | None = None
        self._jwks: dict[str, Any] | None = None
        self._pending: _PendingLogin | None = None
        self._listener: LoopbackCallbackServer | None = None
        self._refresh_lock = asyncio.Lock()
        self.last_error: str | None = None
        self.on_signed_in: list[Callable[[], None]] = []  # e.g. clear account-specific inference blocks
        suffix = "" if transport == ModelTransport.REAL else ".fake"
        self._meta_path = settings.keys_dir / f"chatgpt_account{suffix}.json"
        self._lock_path = settings.keys_dir / f"{REFRESH_LOCK_FILE}{suffix}.lock"
        self._refresh_read_at = 0.0  # monotonic time of the last store read
        self._meta: dict[str, Any] = self._load_meta()

    # ------------------------------------------------------------------ persistent (non-secret) metadata
    def _load_meta(self) -> dict[str, Any]:
        try:
            data = json.loads(self._meta_path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}

    def _save_meta(self) -> None:
        try:
            _write_private_json(self._meta_path, self._meta)
        except OSError as exc:
            log_event(log, "chatgpt account metadata not saved", logging.WARNING, error=type(exc).__name__)

    def host_id(self) -> str:
        return ensure_host_id(self.settings.keys_dir)

    @property
    def _lock_timeout_s(self) -> float:
        # the holder may be inside discovery + a token request (each bounded by http_timeout_s)
        return 2 * self.config.http_timeout_s + 5.0

    # ------------------------------------------------------------------ stored credentials
    def _stored_client_id(self, *, fresh: bool = False) -> str | None:
        if fresh or not self._client_loaded:
            self._client_id = self.store.get(ACCOUNT_CLIENT_ID)
            self._client_loaded = True
        return self._client_id

    def _stored_refresh_token(self, *, fresh: bool = False) -> str | None:
        """The refresh token. ``fresh=True`` re-reads the store: another local process (the API vs the chat CLI)
        may have rotated it. A cached "none" is re-checked every ``STORE_RECHECK_S`` so a sign-in completed in the
        other process is noticed."""
        stale_none = self._refresh_token is None and time.monotonic() - self._refresh_read_at >= STORE_RECHECK_S
        if fresh or not self._refresh_loaded or stale_none:
            self._refresh_token = self.store.get(ACCOUNT_REFRESH_TOKEN)
            self._refresh_loaded = True
            self._refresh_read_at = time.monotonic()
        return self._refresh_token

    def _clear_tokens(self, *, delete_stored: bool) -> list[str]:
        """Forget access/ID/refresh tokens (client id + host id are kept for a later sign-in)."""
        problems: list[str] = []
        self._access = None
        self._id_token = None
        self._refresh_token = None
        self._refresh_loaded = True
        if delete_stored:
            try:
                self.store.delete(ACCOUNT_REFRESH_TOKEN)
            except TokenStoreError as exc:
                problems.append(str(exc))
                self._refresh_loaded = False
        self._meta.update({"plan_usage_granted": False, "granted_scopes": [], "signed_in": False})
        self._save_meta()
        return problems

    # ------------------------------------------------------------------ discovery / JWKS
    async def discover(self) -> Discovery:
        if self._discovery is not None:
            return self._discovery
        cfg = self.config
        try:
            resp = await self.http.get(cfg.discovery_url, timeout=cfg.http_timeout_s)
        except httpx.HTTPError as exc:
            log_event(log, "chatgpt discovery unreachable; using documented endpoints", logging.WARNING,
                      error=type(exc).__name__)
            return Discovery(cfg.issuer, cfg.authorization_endpoint, cfg.token_endpoint, cfg.jwks_uri,
                             cfg.revocation_endpoint, "documented-defaults")
        if resp.status_code != 200:
            raise ChatGptError("DISCOVERY_INVALID", f"OpenID discovery returned HTTP {resp.status_code}")
        try:
            doc = resp.json()
        except ValueError:
            raise ChatGptError("DISCOVERY_INVALID", "OpenID discovery document is not JSON") from None
        if not isinstance(doc, dict):
            raise ChatGptError("DISCOVERY_INVALID", "OpenID discovery document is not an object")
        d = cast(dict[str, Any], doc)
        if d.get("issuer") != cfg.issuer:
            raise ChatGptError("DISCOVERY_INVALID", "OpenID discovery issuer mismatch")
        hosts = cfg.allowed_auth_hosts
        for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not _https_on(d.get(name), hosts):
                raise ChatGptError("DISCOVERY_INVALID", f"OpenID discovery {name} is not https on an allowed host")
        revocation = d.get("revocation_endpoint")
        if revocation is not None and not _https_on(revocation, hosts):
            raise ChatGptError("DISCOVERY_INVALID", "OpenID discovery revocation_endpoint is not https on an allowed host")
        self._discovery = Discovery(
            issuer=str(d["issuer"]), authorization_endpoint=str(d["authorization_endpoint"]),
            token_endpoint=str(d["token_endpoint"]), jwks_uri=str(d["jwks_uri"]),
            revocation_endpoint=str(revocation) if isinstance(revocation, str) else None, source="discovery",
        )
        return self._discovery

    async def _jwks_doc(self, disc: Discovery, *, force: bool = False) -> dict[str, Any]:
        if self._jwks is not None and not force:
            return self._jwks
        try:
            resp = await self.http.get(disc.jwks_uri, timeout=self.config.http_timeout_s)
        except httpx.HTTPError as exc:
            raise ChatGptError("TRANSIENT", f"JWKS unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise ChatGptError("TOKEN_ERROR", f"JWKS returned HTTP {resp.status_code}")
        try:
            doc = resp.json()
        except ValueError:
            raise ChatGptError("TOKEN_ERROR", "JWKS is not JSON") from None
        if not isinstance(doc, dict):
            raise ChatGptError("TOKEN_ERROR", "JWKS is not an object")
        self._jwks = cast(dict[str, Any], doc)
        return self._jwks

    async def _validate_id_token(self, disc: Discovery, token: str, *, audience: str, nonce: str | None) -> dict[str, Any]:
        now = self.clock.now()
        kwargs: dict[str, Any] = {"issuer": disc.issuer, "audience": audience, "nonce": nonce, "now": now,
                                  "leeway_s": self.config.id_token_leeway_s}
        try:
            return validate_id_token(token, await self._jwks_doc(disc), **kwargs)
        except JwsError as first:
            if "kid" not in str(first) and "ambiguous" not in str(first):
                raise ChatGptError("TOKEN_ERROR", f"ID token rejected: {first}") from None
        try:  # key rotation: refetch the JWKS once
            return validate_id_token(token, await self._jwks_doc(disc, force=True), **kwargs)
        except JwsError as exc:
            raise ChatGptError("TOKEN_ERROR", f"ID token rejected: {exc}") from None

    # ------------------------------------------------------------------ sign-in
    @property
    def pending_callback(self) -> str | None:
        return self._pending.redirect_uri if self.login_pending and self._pending is not None else None

    @property
    def login_pending(self) -> bool:
        return self._pending is not None and self._listener is not None and not self._listener.closed

    async def start_login(self) -> dict[str, Any]:
        """Start the loopback listener, then return the authorize URL the human opens in their own browser."""
        await self.cancel_login()
        disc = await self.discover()
        stored_client = self._stored_client_id()
        registration = stored_client is None
        client_id = DYNAMIC_CLIENT_ID if registration else str(stored_client)
        verifier, challenge = pkce_pair()
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        listener = LoopbackCallbackServer(self.settings.chatgpt_oauth_port, self._on_callback,
                                          timeout_s=self.config.callback_timeout_s, on_timeout=self._on_login_timeout)
        port = await listener.start()  # BEFORE the URL is handed out
        redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
        self._pending = _PendingLogin(state=state, nonce=nonce, verifier=verifier, redirect_uri=redirect_uri,
                                      client_id=client_id, registration=registration, started_at=self.clock.now(),
                                      discovery=disc)
        self._listener = listener
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": self.config.scope,
            "resource": self.config.resource,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "ext_agent_host_id": self.host_id(),
        }
        if registration:
            params["agent_name_hint"] = self.config.agent_name_hint
        elif self._id_token:
            params["id_token_hint"] = self._id_token
        self.last_error = None
        url = f"{disc.authorization_endpoint}?{urlencode(params, quote_via=quote)}"
        log_event(log, "chatgpt sign-in started", registration=registration, callback=redirect_uri,
                  discovery=disc.source, transport=self.transport.value)  # never log the URL (id_token_hint)
        return {"authorize_url": url, "callback": redirect_uri, "expires_in_s": int(self.config.callback_timeout_s),
                "registration": registration, "transport": self.transport.value,
                "simulated": self.config.simulated}

    async def cancel_login(self) -> None:
        listener, self._listener = self._listener, None
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done.is_set():
            pending.error = pending.error or "superseded by a new sign-in attempt"
            pending.done.set()
        if listener is not None:
            await listener.close()

    async def wait_login(self, timeout_s: float | None = None) -> str | None:
        """Wait for the pending sign-in to finish; returns None on success or the error text."""
        pending = self._pending
        if pending is None:
            return self.last_error
        await asyncio.wait_for(pending.done.wait(), timeout=timeout_s)
        return pending.error

    def _on_login_timeout(self) -> None:
        pending = self._pending
        if pending is not None and not pending.done.is_set():
            pending.error = "sign-in timed out (no callback received)"
            self.last_error = pending.error
            pending.done.set()

    async def _on_callback(self, query: dict[str, str]) -> tuple[int, str, bool]:
        pending = self._pending
        if pending is None or pending.done.is_set():
            return 400, "No sign-in is in progress.", False
        state = query.get("state", "")
        if not state or not hmac.compare_digest(state, pending.state):
            # Not ours (or forged): ignore it and keep waiting for the real callback.
            return 400, "State mismatch; this callback was ignored.", False
        try:
            await self._complete(pending, query)
        except ChatGptError as exc:
            pending.error = str(exc)
            self.last_error = pending.error
            pending.done.set()
            log_event(log, "chatgpt sign-in failed", logging.WARNING, kind=exc.kind)
            return 400, f"Sign-in failed: {exc}", True
        pending.done.set()
        for hook in list(self.on_signed_in):
            with contextlib.suppress(Exception):
                hook()
        return 200, "Signed in with ChatGPT. You can close this tab and return to your client.", True

    async def _complete(self, pending: _PendingLogin, query: Mapping[str, str]) -> None:
        if "error" in query:
            err = query.get("error", "")[:64]
            raise ChatGptError("LOGIN_FAILED", f"sign-in was not authorized ({err}); no code was exchanged")
        code = query.get("code", "")
        if not code or len(code) > 4096:
            raise ChatGptError("LOGIN_FAILED", "callback has no authorization code")
        returned_client = query.get("client_id") or None
        if pending.registration:
            if returned_client is None or returned_client == DYNAMIC_CLIENT_ID or not _CLIENT_ID_RE.match(returned_client):
                raise ChatGptError("LOGIN_FAILED", "registration incomplete: the callback did not return an issued "
                                                   "client id")
            client_id = returned_client
        else:
            client_id = pending.client_id
            if returned_client is not None and returned_client != client_id:
                raise ChatGptError("LOGIN_FAILED", "callback returned a different client id than the saved one; "
                                                   "result rejected")
        disc = pending.discovery
        form = {
            "grant_type": "authorization_code", "client_id": client_id, "code": code,
            "code_verifier": pending.verifier, "redirect_uri": pending.redirect_uri, "resource": self.config.resource,
        }
        tok = await self._token_request(disc, form, what="code exchange")
        id_token = tok.get("id_token")
        if not isinstance(id_token, str):
            raise ChatGptError("TOKEN_ERROR", "token response has no id_token")
        claims = await self._validate_id_token(disc, id_token, audience=client_id, nonce=pending.nonce)
        known_sub = self._meta.get("subject")
        if not pending.registration and known_sub and known_sub != claims.get("sub"):
            raise ChatGptError("LOGIN_FAILED", "signed in as a different ChatGPT account than the saved registration; "
                                               "credentials were not replaced")
        if pending.registration:
            try:
                self.store.set(ACCOUNT_CLIENT_ID, client_id)
            except TokenStoreError as exc:
                raise ChatGptError("LOGIN_FAILED", f"could not save the issued client id: {exc}") from None
            self._client_id, self._client_loaded = client_id, True
        self._accept_tokens(tok, claims=claims, id_token=id_token)
        log_event(log, "chatgpt sign-in complete", plan_usage_granted=self.plan_usage_granted,
                  transport=self.transport.value)

    async def _token_request(self, disc: Discovery, form: Mapping[str, str], *, what: str) -> dict[str, Any]:
        try:
            resp = await self.http.post(disc.token_endpoint, data=dict(form), timeout=self.config.http_timeout_s,
                                        headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise ChatGptError("TRANSIENT", f"{what}: token endpoint unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            code, msg = _oauth_error(resp)
            kind = "TRANSIENT" if resp.status_code >= 500 else "TOKEN_ERROR"
            hint = ""
            if code == "invalid_grant" and form.get("grant_type") == "authorization_code":
                hint = " (the code was discarded; start a fresh sign-in)"
            raise ChatGptError(kind, f"{what} failed: {code or 'error'} {msg}{hint}".strip(), code=code,
                               status=resp.status_code)
        try:
            body = resp.json()
        except ValueError:
            raise ChatGptError("TOKEN_ERROR", f"{what}: token response is not JSON") from None
        if not isinstance(body, dict) or not isinstance(cast(dict[str, Any], body).get("access_token"), str):
            raise ChatGptError("TOKEN_ERROR", f"{what}: token response has no access_token")
        b = cast(dict[str, Any], body)
        if str(b.get("token_type", "Bearer")).lower() != "bearer":
            raise ChatGptError("TOKEN_ERROR", f"{what}: unexpected token_type")
        return b

    def _accept_tokens(self, tok: Mapping[str, Any], *, claims: Mapping[str, Any] | None, id_token: str | None) -> None:
        now = self.clock.now()
        expires_in = tok.get("expires_in")
        ttl = float(expires_in) if isinstance(expires_in, int | float) and not isinstance(expires_in, bool) else 3600.0
        scopes = _scopes(tok.get("scope"))  # the TOKEN RESPONSE scopes are authoritative
        self._access = _AccessToken(
            token=str(tok["access_token"]), expires_at=now + timedelta(seconds=max(0.0, ttl)),
            earliest_refresh_at=_parse_time(tok.get("earliest_refresh_at"), now), scopes=scopes,
        )
        if id_token:
            self._id_token = id_token
        new_refresh = tok.get("refresh_token")
        persisted_problem: str | None = None
        if isinstance(new_refresh, str) and new_refresh:
            self._refresh_token, self._refresh_loaded = new_refresh, True  # rotate together with the access token
            try:
                self.store.set(ACCOUNT_REFRESH_TOKEN, new_refresh)
            except TokenStoreError as exc:
                persisted_problem = f"refresh token kept in memory only ({exc}); sign in again after a restart"
        elif self._stored_refresh_token() is None:
            persisted_problem = "no refresh token was issued (offline_access not granted); session ends with the access token"
        if claims is not None:
            self._meta.update({
                "issuer": claims.get("iss"), "subject": claims.get("sub"),
                "email_masked": mask_email(claims.get("email") if isinstance(claims.get("email"), str) else None),
            })
        self._meta.update({
            "signed_in": True, "granted_scopes": sorted(scopes), "plan_usage_granted": PLAN_USAGE_SCOPE in scopes,
            "saved_at": _iso(now), "transport": self.transport.value,
        })
        self._save_meta()
        self.last_error = persisted_problem

    # ------------------------------------------------------------------ access tokens
    @property
    def granted_scopes(self) -> frozenset[str]:
        if self._access is not None:
            return self._access.scopes
        return frozenset(str(s) for s in cast(list[Any], self._meta.get("granted_scopes") or []))

    @property
    def plan_usage_granted(self) -> bool:
        return PLAN_USAGE_SCOPE in self.granted_scopes

    @property
    def access_expires_at(self) -> datetime | None:
        return None if self._access is None else self._access.expires_at

    def signed_in(self) -> bool:
        if self._access is not None and self._access.expires_at > self.clock.now():
            return True
        try:
            return self._stored_refresh_token() is not None
        except TokenStoreError:
            return False

    def invalidate_access_token(self) -> None:
        """Drop the in-memory access token (e.g. after a 401); the next call refreshes once."""
        self._access = None

    async def access_token(self, *, require_plan_usage: bool = True) -> str:
        async with self._refresh_lock:  # serialized: never race a rotating refresh token
            if self._needs_refresh(self.clock.now()):
                await self._refresh_locked()
            acc = self._access
            if acc is None:
                raise ChatGptError("NOT_SIGNED_IN", "not signed in with ChatGPT")
        if require_plan_usage and PLAN_USAGE_SCOPE not in acc.scopes:
            raise ChatGptError("PLAN_USAGE_NOT_GRANTED", f"the sign-in did not grant {PLAN_USAGE_SCOPE}; an identity-only "
                                                         "sign-in is never used for inference")
        return acc.token

    def _needs_refresh(self, now: datetime) -> bool:
        acc = self._access
        if acc is None or acc.expires_at <= now:
            return True
        if acc.expires_at - timedelta(seconds=self.config.refresh_margin_s) > now:
            return False
        # Near expiry: refresh, unless the server asked us not to refresh before earliest_refresh_at.
        return not (acc.earliest_refresh_at is not None and acc.earliest_refresh_at > now)

    async def refresh(self) -> None:
        async with self._refresh_lock:
            await self._refresh_locked()

    async def _refresh_locked(self) -> None:
        # Serialized across local processes too: the API process and the chat CLI share ONE rotating refresh
        # token in the store, so each refresh re-reads it under an exclusive file lock (never a stale token).
        async with cross_process_lock(self._lock_path, timeout_s=self._lock_timeout_s):
            await self._refresh_shared()

    async def _refresh_shared(self) -> None:
        try:
            refresh_token = self._stored_refresh_token(fresh=True)
            client_id = self._stored_client_id(fresh=True)
        except TokenStoreError as exc:
            raise ChatGptError("NOT_SIGNED_IN", f"credential store unavailable: {exc}") from None
        if not refresh_token or not client_id:
            self._access = None
            raise ChatGptError("NOT_SIGNED_IN", "not signed in with ChatGPT (no refresh token)")
        disc = await self.discover()
        form = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token,
                "resource": self.config.resource}  # scope omitted: retain the grant
        try:
            tok = await self._token_request(disc, form, what="token refresh")
        except ChatGptError as exc:
            if exc.code in REFRESH_TERMINAL_ERRORS:
                self._clear_tokens(delete_stored=True)
                self.last_error = f"ChatGPT session ended ({exc.code}); sign in again"
                raise ChatGptError("SIGN_IN_REQUIRED", self.last_error, code=exc.code, status=exc.status) from None
            if exc.code == "invalid_client":
                self.last_error = "ChatGPT client registration rejected (invalid_client); check the client configuration"
                raise ChatGptError("TOKEN_ERROR", self.last_error, code=exc.code, status=exc.status) from None
            self.last_error = str(exc)
            raise  # transient: credentials are preserved
        claims: dict[str, Any] | None = None
        id_token = tok.get("id_token")
        if isinstance(id_token, str):
            claims = await self._validate_id_token(disc, id_token, audience=client_id, nonce=None)
            known = self._meta.get("subject")
            if known and claims.get("sub") != known:
                raise ChatGptError("TOKEN_ERROR", "refreshed identity does not match the saved account")
        self._accept_tokens(tok, claims=claims, id_token=id_token if isinstance(id_token, str) else None)

    # ------------------------------------------------------------------ logout
    async def logout(self) -> dict[str, Any]:
        await self.cancel_login()
        async with self._refresh_lock, cross_process_lock(self._lock_path, timeout_s=self._lock_timeout_s):
            try:
                refresh_token = self._stored_refresh_token(fresh=True)
                client_id = self._stored_client_id(fresh=True)
            except TokenStoreError as exc:
                refresh_token, client_id = self._refresh_token, self._client_id
                self.last_error = str(exc)
            revoked = False
            detail = "no refresh token was stored"
            if refresh_token and client_id:
                revoked, detail = await self._revoke(refresh_token, client_id)
            problems = self._clear_tokens(delete_stored=True)
            if problems:
                detail += "; " + "; ".join(problems)
            if refresh_token and not revoked:
                self.last_error = ("signed out locally, but remote revocation was not confirmed: disconnect the app in "
                                   f"ChatGPT Settings ({detail})")
            else:
                self.last_error = None if not problems else "; ".join(problems)
        log_event(log, "chatgpt signed out", revoked=revoked, transport=self.transport.value)
        return {"signed_in": False, "revoked": revoked, "detail": detail}

    async def _revoke(self, refresh_token: str, client_id: str) -> tuple[bool, str]:
        try:
            disc = await self.discover()
        except ChatGptError as exc:
            return False, f"discovery failed ({exc})"
        if disc.revocation_endpoint is None:
            return False, "the discovery document has no revocation_endpoint"
        form = {"token": refresh_token, "token_type_hint": "refresh_token", "client_id": client_id}
        last = "not attempted"
        for attempt in range(max(1, self.config.revoke_attempts)):
            if attempt:
                await asyncio.sleep(self.config.revoke_backoff_s * (2 ** (attempt - 1)))
            try:
                resp = await self.http.post(disc.revocation_endpoint, data=form, timeout=self.config.http_timeout_s)
            except httpx.HTTPError as exc:
                last = f"revocation endpoint unreachable ({type(exc).__name__})"
                continue
            if resp.status_code == 200:
                return True, "refresh token revoked"
            if resp.status_code < 500:
                code, msg = _oauth_error(resp)
                return False, f"revocation refused: HTTP {resp.status_code} {code or ''} {msg}".strip()
            last = f"revocation endpoint HTTP {resp.status_code}"
        return False, last

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        try:
            signed_in = self.signed_in()
        except Exception:  # status never raises
            signed_in = False
        acc = self._access
        return {
            "signed_in": signed_in,
            "plan_usage_granted": bool(signed_in and self.plan_usage_granted),
            "email_masked": self._meta.get("email_masked") if signed_in else None,
            "expires_at": _iso(acc.expires_at) if acc is not None else None,
            "last_error": self.last_error,
            "login_pending": self.login_pending,
            "transport": self.transport.value,
            "simulated": self.config.simulated,
            "credential_store": self.store.kind,
            "issuer": self.config.issuer,
        }

    async def aclose(self) -> None:
        await self.cancel_login()


__all__ = [
    "API_RESOURCE",
    "CALLBACK_PATH",
    "DYNAMIC_CLIENT_ID",
    "ELIGIBILITY_NOTE",
    "INVOCATION_POLICY",
    "PLAN_PRICING_SOURCE",
    "PLAN_USAGE_SCOPE",
    "REFRESH_TERMINAL_ERRORS",
    "SCOPES",
    "USAGE_URL",
    "ChatGptAuth",
    "ChatGptError",
    "ChatGptOAuthConfig",
    "Discovery",
    "LoopbackCallbackServer",
    "cross_process_lock",
    "ensure_host_id",
    "mask_email",
    "pkce_pair",
]
