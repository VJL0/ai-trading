"""Tool backends: how an AI runtime reaches the ONE tool interface.

* ``InProcessToolBackend(runtime, actor)``: calls ``handlers.dispatch`` directly (in-app runtimes such as the
  Sign-in-with-ChatGPT research turn run inside the API process).
* ``HttpToolBackend(base_url, agent_token_path, actor)``: calls the local API
  (``POST /api/v1/agent/tools/{name}`` with ``X-LAT-Agent-Token``). Used by the MCP server so the API process
  stays the single SQLite writer.

Both return the identical, output-model-validated JSON dict and raise ``ToolError`` with the same status codes,
so a model sees the same contract whichever runtime drives it. The HTTP backend:

* only talks to a loopback base URL (fail closed otherwise);
* reads the agent token from ``var/<ns>/keys/agent.token`` and never logs, prints or returns it;
* uses explicit connect/read timeouts and never retries a tool call (an outcome after a timeout is UNKNOWN);
  the single exception is a 401 (rejected by the auth guard *before* any tool ran), after which the token
  file is re-read once in case the API regenerated it.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from packages.contracts.enums import GatewayKind, TradingMode
from packages.core.config import Settings

from .handlers import ToolError, dispatch
from .protocol import ToolRuntime
from .registry import catalog as _catalog
from .registry import get_spec
from .sanitize import sanitize_actor, truncate

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
AGENT_TOKEN_FILE = "agent.token"  # noqa: S105 - a file name, not a secret
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@runtime_checkable
class ToolBackend(Protocol):
    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def catalog(self) -> list[dict[str, Any]]: ...


class InProcessToolBackend:
    def __init__(self, runtime: ToolRuntime, actor: str = "in-process") -> None:
        self._runtime = runtime
        self.actor = sanitize_actor(actor)

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return await dispatch(self._runtime, name, dict(args), actor=self.actor)

    async def catalog(self) -> list[dict[str, Any]]:
        return _catalog()

    def __repr__(self) -> str:
        return f"InProcessToolBackend(actor={self.actor!r})"


class AgentTokenUnavailable(ToolError):
    def __init__(self, path: Path, why: str) -> None:
        super().__init__(503, f"agent token unavailable ({why}) at {path}. Start the local API (make dev); it "
                              "creates the token on first start and the next tool call picks it up.")


def read_agent_token(path: Path) -> str:
    """Read the agent token file (0600, owned by this user). The value is never logged or returned to a model."""
    try:
        st = path.stat()
    except FileNotFoundError:
        raise AgentTokenUnavailable(path, "file not found") from None
    except OSError as exc:
        raise AgentTokenUnavailable(path, type(exc).__name__) from None
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise AgentTokenUnavailable(path, "file permissions are wider than 0600")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise AgentTokenUnavailable(path, "file is not owned by the current user")
    token = path.read_text().strip()
    if not token:
        raise AgentTokenUnavailable(path, "file is empty")
    return token


def _json_default(obj: Any) -> Any:
    # No laundering: a Decimal/float/datetime is sent (or refused) as-is so the server-side contract decides
    # exactly as the in-process path does (money must be decimal *strings*).
    raise TypeError(f"tool arguments must be plain JSON values, got {type(obj).__name__}")


def _require_loopback(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme != "http" or (parts.hostname or "") not in _LOOPBACK_HOSTS:
        raise ValueError("HttpToolBackend only talks to the local API on a loopback http:// address")
    return base_url.rstrip("/")


class HttpToolBackend:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        agent_token_path: Path | str | None = None,
        actor: str = "mcp",
        *,
        connect_timeout_s: float = 3.0,
        read_timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        lazy_token: bool = False,
        startup_settings: Settings | None = None,
    ) -> None:
        self.base_url = _require_loopback(base_url)
        self.actor = sanitize_actor(actor)
        self._startup_settings = startup_settings
        if agent_token_path is None:
            agent_token_path = Settings().keys_dir / AGENT_TOKEN_FILE
        self.token_path = Path(agent_token_path)
        self._token: str | None = None
        if not lazy_token:
            self._token = read_agent_token(self.token_path)
        self._timeout = httpx.Timeout(read_timeout_s, connect=connect_timeout_s)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def __repr__(self) -> str:  # never include the token
        return f"HttpToolBackend(base_url={self.base_url!r}, actor={self.actor!r})"

    @property
    def has_token(self) -> bool:
        return self._token is not None

    def load_token(self) -> None:
        """(Re-)read the agent token file now; raises ``AgentTokenUnavailable`` (a 503 ``ToolError``)."""
        self._token = read_agent_token(self.token_path)

    def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout, transport=self._transport,
                                             trust_env=False, follow_redirects=False)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        if self._token is None:
            self._token = read_agent_token(self.token_path)
        return {"X-LAT-Agent-Token": self._token, "X-LAT-Agent-Name": self.actor, "X-LAT-Request": "1",
                "Content-Type": "application/json", "Accept": "application/json"}

    async def _request(self, method: str, path: str, body: bytes | None, *, mutating: bool) -> httpx.Response:
        client = self._client_or_new()
        for attempt in (1, 2):
            try:
                resp = await client.request(method, path, content=body, headers=self._headers())
            except httpx.TimeoutException:
                if mutating:
                    raise ToolError(504, f"local trading API timed out at {self.base_url}; the outcome is UNKNOWN. "
                                         "Check get_orders before proposing again; do not retry blindly.") from None
                raise ToolError(504, f"local trading API timed out at {self.base_url}") from None
            except httpx.TransportError as exc:
                raise ToolError(503, f"local trading API is not reachable at {self.base_url} ({type(exc).__name__}). "
                                     "Start it with `make dev` (or `make paper`) and try again.") from None
            if resp.status_code == 401 and attempt == 1:
                # Auth is checked before any tool executes, so re-reading a regenerated token is safe even for
                # mutating calls. Only retry if the token on disk actually changed.
                old = self._token
                try:
                    self._token = read_agent_token(self.token_path)
                except ToolError:
                    raise
                if self._token != old:
                    continue
            return resp
        raise AssertionError("unreachable")

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        spec = get_spec(name)
        mutating = spec.mutating if spec is not None else True
        try:
            body = json.dumps(dict(args), default=_json_default, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError) as exc:
            raise ToolError(422, str(exc)) from None
        if name == "start_paper_session" and self._startup_settings is not None and spec is not None:
            try:
                spec.input_model.model_validate(args)
            except ValidationError as exc:
                raise ToolError(422, exc.errors(include_input=False, include_context=False, include_url=False)) from None
            from services.api.launcher import ApiStartupError, ensure_api

            settings = self._startup_settings
            if settings.mode not in (TradingMode.PAPER, TradingMode.REPLAY) or settings.gateway not in (
                GatewayKind.FAKE, GatewayKind.ALPACA_PAPER,
            ):
                raise ToolError(403, "session startup requires an approved paper or simulated runtime")
            try:
                await asyncio.to_thread(
                    ensure_api, settings,
                    require_gateway_match=settings.trading_mode is not None or "gateway" in settings.model_fields_set,
                )
            except (ApiStartupError, OSError) as exc:
                raise ToolError(503, f"the local API could not prepare a paper session: {exc}") from None
        resp = await self._request("POST", f"/api/v1/agent/tools/{name}", body, mutating=mutating)
        payload = _decode(resp)
        if resp.status_code >= 400:
            err = _error_from(resp.status_code, payload, self.base_url)
            if mutating and resp.status_code >= 500 and "UNKNOWN" not in err.message():
                # A server-side failure after the request was accepted: the action may or may not have happened.
                raise ToolError(err.status, f"{err.message()}; the outcome of {name} is UNKNOWN. Inspect get_orders/"
                                            "get_risk_status before trying again; do not retry blindly.")
            raise err
        if spec is None or not isinstance(payload, dict):
            raise ToolError(502, f"unexpected response from the local API for {name}")
        # Re-validate locally: the model only ever sees output that satisfies the strict output contract.
        try:
            return spec.output_model.model_validate(payload).model_dump(mode="json")
        except Exception:
            raise ToolError(502, f"{name} output from the local API failed contract validation (withheld)") from None

    async def catalog(self) -> list[dict[str, Any]]:
        try:
            resp = await self._request("GET", "/api/v1/agent/tools", None, mutating=False)
        except ToolError as exc:
            if exc.status != 503 or self._startup_settings is None:
                raise
            return [entry for entry in _catalog() if entry["name"] == "start_paper_session"]
        payload = _decode(resp)
        if resp.status_code >= 400:
            raise _error_from(resp.status_code, payload, self.base_url)
        tools = cast(dict[str, Any], payload).get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list):
            raise ToolError(502, "unexpected catalog response from the local API")
        return cast(list[dict[str, Any]], tools)


def _decode(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"detail": truncate(resp.text, 300)}


def _error_from(status: int, payload: Any, base_url: str) -> ToolError:
    detail = cast(dict[str, Any], payload).get("detail") if isinstance(payload, dict) else payload
    if status == 401:
        return ToolError(401, "the local API rejected the agent token; restart the MCP server after the API has "
                              "started (it regenerates var/<ns>/keys/agent.token only if missing)")
    if status >= 500 and not isinstance(detail, str | list):
        detail = f"local trading API error {status} at {base_url}"
    return ToolError(status, detail)
