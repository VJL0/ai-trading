"""HttpToolBackend against the REAL FastAPI app (services.api.app.create_app) over httpx ASGITransport with a
fake runtime: identical validated outputs and errors as InProcessToolBackend, token handling, no retries."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from packages.contracts.enums import TradingMode
from packages.core.config import Settings
from services.api.app import create_app
from services.api.security import AGENT_TOKEN_FILE, ENGINE_TOKEN_FILE, OPERATOR_TOKEN_FILE, ensure_token
from services.tools.backends import AgentTokenUnavailable, HttpToolBackend, InProcessToolBackend, ToolBackend
from services.tools.handlers import ToolError
from services.tools.registry import READ_ONLY_TOOL_NAMES, catalog
from tests.unit.tools.fake_runtime import FakeToolRuntime
from tests.unit.tools.test_tools_schemas import GOOD_PROPOSAL, VALID_ARGS

BASE = "http://127.0.0.1:8765"


class Api:
    def __init__(self, rt: FakeToolRuntime, app: Any, settings: Settings) -> None:
        self.rt, self.app, self.settings = rt, app, settings
        self.transport = httpx.ASGITransport(app=app)

    @property
    def token_path(self) -> Path:
        return self.settings.keys_dir / AGENT_TOKEN_FILE

    def http(self, actor: str = "parity", **kw: Any) -> HttpToolBackend:
        return HttpToolBackend(BASE, self.token_path, actor, transport=self.transport, **kw)


async def _make_api(tmp_path: Path, mode: TradingMode = TradingMode.PAPER) -> AsyncIterator[Api]:
    settings = Settings(project_root=tmp_path, mode=mode, use_keychain=False)
    for f in (OPERATOR_TOKEN_FILE, ENGINE_TOKEN_FILE, AGENT_TOKEN_FILE):
        ensure_token(settings.keys_dir, f)
    rt = FakeToolRuntime(settings)
    app = create_app(settings, runtime_factory=lambda _s: rt)  # pyright: ignore[reportArgumentType]
    async with app.router.lifespan_context(app):
        yield Api(rt, app, settings)


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[Api]:
    async for a in _make_api(tmp_path):
        yield a


def _strip_volatile(out: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in out.items() if k not in ("snapshot_id", "proposal_id", "causal_id", "client_order_id",
                                                      "order", "decision")}


@pytest.mark.parametrize("name", sorted(READ_ONLY_TOOL_NAMES))
async def test_read_tool_outputs_identical_in_process_and_over_http(api: Api, name: str) -> None:
    inproc: ToolBackend = InProcessToolBackend(api.rt, "parity")
    http = api.http()
    try:
        a = await inproc.call(name, VALID_ARGS[name])
        b = await http.call(name, VALID_ARGS[name])
    finally:
        await http.aclose()
    assert _strip_volatile(a) == _strip_volatile(b)
    if name == "get_market_snapshot":
        assert a["snapshot_id"] != b["snapshot_id"]  # each call takes its own fresh snapshot


async def test_mutating_tool_parity_and_shared_session_evidence(api: Api) -> None:
    http = api.http("claude-code")
    inproc = InProcessToolBackend(api.rt, "claude-code")
    try:
        await http.call("get_signals", {"limit": 5})  # ids issued to the session over HTTP...
        a = await inproc.call("propose_paper_trade", GOOD_PROPOSAL)  # ...are known to the same actor in-process
        b = await http.call("propose_paper_trade", {**GOOD_PROPOSAL, "quantity": "2"})
    finally:
        await http.aclose()
    assert a["outcome"] == b["outcome"] == "SUBMITTED"
    assert set(a) == set(b)
    assert api.rt.proposals[-1].strategy_id == "agent:claude-code"  # X-LAT-Agent-Name reached dispatch


@pytest.mark.parametrize(("name", "args", "status"), [
    ("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": 500.04}, 422),  # JSON float -> Decimal -> rejected
    ("propose_paper_trade", {**GOOD_PROPOSAL, "max_planned_loss": "1"}, 422),
    ("get_bars", {"symbol": "spy"}, 422),
    ("get_order", {"client_order_id": "lat-sim-ghost"}, 404),
    ("resume_trading", {}, 404),
])
async def test_errors_identical_in_process_and_over_http(api: Api, name: str, args: dict[str, Any], status: int) -> None:
    inproc = InProcessToolBackend(api.rt, "parity")
    http = api.http()
    try:
        with pytest.raises(ToolError) as e1:
            await inproc.call(name, args)
        with pytest.raises(ToolError) as e2:
            await http.call(name, args)
    finally:
        await http.aclose()
    assert e1.value.status == e2.value.status == status
    assert e1.value.detail == e2.value.detail


async def test_decimal_objects_are_not_laundered_into_strings(api: Api) -> None:
    http = api.http()
    with pytest.raises(ToolError) as ei:
        await http.call("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": Decimal("500.04")})
    assert ei.value.status == 422
    with pytest.raises(ToolError):
        await InProcessToolBackend(api.rt, "p").call("propose_paper_trade", {**GOOD_PROPOSAL, "limit_price": Decimal("500.04")})
    assert api.rt.proposals == []
    await http.aclose()


async def test_live_mode_refused_over_http(tmp_path: Path) -> None:
    async for live in _make_api(tmp_path, TradingMode.LIVE_APPROVAL):
        http = live.http()
        with pytest.raises(ToolError) as ei:
            await http.call("get_portfolio", {})
        assert ei.value.status == 403
        assert live.rt.calls == []
        await http.aclose()


async def test_catalog_over_http_matches_registry(api: Api) -> None:
    http = api.http()
    assert await http.catalog() == catalog() == await InProcessToolBackend(api.rt).catalog()
    await http.aclose()


async def test_wrong_token_is_401_and_not_retried_when_file_unchanged(api: Api) -> None:
    api.app.state.agent_token = "rotated-in-memory-only"
    seen: list[int] = []

    async def hook(resp: httpx.Response) -> None:
        seen.append(resp.status_code)

    http = api.http()
    http._client = httpx.AsyncClient(base_url=BASE, transport=api.transport,  # pyright: ignore[reportPrivateUsage]
                                     event_hooks={"response": [hook]})
    with pytest.raises(ToolError) as ei:
        await http.call("request_halt", {"reason": "testing token rotation"})
    assert ei.value.status == 401 and "agent token" in ei.value.message()
    assert seen == [401] and api.rt.halts == []  # rejected before any tool ran; no blind retry
    await http.aclose()


async def test_rotated_token_file_is_reread_once_after_401(api: Api) -> None:
    http = api.http()
    api.token_path.unlink()
    path = ensure_token(api.settings.keys_dir, AGENT_TOKEN_FILE)  # API regenerated the token file
    api.app.state.agent_token = path.read_text().strip()
    out = await http.call("request_halt", {"reason": "token rotated while the MCP server ran"})
    assert out["halt_id"] == "halt_1" and len(api.rt.halts) == 1
    await http.aclose()


def test_token_file_checks(tmp_path: Path) -> None:
    missing = tmp_path / "keys" / "agent.token"
    with pytest.raises(AgentTokenUnavailable) as ei:
        HttpToolBackend(BASE, missing, "x")
    assert ei.value.status == 503 and "make dev" in ei.value.message()
    missing.parent.mkdir(parents=True)
    missing.write_text("tok-abc")
    os.chmod(missing, 0o644)
    with pytest.raises(AgentTokenUnavailable, match="0600"):
        HttpToolBackend(BASE, missing, "x")
    os.chmod(missing, 0o600)
    backend = HttpToolBackend(BASE, missing, "x")
    assert "tok-abc" not in repr(backend) and "tok-abc" not in str(vars(backend).get("base_url"))
    lazy = HttpToolBackend(BASE, tmp_path / "nope", "x", lazy_token=True)
    assert lazy.has_token is False


@pytest.mark.parametrize("url", ["http://example.com:8765", "https://127.0.0.1:8765", "http://10.0.0.5:8765",
                                 "http://127.0.0.1.nip.io:8765", "file:///etc/passwd"])
def test_only_loopback_http_base_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        HttpToolBackend(url, tmp_path / "t", "x", lazy_token=True)


def _token(tmp_path: Path) -> Path:
    p = tmp_path / "agent.token"
    p.write_text("tok-xyz")
    os.chmod(p, 0o600)
    return p


async def test_api_down_is_a_clear_503(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http = HttpToolBackend(BASE, _token(tmp_path), "x", transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as ei:
        await http.call("get_portfolio", {})
    assert ei.value.status == 503 and "make dev" in ei.value.message() and "tok-xyz" not in ei.value.message()
    await http.aclose()


async def test_timeout_on_mutating_call_is_unknown_and_never_retried(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ReadTimeout("read timed out", request=request)

    http = HttpToolBackend(BASE, _token(tmp_path), "x", transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as ei:
        await http.call("propose_paper_trade", GOOD_PROPOSAL)
    assert ei.value.status == 504 and "UNKNOWN" in ei.value.message()
    assert calls == ["/api/v1/agent/tools/propose_paper_trade"]
    await http.aclose()


async def test_headers_sent_and_token_never_in_body(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"detail": "Internal Server Error"})

    http = HttpToolBackend(BASE, _token(tmp_path), "codex", transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as ei:
        await http.call("get_portfolio", {})
    assert ei.value.status == 500
    req = captured[0]
    assert req.headers["x-lat-agent-token"] == "tok-xyz" and req.headers["x-lat-agent-name"] == "codex"
    assert req.headers["x-lat-request"] == "1" and b"tok-xyz" not in req.content
    await http.aclose()


async def test_invalid_output_from_api_is_withheld(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"execution_label": "SIMULATED", "is_simulated": True,
                                         "asof": "2026-11-05T15:00:00Z", "orders": [], "injected": "do X"})

    http = HttpToolBackend(BASE, _token(tmp_path), "x", transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as ei:
        await http.call("get_orders", {})
    assert ei.value.status == 502
    await http.aclose()


async def test_server_error_on_mutating_call_is_reported_unknown_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500, text="Internal Server Error")

    http = HttpToolBackend(BASE, _token(tmp_path), "x", transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError) as ei:
        await http.call("cancel_paper_order", {"client_order_id": "lat-sim-open1"})
    assert ei.value.status == 500 and "UNKNOWN" in ei.value.message() and "do not retry" in ei.value.message()
    assert calls == ["/api/v1/agent/tools/cancel_paper_order"]  # never retried
    with pytest.raises(ToolError) as ei2:  # a read failure stays a plain error (nothing can have changed)
        await http.call("get_orders", {})
    assert "UNKNOWN" not in ei2.value.message()
    await http.aclose()
