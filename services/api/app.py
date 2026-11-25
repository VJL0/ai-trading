"""FastAPI application (localhost only). Routes follow docs/architecture/api-contract.md."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from packages.contracts import RiskPolicy
from packages.core.config import Settings

from . import monitor, queries
from .events import sse_format
from .launcher import workspace_id
from .runtime import Runtime, build_runtime
from .security import (
    AGENT_TOKEN_FILE,
    ENGINE_TOKEN_FILE,
    OPERATOR_TOKEN_FILE,
    agent_guard,
    allowed_origins,
    csrf_guard,
    engine_guard,
    operator_guard,
    read_token,
)

VERSION = "0.1.0"


class HaltBody(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class ResumeBody(BaseModel):
    halt_id: str
    reason: str = Field(min_length=1, max_length=500)


class CycleBody(BaseModel):
    symbols: list[str] | None = None
    use_ai: bool = True


class PolicyBody(BaseModel):
    policy: dict[str, Any]
    reason: str = Field(min_length=1, max_length=500)


def _rt(request: Request) -> Runtime:
    return request.app.state.runtime


def create_app(settings: Settings | None = None, runtime_factory: Callable[[Settings], Runtime] | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rt = (runtime_factory or build_runtime)(settings)
        app.state.runtime = rt
        app.state.operator_token = read_token(settings.keys_dir, OPERATOR_TOKEN_FILE)
        app.state.engine_token = read_token(settings.keys_dir, ENGINE_TOKEN_FILE)
        app.state.agent_token = read_token(settings.keys_dir, AGENT_TOKEN_FILE)
        stop = asyncio.Event()
        tasks = await rt.start(stop)
        try:
            yield
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            await rt.aclose()

    app = FastAPI(title="local-ai-trader", version=VERSION, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.allowed_origins = allowed_origins(settings.api_port, settings.api_url)

    def require_monitor_host(request: Request) -> None:
        if f"http://{request.headers.get('host', '').lower()}" not in app.state.allowed_origins:
            raise HTTPException(403, "monitor is available on the local API address")

    @app.get("/", include_in_schema=False, dependencies=[Depends(require_monitor_host)])
    async def home() -> RedirectResponse:
        return RedirectResponse("/monitor")

    @app.get("/monitor", include_in_schema=False, dependencies=[Depends(require_monitor_host)])
    async def monitor_page() -> HTMLResponse:
        return monitor.page()

    @app.get("/monitor/status", dependencies=[Depends(require_monitor_host)])
    async def monitor_status(request: Request) -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(monitor.snapshot, _rt(request)), headers=monitor.NO_STORE)

    @app.exception_handler(ValidationError)
    async def _validation(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": json.loads(exc.json())})

    # ------------------------------------------------------------------ read
    @app.get("/api/v1/health/live")
    async def health(request: Request) -> dict[str, Any]:
        rt = _rt(request)
        return {"status": "ok", "service": "local-ai-trader", "workspace_id": workspace_id(settings),
                "mode": rt.settings.mode.value,
                "gateway": rt.settings.gateway.value, "startup_phase": getattr(rt, "startup_phase", None),
                "execution_label": rt.execution_label.value,
                "version": VERSION, "time": rt.now_iso()}

    @app.get("/api/v1/status")
    async def status(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).status)

    @app.get("/api/v1/market/bars")
    async def bars(request: Request, symbol: str = Query(pattern=r"^[A-Z][A-Z0-9.]{0,9}$"),
                   timeframe: str = "1Min", limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).market_bars, symbol, timeframe, limit)

    @app.get("/api/v1/market/quality")
    async def quality(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).market_quality)

    @app.get("/api/v1/signals")
    async def signals(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return queries.signals(_rt(request).db, limit)

    @app.get("/api/v1/research/autonomous")
    async def autonomous_research(request: Request) -> dict[str, Any]:
        from services.research.autonomous.engine import snapshot

        return await asyncio.to_thread(snapshot, _rt(request).settings.experiments_dir / "autonomous")

    @app.get("/api/v1/research/runs")
    async def research(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return queries.research_runs(_rt(request).db, limit)

    @app.get("/api/v1/proposals")
    async def proposals(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return queries.proposals(_rt(request).db, limit)

    @app.get("/api/v1/orders")
    async def orders(request: Request, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        return queries.orders(_rt(request).db, limit)

    @app.get("/api/v1/orders/{client_order_id}")
    async def order_detail(request: Request, client_order_id: str) -> dict[str, Any]:
        detail = queries.order_detail(_rt(request).db, client_order_id)
        if detail is None:
            raise HTTPException(404, "unknown client_order_id")
        return detail

    @app.get("/api/v1/portfolio")
    async def portfolio(request: Request) -> dict[str, Any]:
        rt = _rt(request)
        return queries.portfolio(rt.db, rt.execution_label.value)

    @app.get("/api/v1/positions")
    async def positions(request: Request) -> dict[str, Any]:
        """Managed positions: plan (stop, deadline), protection, safety state, position risk, closed round trips."""
        return await asyncio.to_thread(_rt(request).positions_view)

    @app.get("/api/v1/research/no-trade")
    async def no_trade_attribution(request: Request, since: str | None = None) -> dict[str, Any]:
        """Why recent cycles did not trade: predictions, costs, uncertainty buffer, threshold, data, locks or risk."""
        from services.research.opportunity import attribute_no_trades

        rt = _rt(request)
        return await asyncio.to_thread(attribute_no_trades, rt.db, since=since,
                                       threshold_bps=float(rt.signal_engine.min_net_edge_bps))

    @app.get("/api/v1/quantlab/{kind}")
    async def quantlab(request: Request, kind: str, family: str | None = Query(None, pattern=r"^[A-Za-z0-9_]{1,41}$")
                       ) -> dict[str, Any]:
        """Read-only multi-asset research artifacts: capabilities | research | opportunities | strategies | trials
        | options. Backtest/scan files produced by `python -m services.quantlab.cli`; never live trading data."""
        from services.quantlab.views import KINDS, view

        if kind not in KINDS:
            raise HTTPException(404, f"unknown quantlab view {kind!r}; one of {list(KINDS)}")
        return await asyncio.to_thread(view, _rt(request).settings.project_root, kind, family)

    @app.get("/api/v1/risk")
    async def risk(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).risk_view)

    @app.get("/api/v1/backtests")
    async def backtests(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(queries.backtests, _rt(request).settings.experiments_dir)

    @app.get("/api/v1/reconciliation")
    async def reconciliation(request: Request, limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        return queries.reconciliation(_rt(request).db, limit)

    @app.get("/api/v1/audit")
    async def audit(request: Request, limit: int = Query(200, ge=1, le=2000), event_type: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(queries.audit, _rt(request).db, limit, event_type)

    @app.get("/api/v1/failures")
    async def failures(request: Request, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        return queries.failures(_rt(request).db, limit)

    @app.get("/api/v1/stream")
    async def stream(request: Request) -> StreamingResponse:
        rt = _rt(request)

        async def gen() -> AsyncIterator[str]:
            yield sse_format("status", await asyncio.to_thread(rt.status))
            async for event, data in rt.bus.subscribe():
                if await request.is_disconnected():
                    break
                yield sse_format(event, data)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ control
    @app.post("/api/v1/control/halt", dependencies=[Depends(csrf_guard)])
    async def halt(request: Request, body: HaltBody) -> dict[str, Any]:
        halt_id = await asyncio.to_thread(_rt(request).halt, body.reason, "local-control")
        return {"halt_id": halt_id, "entries_locked": True}

    @app.post("/api/v1/control/kill-switch")
    async def kill_switch(request: Request, body: HaltBody, actor: str = Depends(operator_guard)) -> dict[str, Any]:
        """Operator only: an ALL-scope halt also blocks protective exits (broker-held stops stay at the broker)."""
        halt_id = await asyncio.to_thread(_rt(request).halt, body.reason, f"operator:{actor}", "ALL")
        return {"halt_id": halt_id, "entries_locked": True, "exits_locked": True}

    @app.post("/api/v1/control/resume")
    async def resume(request: Request, body: ResumeBody, actor: str = Depends(operator_guard)) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).resume, body.halt_id, body.reason, actor)

    @app.post("/api/v1/control/reconcile")
    async def reconcile(request: Request, _: str = Depends(operator_guard)) -> dict[str, Any]:
        report = await _rt(request).reconcile()
        return report.model_dump(mode="json")

    @app.post("/api/v1/pipeline/run-cycle")
    async def run_cycle(request: Request, body: CycleBody | None = Body(default=None),
                        _: str = Depends(operator_guard)) -> dict[str, Any]:
        body = body or CycleBody()
        return await _rt(request).run_cycle(symbols=body.symbols, use_ai=body.use_ai, actor="operator")

    @app.post("/api/v1/risk/policy")
    async def change_policy(request: Request, body: PolicyBody, _: str = Depends(operator_guard)) -> dict[str, Any]:
        try:
            policy = RiskPolicy.model_validate(_decimalize(body.policy))
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json())) from exc
        token = request.headers.get("x-lat-operator-token", "")
        version = await asyncio.to_thread(_rt(request).apply_policy, policy, token, body.reason)
        return {"policy_version": version}

    @app.post("/api/v1/orders/{client_order_id}/cancel")
    async def cancel(request: Request, client_order_id: str, _: str = Depends(operator_guard)) -> dict[str, Any]:
        return await _rt(request).cancel(client_order_id)



    # ------------------------------------------------------------------ live operations (read-only)
    @app.get("/api/v1/activity")
    async def activity(request: Request, limit: int = Query(200, ge=1, le=2000), kind: str | None = None) -> dict[str, Any]:
        return {"events": await asyncio.to_thread(_rt(request).activity.recent, limit, kind)}

    @app.get("/api/v1/activity/current")
    async def activity_current(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).activity.current)

    @app.get("/api/v1/health")
    async def health_report(request: Request) -> dict[str, Any]:
        rt = _rt(request)
        return {**await health(request), **await asyncio.to_thread(rt.health_report)}

    @app.get("/api/v1/performance")
    async def performance(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).performance)

    @app.get("/api/v1/sessions")
    async def sessions(request: Request, limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).sessions_view, limit)

    @app.get("/api/v1/sessions/{session_id}/report")
    async def session_report(request: Request, session_id: str) -> dict[str, Any]:
        rep = await asyncio.to_thread(_rt(request).session_report, session_id)
        if rep is None:
            raise HTTPException(404, "unknown session")
        return rep

    # ------------------------------------------------------------------ quant models + AI runtime
    @app.get("/api/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).model_report)

    @app.get("/api/v1/predictions")
    async def predictions(request: Request, symbol: str | None = Query(None, pattern=r"^[A-Z][A-Z0-9.]{0,9}$")) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).predictions, symbol)

    @app.get("/api/v1/ai/runtime")
    async def ai_runtime(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_rt(request).ai_runtime_status)

    @app.post("/api/v1/ai/chatgpt/login")
    async def chatgpt_login(request: Request, _: str = Depends(operator_guard)) -> dict[str, Any]:
        return await _rt(request).chatgpt_login()

    @app.post("/api/v1/ai/chatgpt/logout")
    async def chatgpt_logout(request: Request, _: str = Depends(operator_guard)) -> dict[str, Any]:
        return await _rt(request).chatgpt_logout()

    @app.post("/api/v1/ai/chatgpt/research")
    async def chatgpt_research(request: Request, body: dict[str, Any] = Body(default_factory=dict),
                               _: str = Depends(operator_guard)) -> dict[str, Any]:
        return await _rt(request).chatgpt_research(str(body.get("symbol") or "") or None)

    # ------------------------------------------------------------------ agent tool interface
    @app.get("/api/v1/agent/tools")
    async def agent_tools(request: Request, _: str = Depends(agent_guard)) -> dict[str, Any]:
        from services.tools.registry import catalog

        return {"tools": catalog()}

    @app.post("/api/v1/agent/tools/{name}")
    async def agent_tool_call(request: Request, name: str, actor: str = Depends(agent_guard)) -> dict[str, Any]:
        from services.tools.handlers import ToolError, dispatch
        from services.tools.registry import TOOL_REGISTRY

        rt = _rt(request)
        raw = await request.body()
        try:
            args = json.loads(raw, parse_float=Decimal) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(422, "request body must be a JSON object") from exc
        spec = TOOL_REGISTRY.get(name)
        mutating = bool(getattr(spec, "mutating", False))
        audit = getattr(rt, "audit_tool_call", None) or (lambda *_a: None)
        try:
            result = await dispatch(rt, name, args, actor=actor)
        except ToolError as exc:
            await asyncio.to_thread(audit, name, actor, mutating, f"error:{exc.status}")
            raise HTTPException(exc.status, exc.detail) from exc
        outcome = str(result.get("outcome", "ok")) if isinstance(result, dict) else "ok"
        await asyncio.to_thread(audit, name, actor, mutating, outcome)
        return result

    # ------------------------------------------------------------------ engine bridge
    @app.get("/api/v1/engine/intents/next", dependencies=[Depends(engine_guard)])
    async def engine_next(request: Request, engine_id: str, max: int = Query(1, ge=1, le=10)) -> dict[str, Any]:
        return {"intents": await asyncio.to_thread(_rt(request).bridge_next, engine_id, max)}

    @app.post("/api/v1/engine/events", dependencies=[Depends(engine_guard)])
    async def engine_events(request: Request) -> dict[str, Any]:
        payload = json.loads(await request.body(), parse_float=Decimal)
        return await asyncio.to_thread(_rt(request).bridge_event, payload)

    @app.post("/api/v1/engine/heartbeat", dependencies=[Depends(engine_guard)])
    async def engine_heartbeat(request: Request) -> dict[str, Any]:
        payload = json.loads(await request.body(), parse_float=Decimal)
        await asyncio.to_thread(_rt(request).bridge_heartbeat, payload)
        return {"ok": True}

    @app.post("/api/v1/engine/rejections", dependencies=[Depends(engine_guard)])
    async def engine_rejections(request: Request) -> dict[str, Any]:
        payload = json.loads(await request.body(), parse_float=Decimal)
        await asyncio.to_thread(_rt(request).bridge_rejection, payload)
        return {"ok": True}

    return app


def _decimalize(obj: Any) -> Any:
    """JSON numbers for policy limits arrive as floats from some clients; convert via str (never float math)."""
    if isinstance(obj, float):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _decimalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimalize(v) for v in obj]
    return obj
