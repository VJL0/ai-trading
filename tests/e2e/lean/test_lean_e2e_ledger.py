"""E2E (SIMULATED): the REAL self-hosted LEAN engine against the REAL order ledger.

Unlike tests/integration/lean (stub bridge) and tests/integration/order_ledger (scripted Python stand-in for LEAN),
here both ends are the production code: the C# ``LatIntentBridgeAlgorithm`` + ``GatedBrokerageModel`` +
``GatedPaperBrokerage`` (LEAN PaperBrokerage, SIMULATED fills on ``LatSimulatedDataQueue``) talks HTTP to the
``services.order_ledger.bridge`` functions the local API mounts (claim_next_intents / ingest_engine_event /
record_heartbeat / record_engine_rejection), served by a minimal 127.0.0.1 shim that mirrors the four engine routes of
``services/api/app.py`` (engine-token check, ``json.loads(body, parse_float=Decimal)``). Intents are signed by the real
``IntentSigner`` of a throw-away namespace; ``ExecutionService`` + ``LeanBridgeGateway`` queue them; reconciliation is the
real ``Reconciler``. No credentials, no network beyond loopback, nothing reaches Alpaca.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from packages.contracts.enums import (
    DecisionAction,
    GatewayKind,
    OrderState,
    OrderType,
    Proposer,
    ReconciliationStatus,
    RiskVerdict,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.models import ApprovedOrderIntent, RiskDecision, TradeProposal
from packages.core import modes
from packages.core.audit import verify_chain
from packages.core.clock import SystemClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.ids import client_order_id, new_causal_id, new_id
from packages.core.signing import PUBLIC_KEY_FILE, IntentSigner
from services.order_ledger.bridge import (
    EngineTokenAuth,
    LeanBridgeGateway,
    claim_next_intents,
    engine_heartbeat_status,
    ingest_engine_event,
    record_engine_rejection,
    record_heartbeat,
)
from services.order_ledger.entries import EntriesLock
from services.order_ledger.execution import ExecutionService
from services.order_ledger.ledger import OrderLedger
from services.order_ledger.reconcile import Reconciler
from tests.integration.lean import live_sim_harness as harness
from trading.lean.tools import lat_lean

pytestmark = [
    pytest.mark.lean,
    pytest.mark.slow,
    pytest.mark.skipif(not lat_lean.lean_built(), reason="LEAN not built (run scripts/lean_build.sh)"),
]

SIM = GatewayKind.LEAN_PAPER_SIM
S = OrderState
POLICY = "pol_e2e_lean_sim"


class _Shim(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ledger: OrderLedger, db: Database, auth: EngineTokenAuth) -> None:
        super().__init__(("127.0.0.1", 0), _ShimHandler)
        self.ledger, self.db, self.auth = ledger, db, auth
        self.lock = threading.Lock()
        self.log: list[tuple[str, dict[str, Any], dict[str, Any]]] = []  # (path, request, response)
        self.auth_failures = 0


class _ShimHandler(BaseHTTPRequestHandler):
    """The engine routes of services/api/app.py, calling the same order-ledger functions the API runtime calls."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def shim(self) -> _Shim:
        assert isinstance(self.server, _Shim)
        return self.server

    def _reply(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorised(self) -> bool:
        if self.shim.auth.verify(self.headers.get("X-LAT-Engine-Token")):
            return True
        with self.shim.lock:
            self.shim.auth_failures += 1
        self._reply(401, {"detail": "engine token required"})
        return False

    def do_GET(self) -> None:
        if not self._authorised():
            return
        url = urlparse(self.path)
        if url.path != "/api/v1/engine/intents/next":
            self._reply(404, {"detail": "not found"})
            return
        q = parse_qs(url.query)
        engine_id = (q.get("engine_id") or [""])[0]
        n = int((q.get("max") or ["1"])[0])
        out = {"intents": claim_next_intents(self.shim.ledger, engine_id, n)}
        if out["intents"]:
            with self.shim.lock:
                self.shim.log.append((url.path, {"engine_id": engine_id}, out))
        self._reply(200, out)

    def do_POST(self) -> None:
        if not self._authorised():
            return
        path = urlparse(self.path).path
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)), parse_float=Decimal)
        if path == "/api/v1/engine/events":
            resp = ingest_engine_event(self.shim.ledger, payload, gateway_kind=SIM)
        elif path == "/api/v1/engine/heartbeat":
            resp = record_heartbeat(self.shim.db, payload)
        elif path == "/api/v1/engine/rejections":
            resp = record_engine_rejection(self.shim.db, payload)
        else:
            self._reply(404, {"detail": "not found"})
            return
        with self.shim.lock:
            self.shim.log.append((path, payload, resp))
        self._reply(200, resp)


def _sign(signer: IntentSigner, **fields: Any) -> ApprovedOrderIntent:
    unsigned = ApprovedOrderIntent.model_validate({**fields, "signature": "0" * 128})
    return ApprovedOrderIntent.model_validate({**unsigned.model_dump(), "signature": signer.sign(unsigned.signing_string())})


def _approved_intent(ledger: OrderLedger, signer: IntentSigner, side: Side, qty: str, limit: str) -> ApprovedOrderIntent:
    """Proposal + APPROVED decision + signed intent recorded in the ledger (what RiskEngine.authorize produces)."""
    now = datetime.now(UTC)
    causal = new_causal_id()
    proposal = TradeProposal(
        proposal_id=new_id("prop"), causal_id=causal, created_at=now, data_asof=now, action=DecisionAction.TRADE,
        proposer=Proposer.TEST, strategy_id="e2e-lean", symbol="SPY", side=side, quantity=Decimal(qty),
        order_type=OrderType.LIMIT, limit_price=Decimal(limit), max_planned_loss=Decimal("10"),
        evidence_ids=("ev_e2e",), dataset_version="e2e-sim",
    )
    decision = RiskDecision(
        decision_id=new_id("dec"), proposal_id=proposal.proposal_id, causal_id=causal, verdict=RiskVerdict.APPROVED,
        checks=(), reasons=(), policy_version=POLICY, mode=TradingMode.PAPER, evaluated_at=now,
    )
    intent = _sign(
        signer, intent_id=new_id("intent"), client_order_id=client_order_id(modes.client_order_tag(TradingMode.PAPER, SIM)),
        proposal_id=proposal.proposal_id, decision_id=decision.decision_id, causal_id=causal, mode=TradingMode.PAPER,
        gateway=SIM, symbol="SPY", side=side, quantity=Decimal(qty), order_type=OrderType.LIMIT,
        limit_price=Decimal(limit), time_in_force=TimeInForce.DAY, extended_hours=False, policy_version=POLICY,
        created_at=now, expires_at=now + timedelta(seconds=300), requires_human_approval=False, human_approval_id=None,
        key_id=signer.key_id,
    )
    ledger.record_proposal(proposal)
    ledger.record_decision(decision)
    ledger.record_intent(intent)
    return intent


def _start_lean(cfg: Path, run_dir: Path, log_path: Path, token: str) -> subprocess.Popen[bytes]:
    """The LEAN Launcher in its own process group; the engine token only in its environment."""
    with log_path.open("w") as out:
        return subprocess.Popen([lat_lean.dotnet_path(), str(lat_lean.LAUNCHER_DLL), "--config", str(cfg)], cwd=run_dir,
                                stdout=out, stderr=subprocess.STDOUT, env=harness.lean_env(token), start_new_session=True)


def _stop_lean(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


async def _until(pred: Callable[[], bool], within_s: float, what: str, abort: Callable[[], bool] | None = None) -> None:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if pred():
            return
        if abort is not None and abort():
            break
        await asyncio.sleep(0.2)
    if not pred():
        raise AssertionError(f"timed out waiting for: {what}")


@pytest.mark.timeout(300)
async def test_real_lean_engine_executes_ledger_intents_and_refuses_an_oversell(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, use_keychain=False, gateway=SIM)
    settings.ensure_dirs()
    signer = IntentSigner.load_or_create(settings.keys_dir)  # private key stays in this (API-side) process
    auth = EngineTokenAuth(settings.keys_dir)
    token = auth.ensure().read_text().strip()
    clock = SystemClock()
    db = Database(settings.db_path)
    shim = _Shim(OrderLedger(db, clock, TradingMode.PAPER, verifier=signer.verifier), db, auth)
    ledger = shim.ledger
    server = threading.Thread(target=shim.serve_forever, name="ledger-bridge-shim", daemon=True)
    server.start()
    proc: subprocess.Popen[bytes] | None = None
    stop = asyncio.Event()
    consumer: asyncio.Task[None] | None = None
    try:
        entries = EntriesLock(db, TradingMode.PAPER, clock=clock)
        gw = LeanBridgeGateway(ledger, SIM, clock)  # isolated legacy research harness, not a runtime gateway
        assert isinstance(gw, LeanBridgeGateway) and gw.label.value == "SIMULATED"

        def final_gate(intent: ApprovedOrderIntent) -> list[str]:
            return [] if signer.verifier.verify(intent.signing_string(), intent.signature) else ["bad signature"]

        svc = ExecutionService(ledger, gw, final_gate, clock, entries_lock=entries, watchdog_interval_s=3600)
        rec = Reconciler(db, ledger, gw, entries, clock, confirm_delay_s=0)
        consumer = asyncio.create_task(svc.run_event_consumer(stop))
        await asyncio.wait_for(svc.consumer_ready.wait(), 5)

        # ---- the real LEAN Launcher, live mode, holding only the PUBLIC key and the engine token
        run_id = lat_lean.new_run_id("lean-e2e-ledger")
        run_dir = lat_lean.RUNS_DIR / run_id
        cfg = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id, bridge_url=f"http://127.0.0.1:{shim.server_address[1]}",
            pubkey_path=settings.keys_dir / PUBLIC_KEY_FILE, sim_quotes=harness.SIM_QUOTES, exit_after_seconds=45,
            journal_path=settings.var_dir / "lean" / "used-intents.tsv"))
        log_path = run_dir / "launcher-stdout.log"
        proc = engine = _start_lean(cfg, run_dir, log_path, token)

        def dead() -> bool:
            return engine.poll() is not None

        def healthy() -> bool:
            return engine_heartbeat_status(db, SystemClock(), 15, expected_gateway=SIM, expected_mode="PAPER")[0]

        await _until(healthy, 90, "LEAN heartbeat RUNNING (all universe data present)", abort=dead)

        # ---- startup reconciliation must be CLEAN before anything is handed to the engine
        report = await rec.startup_recovery()
        if report.entries_locked_after:
            report = await rec.reconcile()
        assert report.status == ReconciliationStatus.CLEAN and not report.entries_locked_after, report
        assert "SIMULATED" in report.broker_source

        def state(coid: str) -> OrderState | None:
            r = ledger.get_record(coid)
            return r.state if r else None

        # ---- 1. signed BUY 10 @ 450.10: queued -> claimed by LEAN -> gate -> PaperBrokerage fill -> ledger FILLED
        buy = _approved_intent(ledger, signer, Side.BUY, "10", "450.10")
        queued = await svc.submit(buy)
        assert queued is not None and queued.state == S.SUBMITTING and queued.broker_status == "queued_for_engine"
        await _until(lambda: state(buy.client_order_id) == S.FILLED, 45, "BUY filled in the ledger", abort=dead)
        [fill] = ledger.fills(buy.client_order_id)
        assert Decimal(fill["quantity"]) == Decimal("10")
        assert Decimal("450.00") < Decimal(fill["price"]) <= Decimal("450.10")  # LatFillModel: costs added, capped at limit
        [pos] = ledger.positions()
        assert pos.symbol == "SPY" and pos.quantity == Decimal(10)
        buy_rec = ledger.get_record(buy.client_order_id)
        assert buy_rec is not None and buy_rec.broker_order_id, "PaperBrokerage id mapped as broker_order_id"

        # ---- 2. signed SELL 50 while holding 10: LEAN's gate refuses (long-only) -> /engine/rejections -> REJECTED
        oversell = _approved_intent(ledger, signer, Side.SELL, "50", "449.90")
        assert await svc.submit(oversell) is not None
        await _until(lambda: state(oversell.client_order_id) == S.REJECTED, 45, "oversell REJECTED", abort=dead)
        assert ledger.fills(oversell.client_order_id) == []
        with shim.lock:
            rejections = [(p, r) for p, r, _ in shim.log if p.endswith("/rejections")]
        assert any(r.get("client_order_id") == oversell.client_order_id and "short sale refused" in str(r.get("reason"))
                   and "[GatedBrokerageModel]" in str(r.get("reason")) for _, r in rejections), rejections

        # ---- 3. signed SELL 10 @ 449.90 closes the position
        close = _approved_intent(ledger, signer, Side.SELL, "10", "449.90")
        assert await svc.submit(close) is not None
        await _until(lambda: state(close.client_order_id) == S.FILLED, 45, "closing SELL filled", abort=dead)
        [sell_fill] = ledger.fills(close.client_order_id)
        assert Decimal(sell_fill["quantity"]) == Decimal("10")
        assert Decimal("449.90") <= Decimal(sell_fill["price"]) < Decimal("450.00")
        assert all(p.quantity == 0 for p in ledger.positions())

        # ---- broker (ledger-derived SIM) state reconciles CLEAN; every engine event was unique and attributed
        after = await rec.reconcile()
        assert after.status == ReconciliationStatus.CLEAN and not after.entries_locked_after, after
        assert ledger.duplicate_count() == 0
        with shim.lock:
            events = [(r, resp) for p, r, resp in shim.log if p.endswith("/events")]
            claims = [resp for p, _, resp in shim.log if p.endswith("/intents/next")]
        claimed = [i["client_order_id"] for c in claims for i in c["intents"]]
        assert sorted(claimed) == sorted([buy.client_order_id, oversell.client_order_id, close.client_order_id])
        assert all(resp.get("accepted") for _, resp in events), events
        assert {r["client_order_id"] for r, _ in events} == set(claimed)
        assert shim.auth_failures == 0

        # ---- the engine stops (configured run time): its STOPPED heartbeat makes the gateway unhealthy (fail closed)
        await asyncio.to_thread(engine.wait, 120)
        await _until(lambda: not gw.engine_health()[0], 10, "engine health turns false after STOPPED heartbeat")
        assert "STOPPED" in gw.engine_health()[1]
        ok, _, detail = verify_chain(db)
        assert ok, detail
        text = log_path.read_text(errors="replace")
        assert "LAT bridge: all universe securities have data; intent intake ENABLED" in text
        assert "Runtime Error" not in text
    finally:
        stop.set()
        if consumer is not None:
            await asyncio.wait_for(consumer, 5)
        if proc is not None:
            await asyncio.to_thread(_stop_lean, proc)
        shim.shutdown()
        shim.server_close()
        db.close()
