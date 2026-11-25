"""Operator CLI: uv run --frozen python -m services.api.cli <init|replay|paper-plan|reconcile|audit|cycle>.

Prints token *file paths*, never token values or secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from packages.contracts.enums import AiRuntimeMode, FeedMode, GatewayKind, TradingMode
from packages.core import modes
from packages.core.audit import verify_chain
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from packages.core.signing import IntentSigner

from .security import ENGINE_TOKEN_FILE, OPERATOR_TOKEN_FILE, ensure_token, read_token


def cmd_init(settings: Settings) -> int:
    settings.ensure_dirs()
    ensure_token(settings.keys_dir, OPERATOR_TOKEN_FILE)
    ensure_token(settings.keys_dir, ENGINE_TOKEN_FILE)
    signer = IntentSigner.load_or_create(settings.keys_dir)
    with Database(settings.db_path) as db:
        from services.risk.policy_store import PolicyStore

        policy = PolicyStore(db, settings.mode).ensure_default()
        print(f"namespace      : {settings.namespace}  ({settings.var_dir})")
        print(f"ledger         : {settings.db_path}  migrations applied now: {db.applied_migrations or 'none'}")
        print(f"intent key id  : {signer.key_id} (private key 0600 in {settings.keys_dir})")
        print(f"operator token : {settings.keys_dir / OPERATOR_TOKEN_FILE} (0600; human operator only)")
        print(f"engine token   : {settings.keys_dir / ENGINE_TOKEN_FILE} (0600; local LEAN process only)")
        print(f"risk policy    : {policy.version}")
    return 0


def _api(settings: Settings, method: str, path: str, body: Any = None, operator: bool = False) -> Any:
    headers = {"X-LAT-Request": "1", "Content-Type": "application/json"}
    if operator:
        headers["X-LAT-Operator-Token"] = read_token(settings.keys_dir, OPERATOR_TOKEN_FILE)
    data = json.dumps(body).encode() if body is not None else None
    url = f"{settings.api_url}{path}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 - Settings validates loopback
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - Settings validates loopback
        return json.loads(r.read())


def _api_up(settings: Settings) -> bool:
    try:
        _api(settings, "GET", "/api/v1/health")
        return True
    except (urllib.error.URLError, OSError):
        return False


def cmd_reconcile(settings: Settings) -> int:
    if _api_up(settings):
        report = _api(settings, "POST", "/api/v1/control/reconcile", operator=True)
        print(json.dumps(report, indent=2))
        return 0 if report.get("status") == "CLEAN" else 3
    from .runtime import Runtime

    rt = Runtime(settings)

    async def go() -> Any:
        return await rt.reconcile()

    try:
        report = asyncio.run(go())
    finally:
        asyncio.run(rt.aclose())
    print(report.model_dump_json(indent=2))
    return 0 if report.status.value == "CLEAN" else 3


def cmd_cycle(settings: Settings, use_ai: bool) -> int:
    if not _api_up(settings):
        print("API is not running; start it with `make dev` (cycles run inside the API so the ledger has one writer).")
        return 2
    print(json.dumps(_api(settings, "POST", "/api/v1/pipeline/run-cycle", {"use_ai": use_ai}, operator=True), indent=2))
    return 0


def cmd_halt(settings: Settings, reason: str) -> int:
    """Operator kill switch: stop new orders now (fail-safe direction; works whether or not the API runs)."""
    if _api_up(settings):
        print(json.dumps(_api(settings, "POST", "/api/v1/control/halt", {"reason": reason}), indent=2))
        return 0
    from packages.core.audit import add_halt

    with Database(settings.db_path) as db:
        halt_id = db.write(lambda c: add_halt(c, reason=reason, source="operator-cli", mode=settings.mode.value))
    print(f"halt recorded directly in the ledger: {halt_id} (API not running)")
    return 0


def cmd_resume(settings: Settings, halt_id: str, reason: str) -> int:
    if not _api_up(settings):
        print("API is not running; start it with `make dev` (resume requires a CLEAN reconciliation).")
        return 2
    print(json.dumps(_api(settings, "POST", "/api/v1/control/resume", {"halt_id": halt_id, "reason": reason},
                          operator=True), indent=2))
    return 0


def cmd_audit(settings: Settings) -> int:
    with Database(settings.db_path) as db:
        ok, n, detail = verify_chain(db)
        print(f"audit chain: {'OK' if ok else 'BROKEN'} rows={n} ({detail})  db={settings.db_path}")
        for r in reversed(db.query("SELECT seq, ts, event_type, source, causal_id FROM audit_events ORDER BY seq DESC LIMIT 20")):
            print(f"  #{r['seq']:<5} {r['ts']} {r['event_type']:<22} {r['source']:<14} {r['causal_id'] or ''}")
        halts = db.query("SELECT halt_id, reason, created_at FROM trading_halts WHERE cleared_at IS NULL")
        print(f"active halts: {len(halts)}")
        for h in halts:
            print(f"  {h['halt_id']} {h['created_at']} {h['reason']}")
    sbom = settings.project_root / "docs" / "sources" / "sbom.json"
    if sbom.exists():
        data = json.loads(sbom.read_text())
        print("sbom: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()) + f"  ({sbom})")
    else:
        print("sbom: not generated yet (make sbom)")
    return 0 if ok else 4


def cmd_paper_plan(settings: Settings) -> int:
    secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
    keys = secrets.alpaca_paper_available()
    print(f"mode={settings.mode} gateway={settings.gateway} live_execution_compiled={modes.LIVE_EXECUTION_COMPILED}")
    if settings.mode != TradingMode.PAPER:
        print("refusing: `make paper` requires LAT_MODE=PAPER")
        return 2
    if settings.gateway == GatewayKind.ALPACA_PAPER and not keys:
        print("BLOCKED: LAT_GATEWAY=ALPACA_PAPER needs Alpaca PAPER keys (docs/runbooks/broker-paper-onboarding.md)")
        return 2
    label = modes.execution_label(settings.mode, settings.gateway)
    feed = settings.market_data_feed(secrets)
    print(f"execution label: {label}   feed: {feed}")
    if not keys:
        print("Alpaca paper keys absent -> running the explicitly SIMULATED fake broker on the synthetic replay feed.")
    if settings.gateway in (GatewayKind.LEAN_PAPER_SIM, GatewayKind.LEAN_ALPACA_PAPER):
        print("LEAN execution retired: select LAT_GATEWAY=ALPACA_PAPER; LEAN is research/backtesting only")
        return 2
    return 0


def cmd_replay(settings: Settings, cycles: int, step_min: int) -> int:
    """Deterministic end-to-end replay with the fake broker and fake model transport (SIMULATED)."""
    if settings.mode != TradingMode.REPLAY:
        print("replay requires LAT_MODE=REPLAY (isolated var/replay namespace)")
        return 2
    settings = settings.model_copy(update={
        "gateway": GatewayKind.FAKE,
        "feed_mode": FeedMode.SIMULATED,
        "ai_runtime": AiRuntimeMode.API_KEYS,
        "ai_transport": "fake",
        "use_keychain": False,
    })
    import shutil

    if settings.var_dir.exists():
        shutil.rmtree(settings.var_dir)  # replay is reproducible from scratch; its namespace holds no live state
    settings.ensure_dirs()
    from .runtime import Runtime

    rt = Runtime(settings)
    out: list[dict[str, Any]] = []

    async def go() -> None:
        stop = asyncio.Event()
        consumer = asyncio.create_task(rt.execution.run_event_consumer(stop))
        report = await rt.reconciler.startup_recovery()
        rt._after_reconcile(report)
        print(f"startup reconciliation: {report.status.value}")
        assert rt.sim_clock is not None
        for i in range(cycles):
            res = await rt.run_cycle(symbols=None, use_ai=True, actor="replay")
            out.append({k: res.get(k) for k in ("causal_id", "ran_at", "candidate_symbol", "outcome", "reason",
                                                 "client_order_id")})
            print(f"[{i + 1:02d}] {res['ran_at']} candidate={res.get('candidate_symbol')} outcome={res.get('outcome')}"
                  f" {('- ' + str(res.get('reason'))[:140]) if res.get('reason') else ''}")
            for _ in range(step_min):
                rt.sim_clock.step(60)
                rt._on_sim_tick(rt.now())
            await asyncio.sleep(0.05)
        final = await rt.reconcile()
        print(f"final reconciliation: {final.status.value} mismatches={len(final.mismatches)}")
        stop.set()
        consumer.cancel()

    try:
        asyncio.run(go())
    finally:
        asyncio.run(rt.aclose())
    run_dir = settings.experiments_dir / "replay" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cycles.json").write_text(json.dumps(out, indent=2))
    print(f"replay results: {run_dir.relative_to(settings.project_root)}  ledger: {settings.db_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="services.api.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("paper-plan")
    sub.add_parser("reconcile")
    sub.add_parser("audit")
    h = sub.add_parser("halt")
    h.add_argument("--reason", default="operator stop via CLI")
    rs = sub.add_parser("resume")
    rs.add_argument("halt_id")
    rs.add_argument("--reason", default="operator resume via CLI")
    c = sub.add_parser("cycle")
    c.add_argument("--no-ai", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--cycles", type=int, default=12)
    r.add_argument("--step-min", type=int, default=15)
    args = p.parse_args(argv)
    settings = Settings()
    if args.cmd == "init":
        return cmd_init(settings)
    if args.cmd == "paper-plan":
        return cmd_paper_plan(settings)
    if args.cmd == "reconcile":
        return cmd_reconcile(settings)
    if args.cmd == "audit":
        return cmd_audit(settings)
    if args.cmd == "halt":
        return cmd_halt(settings, args.reason)
    if args.cmd == "resume":
        return cmd_resume(settings, args.halt_id, args.reason)
    if args.cmd == "cycle":
        return cmd_cycle(settings, use_ai=not args.no_ai)
    if args.cmd == "replay":
        return cmd_replay(settings, args.cycles, args.step_min)
    return 2


if __name__ == "__main__":
    sys.exit(main())


