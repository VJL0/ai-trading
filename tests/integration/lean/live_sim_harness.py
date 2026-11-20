"""LOCAL SIMULATED live run of the real LEAN engine (self-hosted Launcher, live mode) against the stub bridge.

Everything here is SIMULATED: LEAN's PaperBrokerage (via GatedPaperBrokerage) fills orders locally against the
deterministic LatSimulatedDataQueue; nothing leaves 127.0.0.1. The intent signer is the REAL Python IntentSigner
(fresh throw-away key per run, private key kept in memory only; LEAN receives only the public key).

Scenarios:
  * ``bridge`` - LatIntentBridgeAlgorithm: one valid signed BUY must fill; tampered, forged (other key), expired and
    over-selling (short) intents must be refused by the gate and reported to /engine/rejections.
  * ``rogue``  - test-only LatRogueStrategyAlgorithm: unsigned strategy orders (before and after swapping the
    brokerage model) must be refused at the model hook and at the final brokerage boundary; the bridge halts.

    uv run --frozen python -m tests.integration.lean.live_sim_harness [bridge|rogue] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import GatewayKind, OrderType, Side, TimeInForce, TradingMode
from packages.contracts.models import ApprovedOrderIntent
from packages.core.signing import IntentSigner
from tests.integration.lean.stub_bridge import StubBridge
from trading.lean.tools import lat_lean

SIM_QUOTES = {"SPY": {"bid": "449.95", "ask": "450.05", "last": "450.00"}}


def make_intent(signer: IntentSigner, run_id: str, n: int, *, side: Side, qty: str, limit: str, key_id: str | None = None,
                created: datetime | None = None, ttl_s: int = 300, order_type: OrderType = OrderType.LIMIT,
                gateway: GatewayKind = GatewayKind.LEAN_PAPER_SIM) -> ApprovedOrderIntent:
    created = created or datetime.now(UTC)
    fields: dict[str, Any] = {
        "intent_id": f"int_{run_id}_{n}"[:120], "client_order_id": f"lat-sim-{run_id}-{n}"[:128],
        "proposal_id": f"prop_{n}", "decision_id": f"dec_{n}", "causal_id": f"cz_{run_id}_{n}"[:120],
        "mode": TradingMode.PAPER, "gateway": gateway, "symbol": "SPY", "side": side,
        "quantity": Decimal(qty), "order_type": order_type, "limit_price": Decimal(limit),
        "time_in_force": TimeInForce.DAY, "extended_hours": False, "policy_version": "pol_live_sim",
        "created_at": created, "expires_at": created + timedelta(seconds=ttl_s), "requires_human_approval": False,
        "human_approval_id": None, "key_id": key_id or signer.key_id, "signature": "0" * 128,
    }
    unsigned = ApprovedOrderIntent(**fields)
    fields["signature"] = signer.sign(unsigned.signing_string())
    return ApprovedOrderIntent(**fields)


@dataclass
class Expectation:
    label: str
    intent: ApprovedOrderIntent
    outcome: str  # "fill" | "refuse"
    reason: str = ""  # substring the gate's refusal reason must contain
    keyed: bool = True  # False: the report must carry NO ids (the original order of a duplicate may be live)


# rogue scenario: (gate layer, reason substring, LEAN order type) that must each appear in /engine/rejections
ROGUE_REFUSALS: tuple[tuple[str, str, str], ...] = (
    ("[GatedBrokerageModel]", "unsigned order", "Limit"),
    ("[GatedPaperBrokerage]", "unsigned order", "Market"),
    ("[GatedPaperBrokerage]", "signature verification failed", "Limit"),
)


def refusal_found(rejections: list[dict[str, Any]], *, client_order_id: str | None = None, layer: str = "",
                  reason: str = "", order_type: str = "", unkeyed: bool = False) -> bool:
    """True when a rejection matches. ``client_order_id`` set: the report must be keyed by it; ``unkeyed``: the
    report must carry null ids (and name nothing else than the reason)."""
    for r in rejections:
        text = str(r.get("reason", ""))
        if unkeyed and (r.get("client_order_id") is not None or r.get("intent_id") is not None):
            continue
        if client_order_id is not None and r.get("client_order_id") != client_order_id:
            continue
        if layer in text and reason in text and (not order_type or f" {order_type} SPY " in text):
            return True
    return False


def expectation_met(rejections: list[dict[str, Any]], x: Expectation) -> bool:
    if x.keyed:
        return refusal_found(rejections, client_order_id=x.intent.client_order_id, reason=x.reason)
    return refusal_found(rejections, reason=x.reason, unkeyed=True)


def lean_env(token: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ALPACA_", "APCA_", "ANTHROPIC_", "OPENAI_"))}
    env["LAT_ENGINE_TOKEN"] = token
    env.pop("LAT_INTENT_PUBKEY_PATH", None)  # the per-run config names the key explicitly
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    env["DOTNET_NOLOGO"] = "1"
    return env


@dataclass
class EngineRun:
    stdout_path: Path
    started: datetime
    completed: bool
    exit_code: int | None
    log_text: str
    # every internet socket the LEAN process held while sampled (lsof NAME column), e.g. "127.0.0.1:5->127.0.0.1:6"
    sockets: list[str]


def _sample_sockets(pid: int, out: set[str], stop: threading.Event) -> None:
    """Records the LEAN process's internet sockets once per second (macOS ``lsof``; best effort)."""
    exe = shutil.which("lsof") or "/usr/sbin/lsof"
    while not stop.is_set():
        try:
            res = subprocess.run([exe, "-nP", "-a", "-p", str(pid), "-i"], capture_output=True, text=True, timeout=10,
                                 check=False)
            for line in res.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 9:
                    out.add(" ".join(parts[8:]))
        except (OSError, subprocess.TimeoutExpired):
            pass
        stop.wait(1.0)


def non_loopback(sockets: list[str]) -> list[str]:
    """Sockets whose local or remote host is neither 127.0.0.1 nor ::1 (UDP/TCP listeners on '*' count as non-local)."""
    bad = []
    for name in sockets:
        endpoints = name.split(" ")[0].split("->")
        for ep in endpoints:
            host = ep.rsplit(":", 1)[0].strip("[]")
            if host not in ("127.0.0.1", "::1", "localhost"):
                bad.append(name)
                break
    return bad


def launch_engine(run_dir: Path, cfg: Path, token: str, stub: StubBridge, done: Callable[[dict[str, Any]], bool], *,
                  exit_after_s: int, timeout_s: float) -> EngineRun:
    """Runs the LEAN Launcher on ``cfg`` until ``done(stub snapshot)`` holds (or LEAN exits / timeout), then lets
    LEAN stop on its own (lat-exit-after-seconds) so its final heartbeat and outbox drain are observed."""
    stdout_path = run_dir / "launcher-stdout.log"
    started = datetime.now(UTC)
    with stdout_path.open("w") as out:
        proc = subprocess.Popen([lat_lean.dotnet_path(), str(lat_lean.LAUNCHER_DLL), "--config", str(cfg)],
                                cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, env=lean_env(token),
                                start_new_session=True)
    exit_code: int | None = None
    sockets: set[str] = set()
    stop_sampling = threading.Event()
    sampler = threading.Thread(target=_sample_sockets, args=(proc.pid, sockets, stop_sampling), daemon=True)
    sampler.start()
    try:
        completed = stub.wait_for(done, timeout_s, abort=lambda: proc.poll() is not None)
        try:
            exit_code = proc.wait(timeout=max(5.0, exit_after_s + 30 - (datetime.now(UTC) - started).total_seconds()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            exit_code = proc.wait(timeout=15)
    finally:
        stop_sampling.set()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        sampler.join(timeout=15)
    time.sleep(0.5)
    log_text = stdout_path.read_text(errors="replace")
    if "Runtime Error" in log_text or "During the algorithm initialization" in log_text:
        completed = False
    return EngineRun(stdout_path, started, completed, exit_code, log_text, sorted(sockets))


def new_sim_run(prefix: str) -> tuple[str, Path, Path, IntentSigner]:
    """Fresh run id + run dir + throw-away signer whose PUBLIC key is written for LEAN (private key: memory only)."""
    run_id = lat_lean.new_run_id(prefix)
    run_dir = lat_lean.RUNS_DIR / run_id
    keys = run_dir / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    signer = IntentSigner.generate()
    pub = keys / "intent_signing.pub.pem"
    pub.write_bytes(signer.verifier.public_pem())
    return run_id, run_dir, pub, signer


def run(scenario: str = "bridge", *, exit_after_s: int = 40, timeout_s: float = 150.0) -> dict[str, Any]:
    if scenario not in ("bridge", "rogue"):
        raise ValueError("scenario must be bridge or rogue")
    if not lat_lean.lean_built():
        raise RuntimeError("LEAN not built (scripts/lean_build.sh)")
    run_id, run_dir, pub, signer = new_sim_run(f"lean-live-sim-{scenario}")
    forger = IntentSigner.generate()
    token = secrets.token_hex(24)
    short = run_id[-6:]

    expectations: list[Expectation] = []
    with StubBridge(token) as stub:
        buy = make_intent(signer, short, 1, side=Side.BUY, qty="10", limit="450.10")
        stub.add_intent(buy, label="valid_buy")
        expectations.append(Expectation("valid_buy", buy, "fill" if scenario == "bridge" else "refuse"))
        if scenario == "bridge":
            victim = make_intent(signer, short, 2, side=Side.BUY, qty="10", limit="450.10")
            # tampered in transit: 10 -> 11 shares (still within buying power, so LEAN's own checks pass and the
            # signature check of the gate is what refuses it)
            stub.add_intent(victim, label="tampered_quantity",
                            signing_string=victim.signing_string().replace("|quantity=10|", "|quantity=11|"))
            forged = make_intent(forger, short, 3, side=Side.BUY, qty="10", limit="450.10", key_id=signer.key_id)
            stub.add_intent(forged, label="forged_other_key")
            old = datetime.now(UTC) - timedelta(minutes=10)
            expired = make_intent(signer, short, 4, side=Side.BUY, qty="10", limit="450.10", created=old, ttl_s=60)
            stub.add_intent(expired, label="expired")
            oversell = make_intent(signer, short, 5, side=Side.SELL, qty="50", limit="449.90")
            stub.add_intent(oversell, label="short_sale")
            market = make_intent(signer, short, 6, side=Side.BUY, qty="1", limit="450.10", order_type=OrderType.MARKET)
            stub.add_intent(market, label="signed_market_order")
            other_gw = make_intent(signer, short, 7, side=Side.BUY, qty="1", limit="450.10",
                                   gateway=GatewayKind.LEAN_ALPACA_PAPER)
            stub.add_intent(other_gw, label="wrong_gateway")
            # the API must hand an intent out once; a buggy/hostile API delivering it twice must not double-submit
            stub.add_intent(buy, label="duplicate_delivery")
            expectations += [
                Expectation("tampered_quantity", victim, "refuse", "signature verification failed"),
                Expectation("forged_other_key", forged, "refuse", "signature verification failed"),
                Expectation("expired", expired, "refuse", "expired"),
                Expectation("short_sale", oversell, "refuse", "short sale refused"),
                Expectation("signed_market_order", market, "refuse", "LIMIT only"),
                Expectation("wrong_gateway", other_gw, "refuse", "is for gateway 'LEAN_ALPACA_PAPER'"),
                Expectation("duplicate_delivery", buy, "refuse", f"duplicate delivery of intent {buy.intent_id}", keyed=False),
            ]

        cfg = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id, bridge_url=stub.url, pubkey_path=pub, sim_quotes=SIM_QUOTES,
            exit_after_seconds=exit_after_s, rogue=(scenario == "rogue")))
        def done(s: dict[str, Any]) -> bool:
            rej = s["rejections"]
            if scenario == "rogue":
                return all(refusal_found(rej, layer=lay, reason=why, order_type=ot) for lay, why, ot in ROGUE_REFUSALS) \
                    and refusal_found(rej, layer="[LatIntentBridgeAlgorithm]", reason="halted")
            filled = any(e.get("status") == "Filled" and e.get("client_order_id") == buy.client_order_id for e in s["events"])
            return filled and all(expectation_met(rej, x) for x in expectations if x.outcome == "refuse")

        eng = launch_engine(run_dir, cfg, token, stub, done, exit_after_s=exit_after_s, timeout_s=timeout_s)
        snap = stub.snapshot()

    stdout_path, started, completed, exit_code = eng.stdout_path, eng.started, eng.completed, eng.exit_code
    log_text = eng.log_text
    report = {
        "run_id": run_id,
        "scenario": scenario,
        "execution_label": "SIMULATED",
        "engine": "LEAN (self-hosted Launcher, live mode) + GatedPaperBrokerage + LatSimulatedDataQueue",
        "lean_commit": lat_lean.pins()["lean"]["commit"],
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "completed": completed,
        "exit_code": exit_code,
        # LEAN's Launcher exits 1 whenever the algorithm status is not Completed; a live algorithm ending via Quit()
        # after lat-exit-after-seconds is 'Stopped', so a clean live stop is exit 1 + the LAT stop message.
        "clean_stop": ("configured run time" in log_text and "Algorithm state changed to Stopped" in log_text),
        "expectations": [{"label": x.label, "client_order_id": x.intent.client_order_id, "intent_id": x.intent.intent_id,
                          "quantity": format(x.intent.quantity, "f"), "limit_price": format(x.intent.limit_price, "f"),
                          "side": x.intent.side.value, "expected": x.outcome, "reason_contains": x.reason,
                          "keyed": x.keyed}
                         for x in expectations],
        "stub": snap,
        "assembly_check": [ln.split("LAT-ASSEMBLY-CHECK", 1)[1].strip() for ln in log_text.splitlines()
                           if "LAT-ASSEMBLY-CHECK" in ln],
        "gate_log": [ln for ln in log_text.splitlines() if "LAT-GATE-REFUSED" in ln or "ROGUE" in ln or "HALTED" in ln][:50],
        "process_sockets": eng.sockets,
        "non_loopback_sockets": non_loopback(eng.sockets),
        "config_path": str(cfg.relative_to(lat_lean.ROOT)),
        "launcher_log": str(stdout_path.relative_to(lat_lean.ROOT)),
    }
    (run_dir / "live_sim_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="bridge", choices=["bridge", "rogue"])
    ap.add_argument("--exit-after", type=int, default=40)
    args = ap.parse_args(argv)
    rep = run(args.scenario, exit_after_s=args.exit_after)
    s = rep["stub"]
    print(f"[SIMULATED] LEAN live-sim {rep['run_id']} scenario={rep['scenario']} completed={rep['completed']} "
          f"exit={rep['exit_code']}")
    for e in s["events"]:
        print(f"  event  {e.get('client_order_id')} lean_order={e.get('lean_order_id')} status={e.get('status')} "
              f"fill={e.get('fill_quantity')}@{e.get('fill_price')} broker_id={e.get('broker_order_id')}")
    for r in s["rejections"]:
        print(f"  reject {r.get('client_order_id')}: {str(r.get('reason'))[:160]}")
    print(f"  heartbeats={len(s['heartbeats'])} last_status={s['heartbeats'][-1].get('status') if s['heartbeats'] else None} "
          f"auth_failures={s['auth_failures']} protocol_errors={len(s['protocol_errors'])}")
    for a in rep["assembly_check"]:
        print(f"  assembly {a[:180]}")
    print(f"  sockets sampled={len(rep['process_sockets'])} non_loopback={rep['non_loopback_sockets']}")
    print(f"  report trading/lean/runs/{rep['run_id']}/live_sim_report.json")
    return 0 if rep["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
