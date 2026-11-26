"""End to end: synthetic sessions replayed through the real DecisionCycle, RiskEngine, signed intents, OrderLedger,
ExecutionService/final gate, FakeBroker (fill model) and PositionManager. SIMULATED: no network, no broker."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import Side
from packages.contracts.models import RiskDecision
from packages.contracts.policy import RiskPolicy
from packages.core.signing import IntentSigner
from services.replay.broker import FillModel
from services.replay.engine import ReplayEngine, ReplayHooks
from services.replay.parity import compare
from services.replay.snapshot import BarSnapshot
from services.risk.engine import REQUIRED_TRADE_CHECKS
from tests.integration.replay.replay_support import config, of, replay, snapshot, trade_key


async def test_synthetic_sessions_through_the_real_stack(tmp_path: Path) -> None:
    snap = snapshot(sessions=3, symbols=("QQQ", "SPY"))
    engine, res = await replay(tmp_path, snap, config("tod-SPY-1000-60m"))
    recs = res.records
    assert res.completed and res.summary_path is not None and res.summary_path.is_file()
    assert all(r["execution_label"] == "REPLAY" and r["is_simulated"] for r in recs)
    assert all(r["provenance"]["dataset_version"] == snap.provenance.dataset_version for r in recs)
    assert [d["outcome"] for d in of(recs, "DECISION")] == ["SUBMITTED", "SUBMITTED"]
    # every order was evaluated by the real risk engine with every required check, then signed and gated
    db = engine.rt.db
    rows = db.query("SELECT payload_json FROM risk_decisions")
    decisions = [RiskDecision.model_validate_json(r["payload_json"]) for r in rows]
    assert decisions and all(d.approved for d in decisions)
    assert all({c.name for c in d.checks} >= REQUIRED_TRADE_CHECKS for d in decisions)
    intents = of(recs, "INTENT")
    assert len(intents) == len(decisions) and all(i["signed"] and i["key_id"] == engine.rt.signer.key_id
                                                  and i["mode"] == "REPLAY" for i in intents)
    orders = db.query("SELECT client_order_id, mode, claimed_by, state, order_type FROM orders")
    assert orders and all(o["mode"] == "REPLAY" and o["client_order_id"].startswith("lat-replay-") for o in orders)
    submitted = [o for o in orders if o["claimed_by"]]
    assert all(o["claimed_by"] == "execution:FAKE" for o in submitted)
    assert all(engine.rt.broker.submission_count(o["client_order_id"]) == 1 for o in submitted)
    # the position manager protected every entry with a broker-held STOP_LIMIT and exited on the horizon
    stops = [s for s in of(recs, "STOP") if s["outcome"] == "SUBMITTED"]
    assert len(stops) == 2 and {o["order_type"] for o in orders} == {"LIMIT", "STOP_LIMIT"}
    trades = res.summary["trades"]
    assert len(trades) == 2 and {t["exit_reason"] for t in trades} == {"TIME"}
    assert all(t["symbol"] == "SPY" and t["entry_quantity"] == "2" for t in trades)  # runtime sizing: $1000 / ~$495
    assert len(of(recs, "LEG_COST")) == 4  # once per filled leg
    fills = {f["client_order_id"]: f for f in of(recs, "FILL")}
    for leg in of(recs, "LEG_COST"):
        fill = fills[leg["client_order_id"]]
        day = datetime.fromisoformat(fill["at"]).date()
        adv = snap.adv_shares(fill["symbol"], day)
        assert adv is not None
        est = engine.rt.cost_model.estimate(fill["symbol"], Side(fill["side"]), fill["quantity"], fill["price"],
                                            adv_shares=Decimal(f"{adv:.0f}"))
        assert Decimal(leg["charged_usd"]) == est.fees_usd + est.slippage_usd
    assert engine.rt.ledger.position_quantities().get("SPY", Decimal(0)) == 0
    s = res.summary
    assert s["execution_label"] == "REPLAY" and s["provenance"] == engine.provenance
    for key in ("capital_exposure", "cagr", "sharpe", "max_drawdown", "turnover", "cost_drag", "rejections_by_check"):
        assert key in s
    assert s["sessions"] == 2 and s["capital_exposure"]["max"] > 0
    on_disk = json.loads(res.summary_path.read_text())
    assert on_disk["execution_label"] == "REPLAY" and on_disk["round_trips"] == 2


async def test_final_gate_verifies_intent_before_broker_claim(tmp_path: Path) -> None:
    async def replace_gate_verifier(engine: ReplayEngine, _at: datetime) -> None:
        engine.rt.verifier = IntentSigner.load_or_create(tmp_path / "other-key").verifier

    engine, res = await replay(tmp_path, snapshot(sessions=2), config(),
                               hooks=ReplayHooks(before_decision=replace_gate_verifier))
    assert [r["outcome"] for r in of(res.records, "DECISION")] == ["BLOCKED"]
    assert len(of(res.records, "INTENT")) == 1  # the ledger accepted the original signature
    assert any("signature" in reason for reason in res.summary["order_rejections"] if reason.startswith("final_gate"))
    assert engine.rt.broker.submission_count() == 0


async def test_replay_is_deterministic_with_a_seeded_fill_model(tmp_path: Path) -> None:
    snap = snapshot(sessions=5)
    fill = FillModel(seed=3, entry_nonfill_prob=0.4, entry_partial_prob=0.3)
    _, a = await replay(tmp_path / "a", snap, config(fill=fill))
    _, b = await replay(tmp_path / "b", snap, config(fill=fill))
    assert [trade_key(t) for t in a.summary["trades"]] == [trade_key(t) for t in b.summary["trades"]]
    assert a.summary["entry_matching"] == b.summary["entry_matching"]
    assert sum(a.summary["entry_matching"].values()) == 4


async def test_deterministic_strategy_agrees_with_the_research_simulator(tmp_path: Path) -> None:
    snap = snapshot(sessions=6)
    engine, res = await replay(tmp_path, snap, config("tod-SPY-1000-60m"))
    rep = compare(res.summary, snap, engine.strategy, engine.days, RiskPolicy(), Decimal(5000))
    assert rep["research_trades"] == rep["replay_trades"] == rep["matched"] == 5
    assert rep["all_within_tolerance"], rep
    assert rep["exit_reason_agreement"] == 1.0
    assert rep["entry_delta_bps"]["max_abs"] <= 2.0  # the synthesized half spread (2 bps MEGA bucket / 2) + a tick
    assert abs(rep["research_total_net_bps"] - rep["replay_total_net_bps"]) <= 5.0


async def test_eod_flatten_matches_research(tmp_path: Path) -> None:
    snap = snapshot(sessions=3)
    engine, res = await replay(tmp_path, snap, config("tod-SPY-1520-30m"))
    trades = res.summary["trades"]
    assert len(trades) == 2 and {t["exit_reason"] for t in trades} == {"EOD"}
    assert all(t["exit_at"][11:16] == "20:50" for t in trades)  # 15:50 New York (EST): 10 min before the close
    rep = compare(res.summary, snap, engine.strategy, engine.days, RiskPolicy(), Decimal(5000))
    assert rep["matched"] == 2 and rep["all_within_tolerance"], rep


async def test_equity_carries_last_observed_mid_across_missing_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = snapshot(sessions=2)
    bars = snap.bars
    gap = bars["t"].astype(str).str.startswith("2024-03-05 15:15:00")
    assert gap.sum() == 1
    incomplete = BarSnapshot.from_frame(bars.loc[~gap], source="missing-bar-fixture", feed=snap.provenance.feed)
    ticks: list[str] = []

    async def track_ticks(engine: ReplayEngine, _at: datetime) -> None:
        real_tick = engine._pm_tick

        async def tick() -> None:
            ticks.append(engine.rt.now().isoformat())
            await real_tick()

        monkeypatch.setattr(engine, "_pm_tick", tick)

    _, result = await replay(tmp_path, incomplete, config(), hooks=ReplayHooks(before_decision=track_ticks))
    marks = {r["at"][11:16]: r for r in of(result.records, "MARK") if r["at"].startswith("2024-03-05")}
    before, after = marks["15:15"], marks["15:20"]
    assert before["positions"] == after["positions"] == {"SPY": "2"}
    assert after["position_value"] == before["position_value"]
    assert Decimal(after["equity"]) == Decimal(after["cash"]) + Decimal(after["position_value"])
    assert "2024-03-05T15:16:01+00:00" in ticks
