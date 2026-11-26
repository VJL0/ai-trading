"""Failure injection through the replay of the real stack (SIMULATED, synthetic bars, disposable workspaces):
a gap through the stop, a partial fill followed by a cancel/fill race, a restart in the middle of a replay that
resumes from the journal, and an ambiguous submission reconciled by client order id (never resubmitted)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import OrderEventType
from packages.contracts.policy import RiskPolicy
from services.replay.broker import FillModel
from services.replay.engine import ReplayEngine, ReplayHooks
from services.replay.parity import compare
from services.replay.snapshot import BarEvent
from tests.integration.replay.replay_support import config, of, replay, snapshot, trade_key

GAP_SLOT = 8  # the 10:10 New York bar of the first replayed session (entry filled at 10:00:01)


@pytest.mark.parametrize(("gap", "close_reason", "category"), [(-0.0175, "STOP", "STOP"), (-0.03, "STOP_BREACH", "STOP")])
async def test_gap_through_the_stop(tmp_path: Path, gap: float, close_reason: str, category: str) -> None:
    snap = snapshot(sessions=2, sigma_bps=1.0, events=(BarEvent(1, GAP_SLOT, "SPY", gap_pct=gap),))
    engine, res = await replay(tmp_path, snap, config())
    (trade,) = res.summary["trades"]
    stop = Decimal(trade["stop_price"])
    protective_limit = stop * Decimal("0.995")
    exit_px = Decimal(trade["exit_avg_price"])
    assert trade["exit_at"].startswith("2024-03-05T15:10")  # exits within the gap bar, never waits for the horizon
    if close_reason == "STOP":
        # opened between the stop and its limit: the broker-held STOP_LIMIT itself fills (below the stop)
        assert trade["stop_fill"] and trade["close_reason"] == "STOP" and protective_limit <= exit_px < stop
    else:
        # gapped through the limit: the resting stop cannot fill; the manager escalates to a marketable exit
        assert not trade["stop_fill"] and trade["exit_reason"] == "STOP_BREACH" and exit_px < protective_limit
        assert any(e["action"] == "EXIT_TRIGGERED" and e["purpose"] == "STOP_BREACH" for e in of(res.records, "EXIT"))
    rep = compare(res.summary, snap, engine.strategy, engine.days, RiskPolicy(), Decimal(5000))
    (m,) = rep["trades"]
    assert m["exit_reason_research"] == "STOP_GAP" and m["exit_category_replay"] == category
    assert abs(m["exit_delta_bps"]) <= 10.0


async def test_partial_fill_then_cancel_race(tmp_path: Path) -> None:
    async def race(engine: ReplayEngine, _at: datetime) -> None:
        engine.rt.broker.injector.fill_on_cancel = 1  # the next cancel (the entry remainder) races a fill

    snap = snapshot(sessions=2)
    engine, res = await replay(tmp_path, snap, config(fill=FillModel(entry_partial_prob=1.0)),
                               hooks=ReplayHooks(before_decision=race))
    recs = res.records
    (trade,) = res.summary["trades"]
    entry = trade["entry_quantity"]
    assert entry == "2" and trade["entry_fills"] == 2 and trade["exit_quantity"] == "2"
    assert [m["outcome"] for m in of(recs, "ENTRY_MATCH")] == ["PARTIAL"]
    assert any(c["purpose"] == "ENTRY_TIMEOUT" for c in of(recs, "CANCEL"))
    placed = [s for s in of(recs, "STOP") if s["outcome"] == "SUBMITTED"]
    assert placed[0]["action"] == "PROTECT"  # the filled share was protected before the race
    stop_orders = [o for o in of(recs, "INTENT") if o["order_type"] == "STOP_LIMIT"]
    assert sum(Decimal(o["quantity"]) for o in stop_orders) >= Decimal(2)  # protection followed the race fill
    assert not res.summary["order_rejections"] and trade["exit_reason"] == "TIME"
    assert engine.rt.ledger.position_quantities().get("SPY", Decimal(0)) == 0


async def test_cancelled_entry_with_a_fill_recovered_by_reconciliation_is_costed(tmp_path: Path) -> None:
    async def drop_partial(engine: ReplayEngine, _at: datetime) -> None:
        engine.rt.broker.injector.drop_event_types = frozenset({OrderEventType.PARTIAL_FILL})

    _, result = await replay(tmp_path, snapshot(sessions=2), config(fill=FillModel(entry_partial_prob=1.0)),
                             hooks=ReplayHooks(before_decision=drop_partial))
    (entry,) = [f for f in of(result.records, "FILL") if f["side"] == "BUY"]
    assert any(o["client_order_id"] == entry["client_order_id"] and o["to_state"] == "CANCELED"
               for o in of(result.records, "ORDER"))
    (cost,) = [c for c in of(result.records, "LEG_COST") if c["client_order_id"] == entry["client_order_id"]]
    assert cost["quantity"] == entry["quantity"]


async def test_restart_mid_replay_resumes_from_the_journal(tmp_path: Path) -> None:
    snap = snapshot(sessions=3)
    _, baseline = await replay(tmp_path / "a", snap, config())
    crash_at = datetime(2024, 3, 5, 15, 30, tzinfo=UTC)  # 10:30 New York: holding a protected position
    engine, partial = await replay(tmp_path / "b", snap, config(), stop_after=crash_at)
    assert not partial.completed and engine.rt.ledger.position_quantities().get("SPY") == Decimal(2)
    engine.abandon()  # simulated crash: process objects dropped, workspace and journal left as they are
    engine2, resumed = await replay(tmp_path / "b", snap, config(), resume=True)
    assert resumed.completed
    assert [trade_key(t) for t in resumed.summary["trades"]] == [trade_key(t) for t in baseline.summary["trades"]]
    recs = resumed.records
    assert len(of(recs, "RUN_START")) == 1 and len(of(recs, "RESUMED")) == 1
    assert [r["seq"] for r in recs] == sorted({r["seq"] for r in recs})
    assert [r["status"] for r in of(recs, "STARTUP_RECONCILE")] == ["CLEAN", "CLEAN"]
    fills = [r["fill_id"] for r in of(recs, "FILL")]
    assert len(fills) == len(set(fills)) == 4
    plans = [r["plan_id"] for r in of(recs, "ROUND_TRIP")]
    assert len(plans) == len(set(plans)) == 2
    marks = [r["at"] for r in of(recs, "MARK") if not r["session_end"]]
    assert len(marks) == len(set(marks)) == 2 * 78  # every bar of both sessions exactly once (none replayed twice)
    orders = engine2.rt.db.query("SELECT client_order_id FROM orders")
    assert len({o["client_order_id"] for o in orders}) == len(orders) == len(of(recs, "INTENT"))


async def test_ambiguous_submission_is_reconciled_by_client_order_id_never_resubmitted(tmp_path: Path) -> None:
    async def ambiguous(engine: ReplayEngine, _at: datetime) -> None:
        if not of(engine.journal.records(), "DECISION"):
            inj = engine.rt.broker.injector
            inj.timeout_after_accept = 1  # the broker accepts, the response is lost
            inj.fail_lookups = 1  # and the immediate lookup by client order id fails too

    snap = snapshot(sessions=2)
    engine, res = await replay(tmp_path, snap, config(), hooks=ReplayHooks(before_decision=ambiguous))
    recs = res.records
    entry = of(recs, "DECISION")[0]["client_order_id"]
    states = [o["to_state"] for o in of(recs, "ORDER") if o["client_order_id"] == entry]
    assert "UNKNOWN" in states and states[-1] == "FILLED"
    broker = engine.rt.broker
    assert broker.submission_count(entry) == 1 and broker.accepted_submissions.count(entry) == 1
    entry_intents = [i for i in of(recs, "INTENT") if i["side"] == "BUY"]
    assert [i["client_order_id"] for i in entry_intents] == [entry]
    (trade,) = res.summary["trades"]
    assert trade["exit_reason"] == "TIME" and trade["entry_quantity"] == "2"


async def test_restart_after_journal_tail_before_mark_does_not_duplicate_fills(tmp_path: Path) -> None:
    crash_at = datetime(2024, 3, 5, 15, 5, tzinfo=UTC)

    async def crash(engine: ReplayEngine, at: datetime) -> None:
        if at == crash_at:
            assert of(engine.journal.records(), "FILL")
            raise RuntimeError("injected before mark")

    snap = snapshot(sessions=2)
    with pytest.raises(RuntimeError, match="injected before mark"):
        await replay(tmp_path, snap, config(), hooks=ReplayHooks(after_bar=crash))
    _, resumed = await replay(tmp_path, snap, config(), resume=True)
    for kind, key in (("FILL", "fill_id"), ("LEG_COST", "client_order_id"), ("ORDER", "source_key")):
        keys = [r[key] for r in of(resumed.records, kind)]
        assert len(keys) == len(set(keys))


async def test_restart_at_close_runs_session_close_once(tmp_path: Path) -> None:
    close = datetime(2024, 3, 5, 21, 0, tzinfo=UTC)
    snap = snapshot(sessions=2)
    engine, partial = await replay(tmp_path, snap, config(), stop_after=close)
    assert not of(partial.records, "SESSION_END")
    engine.abandon()
    _, resumed = await replay(tmp_path, snap, config(), resume=True)
    assert len(of(resumed.records, "SESSION_END")) == 1
    assert resumed.summary["sessions"] == 1


async def test_resume_rejects_different_run_config_before_opening_state(tmp_path: Path) -> None:
    snap = snapshot(sessions=2)
    engine, partial = await replay(tmp_path, snap, config(), stop_after=datetime(2024, 3, 5, 15, 5, tzinfo=UTC))
    engine.abandon()
    journal = partial.journal_path.read_bytes()
    with pytest.raises(ValueError, match="original"):
        await replay(tmp_path, snap, config(equity="6000"), resume=True)
    assert partial.journal_path.read_bytes() == journal


async def test_resume_after_run_end_regenerates_missing_summary_without_replay(tmp_path: Path) -> None:
    snap = snapshot(sessions=2)
    _, finished = await replay(tmp_path, snap, config())
    assert finished.summary_path is not None
    journal = finished.journal_path.read_bytes()
    summary = finished.summary_path.read_bytes()
    finished.summary_path.unlink()
    _, resumed = await replay(tmp_path, snap, config(), resume=True)
    assert resumed.completed and resumed.summary_path is not None
    assert resumed.journal_path.read_bytes() == journal
    assert resumed.summary_path.read_bytes() == summary
