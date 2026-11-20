"""LlmInvocationGate: selected candidate only, once per symbol per interval, daily run/cost budget, never per tick."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from packages.core.clock import ManualClock
from packages.core.db import Database
from services.ai.config import AiConfig
from services.ai.gating import LlmInvocationGate

from .conftest import T0


def _gate(
    cfg: AiConfig, *, transport: str = "real", db: Database | None = None, decision: ManualClock | None = None,
    wall: ManualClock | None = None,
) -> tuple[LlmInvocationGate, ManualClock, ManualClock]:
    d = decision or ManualClock(T0)
    w = wall or ManualClock(T0)
    return LlmInvocationGate(cfg, decision_clock=d, wall_clock=w, db=db, transport=transport), d, w


def test_never_without_selected_candidate() -> None:
    g, *_ = _gate(AiConfig())
    d = g.acquire("SPY", "cz", candidate_selected=False)
    assert not d.allowed and "never invoked per tick" in d.reason


def test_ticks_within_interval_are_denied_on_both_clocks() -> None:
    cfg = AiConfig(min_interval_per_symbol_s=900, min_wall_interval_per_symbol_s=30, daily_budget_usd=Decimal("100"))
    g, dclock, wclock = _gate(cfg)
    first = g.acquire("SPY", "cz1")
    assert first.allowed and first.reserved_usd == cfg.est_max_cost_per_run_usd
    g.settle(first, Decimal("0.10"))
    # a burst of ticks one second apart never re-invokes the model
    reasons = []
    for i in range(50):
        dclock.advance(1)
        wclock.advance(1)
        d = g.acquire("SPY", f"tick{i}")
        assert not d.allowed
        reasons.append(d.reason)
    assert "decision clock" in reasons[-1]
    # decision interval passed, but the wall-clock interval (fast replay) still applies
    dclock.advance(900)
    wclock.set(T0 + timedelta(seconds=10))
    assert "wall clock" in g.acquire("SPY", "cz2").reason
    wclock.advance(60)
    assert g.acquire("SPY", "cz3").allowed
    # other symbols are independent
    assert g.check("QQQ").allowed


def test_in_flight_run_blocks_concurrent_review() -> None:
    g, *_ = _gate(AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0))
    a = g.acquire("SPY", "cz1")
    b = g.acquire("SPY", "cz2")
    assert a.allowed and not b.allowed and "in flight" in b.reason
    g.settle(a, Decimal(0))
    assert g.acquire("SPY", "cz3").allowed


def test_daily_run_limit_and_rollover() -> None:
    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0, max_runs_per_day=2,
                   daily_budget_usd=Decimal("100"))
    g, _d, wall = _gate(cfg)
    for sym in ("SPY", "QQQ"):
        g.settle(g.acquire(sym, "cz"), Decimal("0.01"))
    denied = g.acquire("AAPL", "cz")
    assert not denied.allowed and "daily model-run limit" in denied.reason
    wall.advance(hours=24)  # next budget day (America/New_York)
    assert g.acquire("AAPL", "cz").allowed


def test_daily_cost_budget_is_pessimistic() -> None:
    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0, daily_budget_usd=Decimal("2.00"),
                   est_max_cost_per_run_usd=Decimal("1.50"))
    g, *_ = _gate(cfg)
    a = g.acquire("SPY", "cz1")
    assert a.allowed
    # while SPY's reservation is outstanding a second run cannot fit
    b = g.acquire("QQQ", "cz2")
    assert not b.allowed and "budget exhausted" in b.reason
    g.settle(a, Decimal("0.40"))
    c = g.acquire("QQQ", "cz3")
    assert c.allowed
    g.settle(c, None)  # unknown actual cost -> charged at the reservation
    assert Decimal(g.status()["spent_today_usd"]) == Decimal("1.90")
    assert not g.acquire("AAPL", "cz4").allowed


def test_fake_transport_reserves_nothing_but_still_counts_runs() -> None:
    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=0, daily_budget_usd=Decimal(0),
                   max_runs_per_day=1)
    g, *_ = _gate(cfg, transport="fake")
    a = g.acquire("SPY", "cz")
    assert a.allowed and a.reserved_usd == 0
    g.settle(a, Decimal(0))
    assert not g.acquire("QQQ", "cz").allowed  # run cap applies to fake runs too
    real, *_ = _gate(cfg, transport="real")
    assert not real.acquire("SPY", "cz").allowed  # zero budget => no paid runs at all


def test_ledger_survives_restart_and_unsettled_reservation_counts(tmp_path) -> None:
    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=600, daily_budget_usd=Decimal("3.00"),
                   est_max_cost_per_run_usd=Decimal("1.50"))
    wall = ManualClock(T0)
    with Database(tmp_path / "g.sqlite") as db:
        g, *_ = _gate(cfg, db=db, wall=wall)
        a = g.acquire("SPY", "cz1")
        g.settle(a, Decimal("0.25"))
        crashed = g.acquire("QQQ", "cz2")  # the process "dies" before settling
        assert crashed.allowed
        rows = db.query("SELECT status FROM ai_invocations ORDER BY wall_ts")
        assert [r["status"] for r in rows] == ["SETTLED", "RESERVED"]
    with Database(tmp_path / "g.sqlite") as db2:
        wall.advance(60)
        g2, *_ = _gate(cfg, db=db2, wall=wall)
        st = g2.status()
        assert st["runs_today"] == 2 and Decimal(st["spent_today_usd"]) == Decimal("1.75")
        assert not g2.acquire("AAPL", "cz3").allowed  # 1.75 + 1.50 > 3.00
        assert "wall clock" in g2.acquire("SPY", "cz4").reason  # per-symbol wall interval reloaded
        denied = db2.query("SELECT COUNT(*) n FROM ai_invocations WHERE status='DENIED'")[0]["n"]
        assert denied == 2
