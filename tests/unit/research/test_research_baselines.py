"""Baselines + point-in-time backtester."""

from __future__ import annotations

import itertools
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.research.baselines import (
    BuyAndHold,
    FilteredMomentum,
    IntradayMeanReversion,
    Momentum,
    compute_metrics,
    default_strategies,
    run_all_baselines,
    run_backtest,
)
from services.research.costs import CostModel, FeeSchedule, LiquidityBucket, SlippageModel, SpreadModel
from services.research.frames import BarDataError, prepare_bars

from .bar_factory import bar_row, frame, make_bars, ts

T0 = pd.Timestamp("2024-03-04 14:30", tz="UTC")  # 09:30 America/New_York


def zero_cost_model() -> CostModel:
    return CostModel(
        fees=FeeSchedule(sec_fee_rate=Decimal(0), finra_taf_per_share=Decimal(0)),
        spread=SpreadModel(buckets=(LiquidityBucket(Decimal(0), Decimal(0), "zero"),), unknown_liquidity_spread_bps=Decimal(0)),
        slippage=SlippageModel(fixed_bps=Decimal(0), impact_coef_bps=Decimal(0), unknown_adv_impact_bps=Decimal(0)),
    )


def _minute_frame(opens: list[str], *, latency_s: float = 0.0) -> pd.DataFrame:
    rows = [
        bar_row("AAA", T0 + pd.Timedelta(minutes=i), o=o, c=o, latency_s=latency_s) for i, o in enumerate(opens)
    ]
    return prepare_bars(frame(rows))


def test_fill_at_next_bar_open_when_data_is_immediately_available() -> None:
    bars = _minute_frame(["10", "11", "12", "13", "14", "15"])
    res = run_backtest(bars, np.ones(len(bars), dtype=np.int8), zero_cost_model(), strategy="t")
    buy = res.trades[0]
    assert buy.side == "BUY" and buy.decision_index == 0 and buy.bar_index == 1
    assert buy.price == Decimal("11")
    assert buy.quantity == Decimal(100_000 // 11)


def test_fill_waits_for_first_bar_opening_after_available_at() -> None:
    # bar i is only knowable 90 s after it ends -> decision at T0+150s; bars 1 (T0+60s) and 2 (T0+120s)
    # had already opened, so the first executable open is bar 3.
    bars = _minute_frame(["10", "11", "12", "13", "14", "15"], latency_s=90)
    res = run_backtest(bars, np.ones(len(bars), dtype=np.int8), zero_cost_model(), strategy="t")
    buy = res.trades[0]
    assert buy.bar_index == 3 and buy.price == Decimal("13")
    assert buy.ts >= buy.decision_ts


def test_zero_cost_buy_and_hold_exact_accounting() -> None:
    bars = _minute_frame(["10", "11", "12", "13", "14", "15"])
    res = run_backtest(bars, np.ones(len(bars), dtype=np.int8), zero_cost_model(), strategy="bh")
    qty = Decimal(100_000 // 11)
    final = Decimal(100_000) - qty * 11 + qty * 15
    assert res.equity[-1] == final
    assert Decimal(res.metrics["final_equity"]) == final.quantize(Decimal("0.01"))
    assert res.trades[-1].reason == "terminal_liquidation"
    assert res.metrics["total_costs"] == "0.00"


def test_costs_reduce_returns_and_accounting_identity_holds() -> None:
    bars = prepare_bars(make_bars(("AAA",), timeframe="1Day", n=250, seed=3))
    tgt = Momentum(lookback=10).targets(bars)
    free = run_backtest(bars, tgt, zero_cost_model(), strategy="m")
    costly = run_backtest(bars, tgt, CostModel(), strategy="m")
    assert len(costly.trades) > 4
    assert Decimal(costly.metrics["total_return"]) < Decimal(free.metrics["total_return"])
    # flat at the end: final equity = initial + sum of realised P&L (net of all costs), exactly
    realised = sum((t.realized_pl for t in costly.trades if t.realized_pl is not None), Decimal(0))
    assert costly.equity[-1] == Decimal(100_000) + realised
    total_cost = sum((t.cost_usd for t in costly.trades), Decimal(0))
    assert Decimal(costly.metrics["total_costs"]) == total_cost.quantize(Decimal("0.01"))
    assert all(isinstance(t.price, Decimal) and isinstance(t.cost_usd, Decimal) for t in costly.trades)
    # fees only on SELL legs
    assert all(t.fees_usd == 0 for t in costly.trades if t.side == "BUY")
    assert all(t.fees_usd > 0 for t in costly.trades if t.side == "SELL")


def test_long_only_one_position_alternating_legs() -> None:
    bars = prepare_bars(make_bars(("AAA",), timeframe="1Day", n=250, seed=5))
    res = run_backtest(bars, Momentum(lookback=5).targets(bars), CostModel(), strategy="m")
    sides = [t.side for t in res.trades]
    assert sides[0] == "BUY"
    assert all(a != b for a, b in itertools.pairwise(sides))
    buys = [t for t in res.trades if t.side == "BUY"]
    assert all(t.quantity > 0 and t.quantity == t.quantity.to_integral_value() for t in buys)
    # no leverage: before every buy the book is flat, so equity at the previous close is pure cash and the
    # buy's notional + costs never exceeds it; cash never goes negative afterwards either.
    for t in buys:
        cash_before = res.equity[t.bar_index - 1]
        assert t.notional + t.cost_usd <= cash_before
        # and it is not trivially small: the position uses most of the cash (whole shares, costs included)
        assert cash_before - t.notional - t.cost_usd < 2 * t.price + t.cost_usd


def test_targets_reject_short_or_leverage() -> None:
    bars = _minute_frame(["10", "11", "12"])
    with pytest.raises(ValueError, match="long-only"):
        run_backtest(bars, np.array([1, -1, 0]), CostModel(), strategy="x")
    with pytest.raises(ValueError, match="long-only"):
        run_backtest(bars, np.array([2, 0, 0]), CostModel(), strategy="x")


@pytest.mark.parametrize(("timeframe", "n"), [("1Day", 200), ("5Min", 6)])
def test_strategy_targets_are_causal(timeframe: str, n: int) -> None:
    """Target at t is unchanged by truncating or corrupting everything after t (no look-ahead)."""
    bars = prepare_bars(make_bars(("AAA",), timeframe=timeframe, n=n, seed=11))
    k = len(bars) * 2 // 3
    future_mut = bars.copy()
    for col in ("open", "high", "low", "close"):
        future_mut.loc[k:, col] = [v * 3 for v in future_mut.loc[k:, col]]
        future_mut.loc[k:, f"{col}_f"] = future_mut.loc[k:, f"{col}_f"] * 3
    future_mut.loc[k:, "volume_f"] = future_mut.loc[k:, "volume_f"] * 50
    strategies = [*default_strategies(timeframe), Momentum(lookback=3), FilteredMomentum(min_adv_usd=1.0)]
    for strat in strategies:
        ok, _ = strat.applicable(timeframe)
        if not ok:
            continue
        full = strat.targets(bars)
        prefix = strat.targets(bars.iloc[:k].reset_index(drop=True))
        mutated = strat.targets(future_mut)
        assert np.array_equal(full[:k], prefix), strat.name
        assert np.array_equal(full[:k], mutated[:k]), strat.name


def test_backtest_is_prefix_invariant() -> None:
    bars = prepare_bars(make_bars(("AAA",), timeframe="1Day", n=220, seed=13))
    tgt = Momentum(lookback=10).targets(bars)
    full = run_backtest(bars, tgt, CostModel(), strategy="m")
    k = 150
    pre = run_backtest(bars.iloc[:k].reset_index(drop=True), tgt[:k], CostModel(), strategy="m")
    pre_trades = [(t.bar_index, t.side, t.quantity, t.price) for t in pre.trades if t.reason != "terminal_liquidation"]
    full_trades = [(t.bar_index, t.side, t.quantity, t.price) for t in full.trades if t.bar_index < k - 1]
    assert pre_trades == full_trades[: len(pre_trades)]
    assert pre.equity[: k - 1] == full.equity[: k - 1]


def test_intraday_mean_reversion_is_flat_by_close() -> None:
    bars = prepare_bars(make_bars(("AAA", "BBB"), timeframe="1Min", n=4, seed=21, vol=0.02))
    for sym in ("AAA", "BBB"):
        sb = bars.loc[bars["symbol"] == sym].reset_index(drop=True)
        strat = IntradayMeanReversion(window=20, entry_z=1.5)
        res = run_backtest(sb, strat.targets(sb), CostModel(), strategy=strat.name)
        assert res.trades, "fixture should trigger mean-reversion entries"
        assert res.diagnostics["overnight_holds"] == 0
        days = list(sb["session_date"])
        for buy, sell in zip(res.trades[::2], res.trades[1::2], strict=False):
            assert buy.side == "BUY" and sell.side == "SELL"
            assert days[buy.bar_index] == days[sell.bar_index]


def test_intraday_entry_never_fills_in_a_later_session() -> None:
    # day 1 data stops at 15:52 ET (gap before the close); an entry decided on the last day-1 bar must not
    # fill at the day-2 open when DAY semantics are on, while a plain backtest (momentum convention) fills it.
    day1 = [bar_row("AAA", pd.Timestamp("2024-03-04 20:50", tz="UTC") + pd.Timedelta(minutes=i)) for i in range(2)]
    day2 = [bar_row("AAA", pd.Timestamp("2024-03-05 14:30", tz="UTC") + pd.Timedelta(minutes=i), o="101", c="101")
            for i in range(3)]
    bars = prepare_bars(frame(day1 + day2))
    tgt = np.array([0, 1, 0, 0, 0], dtype=np.int8)  # entry signal on the last day-1 bar only
    plain = run_backtest(bars, tgt, zero_cost_model(), strategy="x")
    assert plain.trades and plain.trades[0].side == "BUY" and plain.trades[0].bar_index == 2
    bounded = run_backtest(bars, tgt, zero_cost_model(), strategy="x", entries_expire_at_session_end=True)
    assert bounded.trades == []
    assert bounded.diagnostics["expired_entry_orders"] == 1
    # exits are never expired: a position held into a gap is closed at the next session's first open
    hold = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    res = run_backtest(bars, hold, zero_cost_model(), strategy="x", entries_expire_at_session_end=True)
    assert [(t.side, t.bar_index) for t in res.trades] == [("BUY", 1), ("SELL", 2)]
    assert IntradayMeanReversion().entries_expire_at_session_end is True


def test_mean_reversion_not_applicable_to_daily() -> None:
    ok, why = IntradayMeanReversion().applicable("1Day")
    assert not ok and "intraday" in why


def test_metrics_drawdown_sharpe_turnover() -> None:
    equity = [Decimal(100), Decimal(120), Decimal(90), Decimal(130)]
    end_ts = [ts(pd.Timestamp("2024-01-02 21:00", tz="UTC") + pd.Timedelta(days=i)) for i in range(4)]
    days = [t.date() for t in end_ts]
    m = compute_metrics(
        equity=equity, end_ts=end_ts, session_dates=days, initial_cash=Decimal(100), timeframe="1Day", trades=[],
        totals={"cost": Decimal(0), "fees": Decimal(0), "spread": Decimal(0), "slippage": Decimal(0),
                "notional": Decimal(220)},
        bars_in_market=2, first_start=ts(end_ts[0] - pd.Timedelta(hours=6, minutes=30)),
    )
    assert m["max_drawdown"] == "0.250000"  # (120 - 90) / 120
    assert m["total_return"] == "0.300000"
    assert m["turnover"] == "2.000000"  # 220 / mean equity 110
    assert m["exposure"] == "0.5000"
    assert "sqrt(252)" in m["sharpe_annualization"]
    daily = [0.0, 0.2, -0.25, 130 / 90 - 1]
    expected = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)
    assert m["sharpe"] == f"{expected:.4f}"
    assert m["cagr"] is not None


def test_run_all_baselines_writes_provenance(tmp_path: Path) -> None:
    df = make_bars(("AAA", "BBB"), timeframe="1Min", n=3, seed=2, dataset_version="synthetic-test-v9")
    out = run_all_baselines(df, CostModel(), tmp_path, run_id="bl_test")
    run_dir = tmp_path / "bl_test"
    results = json.loads((run_dir / "results.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert results["dataset_version"] == "synthetic-test-v9" == manifest["dataset_version"]
    assert results["is_simulated"] is True
    assert results["code_version"].startswith("src-sha256:")
    assert "NO EDGE CLAIMED" in results["no_edge_note"]
    assert set(results["parameters"]["strategies"]) == {"buy_and_hold", "momentum", "intraday_mean_reversion",
                                                       "filtered_momentum"}
    assert results["cost_model"]["version"] == "cost-model-v1"
    assert len(out["summary_files"]) == 8
    summary = json.loads((run_dir / out["summary_files"][0]).read_text())
    for key in ("run_id", "engine", "strategy", "dataset", "is_simulated", "started_at", "metrics", "equity_curve",
                "source_path", "notes"):
        assert key in summary
    assert summary["engine"] == "PYTHON_BASELINE" and summary["is_simulated"] is True
    assert "SIMULATED" in summary["notes"]
    for key in ("total_return", "sharpe", "max_drawdown", "trades", "fees"):
        assert key in summary["metrics"]
    for run in results["runs"]:
        for key in ("total_return", "max_drawdown", "turnover", "trades", "total_costs", "exposure", "sharpe"):
            assert key in run["metrics"]


def test_daily_run_skips_intraday_strategy_with_reason(tmp_path: Path) -> None:
    out = run_all_baselines(make_bars(("AAA",), n=120), CostModel(), tmp_path, run_id="bl_daily")
    statuses = {(r["strategy"], r["status"]) for r in out["runs"]}
    assert ("intraday_mean_reversion", "SKIPPED") in statuses
    assert ("buy_and_hold", "OK") in statuses
    bh = next(r for r in out["runs"] if r["strategy"] == "buy_and_hold")
    assert bh["metrics"]["cagr"] is not None


def test_bad_input_fails_closed() -> None:
    good = make_bars(("AAA",), timeframe="1Day", n=30)
    naive = good.copy()
    naive["start_ts"] = naive["start_ts"].dt.tz_localize(None)
    with pytest.raises(BarDataError, match=r"naive|tz"):
        prepare_bars(naive)
    leak = good.copy()
    leak.loc[5, "available_at_ts"] = leak.loc[5, "end_ts"] - pd.Timedelta(seconds=1)
    with pytest.raises(BarDataError, match="look-ahead"):
        prepare_bars(leak)
    dup = pd.concat([good, good.iloc[[3]]], ignore_index=True)
    with pytest.raises(BarDataError, match="duplicate"):
        prepare_bars(dup)
    gap = good.copy()
    gap.loc[4, "close"] = None
    with pytest.raises(BarDataError, match="null"):
        prepare_bars(gap)


def test_float_prices_converted_exactly_at_io_boundary() -> None:
    bars = prepare_bars(make_bars(("AAA",), n=10, price_type="float"))
    assert all(isinstance(v, Decimal) for v in bars["close"])
    assert all(v == v.quantize(Decimal("0.01")) for v in bars["close"])


def test_buy_and_hold_single_entry() -> None:
    bars = prepare_bars(make_bars(("AAA",), n=50))
    res = run_backtest(bars, BuyAndHold().targets(bars), CostModel(), strategy="bh")
    assert [t.side for t in res.trades] == ["BUY", "SELL"]
    assert res.trades[0].bar_index == 1
