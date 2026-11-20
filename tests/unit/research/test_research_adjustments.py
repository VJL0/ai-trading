"""Point-in-time corporate-action handling for RAW bars: splits, reverse splits, dividends, as-of filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.research.adjustments import apply_corporate_actions
from services.research.baselines import BuyAndHold, run_all_baselines, run_backtest
from services.research.costs import CostModel
from services.research.frames import NY, BarDataError, prepare_bars
from services.research.signals import QuantSignalEngine

from .bar_factory import bar_row, frame, make_bars, to_dt
from .test_research_baselines import zero_cost_model

SESSIONS = [pd.Timestamp(f"2024-03-{d:02d} 14:30", tz="UTC") for d in (4, 5, 6, 7, 8, 11, 12, 13)]


@dataclass(frozen=True)
class _Action:  # attribute-style record (like services.market_data CorporateAction)
    symbol: str
    action_type: str
    ex_date: date
    available_at_ts: datetime
    split_factor: Decimal | None = None
    cash_amount: Decimal | None = None


def _daily(closes: list[str], volume: str = "1000") -> pd.DataFrame:
    rows = [bar_row("AAA", s, minutes=390, o=c, c=c, v=volume, timeframe="1Day") for s, c in zip(SESSIONS, closes,
                                                                                                 strict=False)]
    return frame(rows)


def _open_utc(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)


def test_forward_split_is_backward_adjusted_exactly() -> None:
    # 2-for-1 split effective 2024-03-07: raw close falls 100 -> 50 overnight although nothing happened
    raw = _daily(["100", "102", "104", "52", "53", "54"])
    split = _Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5"))
    res = apply_corporate_actions(raw, [split])
    bars = prepare_bars(res.bars)
    assert list(bars["close"]) == [Decimal("50"), Decimal("51"), Decimal("52"), Decimal("52"), Decimal("53"),
                                   Decimal("54")]
    assert list(bars["volume"]) == [Decimal(2000)] * 3 + [Decimal(1000)] * 3
    assert all(isinstance(v, Decimal) for v in bars["open"])
    assert res.report["splits"] == {"AAA": [{"ex_date": "2024-03-07", "split_factor": "0.5"}]}
    # the raw input frame is untouched
    assert list(raw["close"])[:3] == [Decimal("100"), Decimal("102"), Decimal("104")]


def test_reverse_split_and_cumulative_factors() -> None:
    raw = _daily(["2", "2.1", "42", "43", "86", "88"])  # 1-for-20 reverse split, then a 1-for-2 reverse split
    acts = [
        {"symbol": "AAA", "action_type": "SPLIT", "ex_date": "2024-03-06", "available_at_ts": _open_utc(date(2024, 3, 6)),
         "split_factor": "20"},
        {"symbol": "AAA", "action_type": "SPLIT", "ex_date": "2024-03-08", "available_at_ts": _open_utc(date(2024, 3, 8)),
         "split_factor": "2"},
    ]
    bars = prepare_bars(apply_corporate_actions(raw, acts).bars)
    assert list(bars["close"]) == [Decimal("80"), Decimal("84"), Decimal("84"), Decimal("86"), Decimal("86"),
                                   Decimal("88")]
    assert bars["volume"].iloc[0] == Decimal("25")  # 1000 / 40
    logret = np.diff(np.log(bars["close_f"].to_numpy()))
    assert np.abs(logret).max() < 0.06  # no fake jumps left


def test_records_not_yet_published_at_asof_are_ignored() -> None:
    raw = _daily(["100", "102", "104", "52", "53", "54"])
    split = _Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5"))
    before = apply_corporate_actions(raw, [split], asof=datetime(2024, 3, 6, 22, 0, tzinfo=UTC))
    assert before.report["ignored_not_yet_published"] == 1
    assert list(before.bars["close"])[:3] == [Decimal("100"), Decimal("102"), Decimal("104")]
    after = apply_corporate_actions(raw, [split], asof=datetime(2024, 3, 7, 15, 0, tzinfo=UTC))
    assert after.report["ignored_not_yet_published"] == 0
    assert list(after.bars["close"])[:3] == [Decimal("50"), Decimal("51"), Decimal("52")]
    unknown_time = {"symbol": "AAA", "action_type": "SPLIT", "ex_date": "2024-03-07", "split_factor": "0.5"}
    assert apply_corporate_actions(raw, [unknown_time], asof=to_dt(SESSIONS[5])).report["ignored_not_yet_published"] == 1


def test_already_adjusted_data_is_never_adjusted_twice_and_bad_records_fail_closed() -> None:
    raw = _daily(["100", "102", "104", "52"])
    split = _Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5"))
    for policy in ("split", "all"):
        res = apply_corporate_actions(raw, [split], adjustment=policy)
        assert res.bars is raw and res.dividends == {}
    with pytest.raises(BarDataError, match="positive"):
        apply_corporate_actions(raw, [_Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)),
                                              split_factor=Decimal("0"))])
    with pytest.raises(BarDataError, match="unknown adjustment"):
        apply_corporate_actions(raw, [], adjustment="mystery")
    other = _Action("ZZZ", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5"))
    assert apply_corporate_actions(raw, [other]).report["splits"] == {}  # other symbols are irrelevant


def test_dividends_are_split_adjusted_and_credited_to_holders_only() -> None:
    raw = _daily(["100", "100", "100", "50", "50", "50", "50", "50"])
    acts = [
        _Action("AAA", "DIVIDEND", date(2024, 3, 5), _open_utc(date(2024, 3, 5)), cash_amount=Decimal("1.00")),
        _Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5")),
        _Action("AAA", "DIVIDEND", date(2024, 3, 12), _open_utc(date(2024, 3, 12)), cash_amount=Decimal("0.40")),
        _Action("AAA", "RENAME", date(2024, 3, 12), _open_utc(date(2024, 3, 12))),
    ]
    res = apply_corporate_actions(raw, acts)
    assert res.dividends == {"AAA": {date(2024, 3, 5): Decimal("0.5"), date(2024, 3, 12): Decimal("0.4")}}
    assert res.report["reported_not_applied"] == {"RENAME": 1}
    bars = prepare_bars(res.bars)
    # held from the first open through the end -> both dividends credited on whole adjusted shares
    held = run_backtest(bars, np.ones(len(bars), dtype=np.int8), zero_cost_model(), strategy="bh",
                        dividends=res.dividends["AAA"])
    qty = held.trades[0].quantity
    assert held.trades[0].bar_index == 1  # bought at the 03-05 open = ex-date -> NOT entitled to that dividend
    assert held.diagnostics["dividends_credited"] == 1
    assert held.metrics["dividend_income"] == format((qty * Decimal("0.4")).quantize(Decimal("0.01")), "f")
    # a sale at the ex-date open keeps the dividend (holder of record at the previous close)
    tgt = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int8)  # decision at 03-11 close -> sell at 03-12 open
    sold = run_backtest(bars, tgt, zero_cost_model(), strategy="x", dividends=res.dividends["AAA"])
    assert [t.side for t in sold.trades] == ["BUY", "SELL"] and sold.trades[1].bar_index == 6
    assert sold.diagnostics["dividends_credited"] == 1
    no_div = run_backtest(bars, tgt, zero_cost_model(), strategy="x")
    assert sold.equity[-1] - no_div.equity[-1] == qty * Decimal("0.4")
    with pytest.raises(ValueError, match="dividend"):
        run_backtest(bars, tgt, zero_cost_model(), strategy="x", dividends={date(2024, 3, 12): Decimal("-1")})


def test_run_all_baselines_records_adjustments_and_dividends(tmp_path: Path) -> None:
    raw = _daily(["100", "100", "100", "50", "50", "50", "50", "50"])
    acts = [_Action("AAA", "SPLIT", date(2024, 3, 7), _open_utc(date(2024, 3, 7)), split_factor=Decimal("0.5")),
            _Action("AAA", "DIVIDEND", date(2024, 3, 12), _open_utc(date(2024, 3, 12)), cash_amount=Decimal("0.40"))]
    res = apply_corporate_actions(raw, acts)
    out = run_all_baselines(res.bars, zero_cost_model(), tmp_path, strategies=[BuyAndHold()],
                            run_id="bl_adj", dividends=res.dividends, data_adjustments=res.report)
    assert out["data_adjustments"]["splits"]["AAA"][0]["ex_date"] == "2024-03-07"
    run = out["runs"][0]
    assert run["diagnostics"]["dividends_credited"] == 1
    # split-adjusted, flat prices: the only P&L is the dividend
    assert Decimal(run["metrics"]["total_return"]) > 0


def test_signal_engine_blocks_unadjusted_split_in_lookback_and_accepts_adjusted() -> None:
    df = make_bars(("AAA",), timeframe="1Day", n=60, seed=3, drift=0.002, vol=0.004)
    asof = to_dt(df["available_at_ts"].max())
    raw = df.copy()
    cut = len(raw) - 5  # 2-for-1 split five sessions before asof, shown raw
    for col in ("open", "high", "low", "close"):
        raw.loc[cut:, col] = [v / 2 for v in raw.loc[cut:, col]]
    raw.loc[cut:, "volume"] = [v * 2 for v in raw.loc[cut:, "volume"]]
    eng = QuantSignalEngine(CostModel(), horizon_minutes=5 * 390, min_net_edge_bps=0)
    blocked = eng.generate(raw, asof, ["AAA"], "cz")
    assert blocked and not blocked[0].is_candidate and "unadjusted corporate action" in blocked[0].reason
    ex = to_dt(raw["start_ts"].iloc[cut]).astimezone(NY).date()
    split = _Action("AAA", "SPLIT", ex, to_dt(raw["start_ts"].iloc[cut]), split_factor=Decimal("0.5"))
    fixed = apply_corporate_actions(raw, [split], asof=asof).bars
    ok = eng.generate(fixed, asof, ["AAA"], "cz")
    assert ok and "unadjusted corporate action" not in ok[0].reason
    with pytest.raises(ValueError):
        QuantSignalEngine(CostModel(), max_abs_bar_log_return=0)
