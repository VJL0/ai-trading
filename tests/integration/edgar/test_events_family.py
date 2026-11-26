"""Synthetic five-session ablation through the normal QuantLab gates and trial ledger."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from services.market_data.edgar.parsers import Fact, Filing, Ownership, Transaction
from services.market_data.edgar.store import Disclosure, EdgarStore, store_path
from services.quantlab import cli
from services.quantlab.events import attach_events
from services.quantlab.novelty import novelty_report
from services.quantlab.panel import EquityPanel
from services.quantlab.pipeline import Lab
from services.quantlab.strategies import events_family as L
from tests.unit.quantlab import synth

_STOCKS = 36


def _dev() -> EquityPanel:
    etfs = ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV")
    syms = {s: {"drift": 0.0003, "vol": 0.01, "price": 100, "volume": 4e7} for s in etfs}
    for i in range(_STOCKS):
        # Plant a strong persistent return rank related to gross profitability so that some edges clear costs.
        syms[f"S{i:02d}"] = {"drift": 0.0002 + 0.00011 * (i - 17), "vol": 0.014, "price": 50, "volume": 3e6}
    raw, adjusted = synth.bars(syms, n=1550, start="2016-01-04")
    return EquityPanel.build(raw, adjusted, synth.assets(list(syms), etfs=etfs), min_obs=50)


def _disclosures() -> Iterator[Disclosure]:
    for year in range(2016, 2022):
        for i in range(_STOCKS):
            symbol = f"S{i:02d}"
            end, prior = date(year - 1, 12, 31), date(year - 2, 12, 31)
            filed = date(year, 2, 12 + i % 10)
            period_start = date(year - 1, 1, 1)
            facts = (
                Fact("assets", None, prior, 80 + i * 5 + (year - 2016) * 5),
                Fact("assets", None, end, 110 + i * 4 + (year - 2015) * (i % 4 + 2)),
                Fact("shares", None, prior, 100 + i),
                Fact("shares", None, end, 100 + 2 * i + (year - 2016) * 5),
                Fact("net_income", period_start, end, 8 + i * 2),
                Fact("operating_cash_flow", period_start, end, 3 + i * 2.5),
                Fact("gross_profit", period_start, end, 25 + i * 2),
            )
            eps = tuple(Fact("eps", date(year - 1, start, 1), date(year - 1, end_month, last_day),
                             i / 30 + (year - 2015) ** 2 / (q + 1))
                        for q, (start, end_month, last_day) in enumerate(((1, 3, 31), (4, 6, 30), (7, 9, 30), (10, 12, 31))))
            timely = datetime(year, filed.month, filed.day, 13, tzinfo=UTC)
            yield Disclosure(Filing(i + 1, f"annual-{year}", "10-K", timely, filed, end, earnings=True),
                             facts=facts + eps, tickers=(symbol,))
            tx_day = date(year, 3, 1 + i % 5)
            release = datetime(year, 3, tx_day.day + 2, 13, tzinfo=UTC)
            owners = ("1", "2", "3")
            tx = tuple(Transaction(o, ("officer",), tx_day, "P" if i % 3 else "S", 100, 5.0, int(o)) for o in owners)
            ownership = Ownership((symbol,), owners, None, tx)
            yield Disclosure(Filing(i + 1, f"insiders-{year}", "4", release, release.date(), tx_day),
                             ownership=ownership, tickers=(symbol,))


def test_l_events_ledger_and_after_cost_ablation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    store = EdgarStore(store_path(tmp_path))
    store.ingest(_disclosures())
    dev = _dev()
    dev.save(tmp_path / "datasets" / "quantlab" / "panels" / "equity_synthetic.npz")
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path))
    assert cli.main(["research", "--asset", "events", "--families", L.FAMILY]) == 0
    result = json.loads(capsys.readouterr().out)
    lab = Lab(tmp_path)
    fam = next(f for f in result["families"] if f["family"] == L.FAMILY)
    assert fam["n_variants"] == fam["n_trials"] == 4
    report_path = lab.out / "runs" / result["run_id"] / "events_comparison.json"
    comparison = json.loads(report_path.read_text())
    assert result["events_comparison"] == comparison
    assert comparison["classification"] == "development_backtest"
    assert set(comparison["arms"]) == set(L.ARMS)
    baseline = comparison["arms"]["price_only"]
    assert baseline["start"] == "2020-01-02"
    assert any(variant["cost_drag_annual"] > 0 for variant in comparison["arms"].values())
    assert comparison["missing_development_fields"] == []
    for variant in comparison["arms"].values():
        assert variant["days"] == baseline["days"] > 250
        assert variant["delta_cagr_vs_price_only"] == pytest.approx(variant["cagr"] - baseline["cagr"])
    assert lab.ledger.count(L.FAMILY) == 4
    repeated = L.run_events_research(lab, dev, store, run_id="synthetic_edgar_repeat")
    assert repeated["skipped_no_new_evidence"] == [L.FAMILY]
    assert lab.ledger.count(L.FAMILY) == 4
    future_panel = replace(dev, dates=pd.date_range("2025-11-21", periods=len(dev.dates)))
    with pytest.raises(ValueError, match="development-only"):
        L.run_events_research(lab, future_panel, store)


def test_edgar_factor_passes_through_novelty_screen(tmp_path: Path) -> None:
    store = EdgarStore(store_path(tmp_path))
    store.ingest(_disclosures())
    dev = attach_events(_dev(), store)
    mask = dev.universe(min_adv=1e6, min_history=60, kinds=("COMMON_STOCK",))
    report = novelty_report("rank(gross_profitability())", dev, mask, {"same": "gross_profitability()"})
    assert report["nearest"] == "agent:same"
    assert report["redundant"] is True
