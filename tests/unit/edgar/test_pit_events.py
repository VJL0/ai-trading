from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.agents import mechanisms
from services.agents.context import dsl_grammar
from services.market_data.edgar.parsers import Fact, Filing, Ownership, Transaction, acceptance, company_facts
from services.market_data.edgar.store import Disclosure, EdgarStore, store_path
from services.quantlab.events import EVENT_FIELDS, attach_events
from services.quantlab.factors import EventDataUnavailable, SpecError, evaluate_expression, prepare_agent_events, validate_spec
from services.quantlab.panel import EquityPanel


def _panel(dates: list[str], symbols: tuple[str, ...] = ("OLD", "NEW")) -> EquityPanel:
    return EquityPanel(pd.DatetimeIndex(dates), np.array(symbols), {"close": np.ones((len(dates), len(symbols)))},
                       np.array(["COMMON_STOCK"] * len(symbols)))


def test_restatement_asof_and_delayed_ticker_switch(tmp_path: Path) -> None:
    store = EdgarStore(store_path(tmp_path))
    period = date(2023, 12, 31)
    prior = date(2022, 12, 31)
    original = Filing(7, "original", "10-K", acceptance("2024-02-01T08:00:00"), date(2024, 2, 1), period, earnings=True)
    amended = Filing(7, "restated", "10-K/A", acceptance("2024-03-08T16:30:00"), date(2024, 3, 8), period, earnings=True)
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"accn": "original", "end": str(prior), "val": 100},
        {"accn": "original", "end": str(period), "val": 110},
        {"accn": "restated", "end": str(period), "val": 160},
        {"accn": "not-in-submissions", "end": str(period), "val": 999},
    ]}}}}}
    facts = company_facts(payload, {f.accession: f for f in (original, amended)})
    store.ingest([Disclosure(amended, facts["restated"], tickers=("NEW",)),
                  Disclosure(original, facts["original"], tickers=("OLD",))])
    store.snapshot_tickers({7: ("NEW",)}, datetime(2024, 5, 1, tzinfo=UTC))
    records = list(store.as_of(datetime(2024, 3, 8, 23, tzinfo=UTC)))
    assert [r.filing.accession for r in records] == ["original"]
    assert [r.filing.accession for r in store.as_of(datetime(2024, 3, 11, 13, 30, tzinfo=UTC), cik=7)] == [
        "original", "restated",
    ]
    assert store.cik_for_ticker("OLD", datetime(2024, 3, 8, 23, tzinfo=UTC)) == 7
    assert store.cik_for_ticker("NEW", datetime(2024, 3, 8, 23, tzinfo=UTC)) is None
    assert store.cik_for_ticker("NEW", datetime(2024, 3, 11, 13, 30, tzinfo=UTC)) == 7
    assert store.cik_for_ticker("OLD", datetime(2024, 3, 11, 13, 30, tzinfo=UTC)) is None
    joined = attach_events(_panel(["2024-02-01", "2024-03-08", "2024-03-11"]), store)
    np.testing.assert_allclose(joined.f["asset_growth"], [[0.1, np.nan], [0.1, np.nan], [np.nan, 0.6]])
    assert joined.f["earnings_announcement"][0, 0] == 1
    assert joined.f["earnings_announcement"][-1, 1] == 0  # an amendment does not restart drift
    assert joined.f["earnings_drift_age"][-1, 1] > joined.f["earnings_drift_age"][1, 0]
    assert mechanisms.is_expressible("fundamentals", tmp_path)
    assert "earnings" in mechanisms.available_mechanisms(tmp_path)
    assert not mechanisms.is_expressible("insider", tmp_path)
    dev, fingerprint = prepare_agent_events(joined, tmp_path, "asset_growth()")
    revised, other = prepare_agent_events(joined, tmp_path, "earnings_announcement()")
    assert fingerprint != other
    assert revised.shape == dev.shape
    assert validate_spec({"expression": " (asset_growth()) ".strip()})["expression"] == "asset_growth()"
    np.testing.assert_allclose(evaluate_expression("asset_growth()", dev, np.ones(dev.shape, dtype=bool)),
                               dev.f["asset_growth"])


def test_insiders_cluster_on_distinct_owners_and_replace_original(tmp_path: Path) -> None:
    store = EdgarStore(store_path(tmp_path))
    tx_day = date(2024, 3, 4)

    def record(accession: str, form: str, filed: date, owners: tuple[str, ...], code: str,
               original: date | None = None) -> Disclosure:
        filing = Filing(7, accession, form, datetime(filed.year, filed.month, filed.day, 12, tzinfo=UTC), filed, tx_day)
        transactions = tuple(Transaction(owner, ("director",), tx_day, code, 100, 2.0, 0) for owner in owners)
        return Disclosure(filing, ownership=Ownership(("OLD",), owners, original, transactions), tickers=("OLD",))

    dated = [date(2024, 3, day) for day in (5, 6, 7)]
    store.ingest([record("joint", "4", dated[0], ("01", "02"), "P"), record("solo", "4", dated[1], ("03",), "P"),
                  record("amend", "4/A", dated[2], ("03",), "S", dated[1])])
    panel = attach_events(_panel(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-04-03"]), store)
    np.testing.assert_allclose(panel.f["insider_cluster_buy"][:, 0], [np.nan, 0, 1, 0, 0])
    np.testing.assert_allclose(panel.f["insider_net_buy_ratio"][:, 0], [np.nan, 1, 1, 0, 0])
    assert mechanisms.is_expressible("insider", tmp_path)


def test_empty_store_prompts_for_data_and_dsl_stays_closed(tmp_path: Path) -> None:
    dates = _panel(["2024-03-05"])
    assert mechanisms.available_mechanisms(tmp_path) == mechanisms.EXPRESSIBLE
    mech = mechanisms.get("insider", tmp_path)
    assert mech is not None and not mech.expressible_now and mech.data_needed
    assert set(EVENT_FIELDS) <= {f.name for f in dsl_grammar().functions}
    with pytest.raises(EventDataUnavailable, match="sue"):
        prepare_agent_events(dates, tmp_path, "sue()")
    for expression in ("sue(5)", "sue", "sue(period=4)", "events.store()"):
        with pytest.raises(SpecError):
            validate_spec({"expression": expression})


def test_seasonal_surprise_and_fiscal_ratios(tmp_path: Path) -> None:
    store = EdgarStore(store_path(tmp_path))
    periods = pd.period_range("2021Q4", periods=9, freq="Q")
    values = (1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0)
    for period, value in zip(periods, values, strict=True):
        start, end = period.start_time.date(), period.end_time.date()
        # Only the final filing's publication instant enables the ninth quarterly EPS.
        filed = date(end.year + (end.month == 12), end.month % 12 + 1, 26)
        filing = Filing(7, str(period), "10-Q", datetime.combine(filed, datetime.min.time(), UTC), filed, end)
        facts = [Fact("eps", start, end, value)]
        if period == periods[-1]:
            prior = date(2022, 12, 31)
            facts += [
                Fact("assets", None, prior, 100), Fact("assets", None, end, 120),
                Fact("shares", None, prior, 100), Fact("shares", None, end, 90),
                Fact("net_income", date(2023, 1, 1), end, 20),
                Fact("operating_cash_flow", date(2023, 1, 1), end, 31),
                Fact("revenue", date(2023, 1, 1), end, 100),
                Fact("cost_of_revenue", date(2023, 1, 1), end, 70),
            ]
        store.ingest([Disclosure(filing, tuple(facts), tickers=("OLD",))])
    panel = attach_events(_panel(["2024-01-25", "2024-01-26"]), store)
    assert np.isnan(panel.f["sue"][0, 0])
    assert float(panel.f["sue"][1, 0]) == pytest.approx(6.0 / np.std([1, 2, 3, 4], ddof=1))
    assert float(panel.f["accruals"][1, 0]) == pytest.approx(-0.1)
    assert float(panel.f["asset_growth"][1, 0]) == pytest.approx(0.2)
    assert float(panel.f["gross_profitability"][1, 0]) == pytest.approx(0.25)
    assert float(panel.f["net_share_issuance"][1, 0]) == pytest.approx(-0.1)
