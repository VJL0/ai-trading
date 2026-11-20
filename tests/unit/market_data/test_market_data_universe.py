"""Point-in-time eligibility / research-universe snapshots (allowlist, denylist, $5 floor, dollar volume)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import DataFeed
from packages.contracts.models import Bar
from packages.contracts.policy import DEFAULT_ALLOWLIST, RiskPolicy
from packages.core.db import Database
from services.market_data.calendar import get_calendar
from services.market_data.manifest import DatasetSpec
from services.market_data.store import MarketDataStore
from services.market_data.universe import (
    EligibilityReason,
    build_universe_snapshot,
    eligible_at,
    latest_universe_snapshot,
    persist_universe_snapshot,
)
from tests.unit.market_data.md_helpers import make_bar

CAL = get_calendar()
DAYS = CAL.sessions_ending(date(2026, 11, 4), 10)
AFTER = CAL.session_close(DAYS[-1]) + timedelta(hours=1)


def daily(symbol: str, d: date, price: str, volume: str) -> Bar:
    o, c = CAL.session_open(d), CAL.session_close(d)
    px = Decimal(price)
    return make_bar(symbol, o, timeframe="1Day", end_ts=c, o=price, h=str(px + Decimal("0.10")),
                    lo=str(px - Decimal("0.10")), c=price, v=volume, latency_s=900)


@pytest.fixture
def store(tmp_path: Path) -> MarketDataStore:
    st = MarketDataStore(tmp_path / "datasets")
    bars: list[Bar] = []
    for d in DAYS:
        bars += [daily("SPY", d, "680.00", "60000000"),     # liquid ETF
                 daily("PNNY", d, "2.50", "90000000"),      # penny stock
                 daily("THIN", d, "40.00", "1000"),         # $40k/day dollar volume
                 daily("TQQQ", d, "90.00", "80000000"),     # leveraged ETF (denylisted)
                 daily("NOPE", d, "50.00", "9000000")]      # not on the allowlist
    bars += [daily("NEWB", d, "30.00", "9000000") for d in DAYS[-2:]]  # only two sessions of history
    st.write_bars(bars, manifest=DatasetSpec(source_id="test:daily", feed=DataFeed.IEX, generator="tests",
                                              generator_version="1", is_synthetic=False))
    return st


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    d = Database(tmp_path / "ledger.sqlite")
    yield d
    d.close()


POLICY = RiskPolicy(symbol_allowlist=(*DEFAULT_ALLOWLIST, "PNNY", "THIN", "TQQQ", "NEWB", "GHST"))


def reasons(snap: object, sym: str) -> tuple[str, ...]:
    return next(r.reasons for r in snap.rows if r.symbol == sym)  # type: ignore[attr-defined]


def test_eligibility_rules(store: MarketDataStore) -> None:
    snap = build_universe_snapshot(store, POLICY, AFTER, candidates=["SPY", "PNNY", "THIN", "TQQQ", "NOPE", "NEWB",
                                                                     "GHST", "bad$"])
    assert snap.eligible_symbols == ["SPY"]
    assert reasons(snap, "SPY") == ()
    assert reasons(snap, "PNNY") == (EligibilityReason.BELOW_MIN_PRICE.value,)
    assert reasons(snap, "THIN") == (EligibilityReason.BELOW_MIN_DOLLAR_VOLUME.value,)
    # denylisted even though an (edited) allowlist contains it: the denylist always wins
    assert EligibilityReason.DENYLISTED_LEVERAGED_INVERSE.value in reasons(snap, "TQQQ")
    assert reasons(snap, "NOPE") == (EligibilityReason.NOT_ALLOWLISTED.value,)
    assert reasons(snap, "NEWB") == (EligibilityReason.INSUFFICIENT_HISTORY.value,)
    assert reasons(snap, "GHST") == (EligibilityReason.NO_DATA.value,)  # no data => ineligible (fail closed)
    assert reasons(snap, "BAD$") == (EligibilityReason.INVALID_SYMBOL.value,)
    spy = next(r for r in snap.rows if r.symbol == "SPY")
    assert spy.last_close == Decimal("680") and spy.sessions_observed == 10 and spy.source == "1Day"
    assert spy.avg_dollar_volume == Decimal("40800000000.00") and isinstance(spy.avg_dollar_volume, Decimal)
    assert snap.asof_date == CAL.ny_date(AFTER) and snap.policy_version == POLICY.version


def test_point_in_time_only(store: MarketDataStore) -> None:
    # 10 minutes after the 5th session close the 5th daily bar is not yet available (900 s latency)
    asof = CAL.session_close(DAYS[4]) + timedelta(minutes=10)
    snap = build_universe_snapshot(store, POLICY, asof, candidates=["SPY"], min_sessions=5)
    spy = snap.rows[0]
    assert spy.sessions_observed == 4 and not spy.eligible
    # the just-closed session's daily bar is not published yet: too little history AND not current
    assert spy.reasons == (EligibilityReason.INSUFFICIENT_HISTORY.value, EligibilityReason.STALE_DATA.value)
    later = build_universe_snapshot(store, POLICY, asof + timedelta(minutes=10), candidates=["SPY"], min_sessions=5)
    assert later.rows[0].eligible and later.rows[0].sessions_observed == 5
    before_any = build_universe_snapshot(store, POLICY, CAL.session_open(DAYS[0]), candidates=["SPY"])
    assert before_any.rows[0].reasons == (EligibilityReason.NO_DATA.value,)


def test_default_candidates_are_policy_allowlist(store: MarketDataStore) -> None:
    snap = build_universe_snapshot(store, RiskPolicy(), AFTER)
    assert {r.symbol for r in snap.rows} == set(DEFAULT_ALLOWLIST)
    assert snap.eligible_symbols == ["SPY"]  # every other allowlisted symbol has no data here
    assert snap.params["candidates"] == "policy_allowlist"


def test_persist_and_pit_lookup(store: MarketDataStore, db: Database) -> None:
    early = build_universe_snapshot(store, POLICY, CAL.session_close(DAYS[6]) + timedelta(hours=1), candidates=["SPY"])
    late = build_universe_snapshot(store, POLICY, AFTER, candidates=["SPY", "PNNY"])
    for snap in (early, late, late):  # persisting twice is idempotent
        persist_universe_snapshot(snap, store=store, db=db)
    assert len(db.query("SELECT snapshot_id FROM md_universe_snapshots")) == 2
    rows = store.load_universe_rows(late.snapshot_id)
    assert [(r["symbol"], r["eligible"]) for r in rows] == [("PNNY", False), ("SPY", True)]
    assert rows[0]["asof_date"] == CAL.ny_date(AFTER) and rows[1]["last_close"] == Decimal("680.000000")
    mid = CAL.session_close(DAYS[8])
    assert latest_universe_snapshot(db, asof=mid)["snapshot_id"] == early.snapshot_id  # type: ignore[index]
    assert latest_universe_snapshot(db)["snapshot_id"] == late.snapshot_id  # type: ignore[index]
    assert eligible_at(db, CAL.session_open(DAYS[0])) == []  # no snapshot yet at that time => nothing (fail closed)
    rebuilt = build_universe_snapshot(store, POLICY, AFTER, candidates=["SPY", "PNNY"])
    assert rebuilt.snapshot_id == late.snapshot_id  # content-derived id


def test_stale_data_is_ineligible(store: MarketDataStore) -> None:
    # five sessions after the data ends, SPY still has 5+ sessions inside the 20-session window but nothing recent
    asof = CAL.session_close(CAL.sessions_ending(date(2026, 11, 11), 1)[0]) + timedelta(hours=1)
    snap = build_universe_snapshot(store, POLICY, asof, candidates=["SPY"])
    assert snap.rows[0].reasons == (EligibilityReason.STALE_DATA.value,) and not snap.eligible_symbols


def test_rejects_float_threshold(store: MarketDataStore) -> None:
    with pytest.raises(TypeError, match="Decimal"):
        build_universe_snapshot(store, POLICY, AFTER, min_avg_dollar_volume=1e7)  # type: ignore[arg-type]
