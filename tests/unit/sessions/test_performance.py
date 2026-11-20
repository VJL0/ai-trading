"""Exact accounting and calendar boundary regressions."""
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from services.sessions.calendar import et_datetime, is_market_open, session_bounds
from services.sessions.performance import FillRow, dec, fifo


def test_fifo_partial_close_and_round_trip() -> None:
    now = datetime(2026, 11, 5, 15, tzinfo=UTC)
    def fill(i: str, side: str, qty: str, price: str) -> FillRow:
        return FillRow(i, i, "SPY", side, Decimal(qty), Decimal(price), now)
    result = fifo([fill("1", "BUY", "2", "100.10"), fill("2", "BUY", "1", "101.20"),
                   fill("3", "SELL", "1", "102.30"), fill("4", "SELL", "2", "99.50")])
    assert result.realized_by_fill == {"3": Decimal("2.20"), "4": Decimal("-2.30")}
    assert result.round_trips[0].pnl == Decimal("-0.10")
    assert result.open_lots == {} and result.anomalies == []


def test_invalid_accounting_is_not_silently_rounded() -> None:
    with pytest.raises(ValueError, match="binary float"):
        dec(1.23)


def test_calendar_holiday_early_close_and_dst() -> None:
    assert session_bounds(date(2026, 11, 26)) is None
    bounds = session_bounds(date(2026, 11, 27))
    assert bounds is not None and bounds[1] == datetime(2026, 11, 27, 18, tzinfo=UTC)
    assert not is_market_open(datetime(2026, 11, 27, 19, tzinfo=UTC))
    assert et_datetime(date(2026, 10, 30), "09:30").hour == 13
    assert et_datetime(date(2026, 11, 2), "09:30").hour == 14


def test_account_curve_does_not_mix_fake_and_paper_capital(tmp_path):
    import json

    from packages.core.db import Database
    from services.sessions.performance import compute_performance
    db = Database(tmp_path / 'ledger.sqlite')
    try:
        for sid, source, equity, stamp in [
            ('sim', 'FAKE_BROKER', '100000', '2026-11-05T15:00:00Z'),
            ('paper1', 'ALPACA_PAPER', '5000', '2026-11-05T15:01:00Z'),
            ('paper2', 'ALPACA_PAPER', '5001', '2026-11-05T15:02:00Z'),
        ]:
            db.write(lambda c, sid=sid, source=source, equity=equity, stamp=stamp: c.execute(
                'INSERT INTO account_snapshots VALUES(?,?,?,?,?,?)',
                (sid, stamp, source, 1, equity, json.dumps({'cash': equity}))))
        result = compute_performance(db, now=datetime(2026, 11, 5, 16, tzinfo=UTC), label='PAPER',
                                     is_simulated=False, gateway='ALPACA_PAPER')
        assert result['portfolio_value'] == '5001'
        assert result['max_drawdown'] == '0.0000'
        assert len(result['equity_curve']) == 2
    finally:
        db.close()
