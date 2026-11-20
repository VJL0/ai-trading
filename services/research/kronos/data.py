"""Compact point-in-time sequence cache; the outer holdout never enters worker storage."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from .adapter import COLUMNS


def sequence_arrays(bars: pd.DataFrame, ds: Any, idx: np.ndarray) -> dict[str, np.ndarray]:
    cutoff = int(ds.decision_ns[idx].max())
    arrays = {}
    symbols = sorted(ds.symbols)
    for i, symbol in enumerate(symbols):
        b = bars[(bars.symbol == symbol) & (bars.available_at_ts.astype('int64') <= cutoff)]
        x = np.asarray(b[[c + '_f' for c in COLUMNS[:5]]], dtype=np.float32)
        arrays[f'bars_{i}'] = np.column_stack([x, x[:, 4] * x[:, :4].mean(axis=1)])
        arrays[f'start_{i}'] = b.start_ts.astype('int64').to_numpy()
        arrays[f'avail_{i}'] = b.available_at_ts.astype('int64').to_numpy()
    arrays['symbol_id'] = np.array([symbols.index(s) for s in ds.panel.iloc[idx].symbol], dtype=np.int32)
    # Complete-case label availability, shared by ALL comparators. A thin-feed missing minute
    # must not turn a 60-minute forecast into a 65-minute observed-bar target.
    aligned = np.zeros(len(idx), dtype=bool)
    interval = int(ds.bar_minutes) * 60_000_000_000
    for i, symbol in enumerate(symbols):
        starts = bars.loc[bars.symbol == symbol, 'start_ts'].astype('int64').to_numpy()
        rows = np.flatnonzero(arrays['symbol_id'] == i)
        decisions = ds.decision_ns[idx[rows]]
        entry = np.searchsorted(starts, decisions, side='left')
        last = entry + ds.horizons_bars[60]
        valid = last < len(starts)
        r, e, z = rows[valid], entry[valid], last[valid]
        expected = ((decisions[valid] + interval-1)//interval)*interval
        aligned[r] = (starts[e] == expected) & (starts[z] == expected + ds.horizons_bars[60]*interval)
    arrays['calendar_aligned'] = aligned
    arrays['vol_X'] = ds.vol_X[idx]
    for h in ds.net_bps:
        arrays[f'net_{h}'] = ds.net_bps[h][idx]
        arrays[f'exit_{h}'] = ds.exit_ns[h][idx]
    return arrays


def context(a: dict[str, np.ndarray], row: int, length: int) -> pd.DataFrame:
    sym = int(a['symbol_id'][row])
    decision = a['decision'][row]
    stop = int(np.searchsorted(a[f'avail_{sym}'], decision, side='right'))
    start = max(0, stop-length)
    frame = pd.DataFrame(a[f'bars_{sym}'][start:stop], columns=COLUMNS,
                         index=pd.to_datetime(a[f'start_{sym}'][start:stop], utc=True))
    frame['available_at_ts'] = pd.to_datetime(a[f'avail_{sym}'][start:stop], utc=True)
    return frame


def future_calendar(decision: int, count: int, minutes: int) -> pd.DatetimeIndex:
    """Suite labels require same-session exits. Calendar has no future observed prices/timestamps."""
    start = pd.Timestamp(decision, tz='UTC').ceil(f'{minutes}min')
    future = pd.date_range(start, periods=count, freq=f'{minutes}min')
    local = future.tz_convert('America/New_York')
    first, last = cast(pd.Timestamp, local[0]), cast(pd.Timestamp, local[-1])
    if np.any(local.date != first.date()) or last.hour * 60 + last.minute >= 960:
        raise ValueError('forecast would extend beyond same regular session')
    return future
