"""Multi-task labels retain existing net-return/volatility/cost column semantics."""
import numpy as np

NAMES = ('net_return_60_bps', 'realized_volatility', 'execution_shortfall_proxy_bps',
         'net_return_5_bps', 'net_return_15_bps', 'success_5', 'success_15', 'success_60',
         'downside_60_bps', 'log_dollar_volume_60')


def multitask_targets(ds, bars):
    liquidity = np.full(len(ds.decision_ns), np.nan)
    for symbol in ds.symbols:
        b = bars[bars.symbol == symbol]
        starts = b.start_ts.astype('int64').to_numpy()
        dollar_volume = b.volume_f.to_numpy()*b.close_f.to_numpy()
        cumulative = np.r_[0, np.cumsum(dollar_volume)]
        rows = np.flatnonzero(ds.panel.symbol.to_numpy() == symbol)
        entry = np.searchsorted(starts, ds.decision_ns[rows], side='left')
        end = np.searchsorted(starts, ds.exit_ns[60][rows], side='left')
        valid = (entry < len(b)) & (end <= len(b)) & (end > entry)
        liquidity[rows[valid]] = np.log1p(cumulative[end[valid]]-cumulative[entry[valid]])
    return np.column_stack([ds.net_bps[60], np.sqrt(np.maximum(ds.rv_target, 0)), ds.realized_cost_bps,
                            ds.net_bps[5], ds.net_bps[15], *[(ds.net_bps[h] > 0).astype(float) for h in (5, 15, 60)],
                            np.maximum(-ds.net_bps[60], 0), liquidity])


def probability_metrics(prediction, targets):
    if prediction.shape[1] < 8:
        return {}
    result = {}
    for col, horizon in zip((5, 6, 7), (5, 15, 60), strict=True):
        p, y = np.clip(prediction[:, col], 0, 1), targets[:, col]
        ece = 0.
        for low in np.arange(10)/10:
            inside = (p >= low) & (p < low+.1 if low < .9 else p <= 1)
            if inside.any():
                ece += inside.mean()*abs(p[inside].mean()-y[inside].mean())
        result[str(horizon)] = {'brier': float(np.mean((p-y)**2)), 'ece': float(ece)}
    return result
