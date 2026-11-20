"""Paired, all-attempt corrected development comparison; never grants deployment authority."""
from __future__ import annotations

import numpy as np

from services.research.autonomous.evidence import block_p_value

from .ablation import MODES


def compare(rows: list[dict], dataset: str) -> dict:
    matched = [r for r in rows if r['dataset'] == dataset and r['spec']['family'] == 'ridge']
    base = next((r for r in matched if not r['spec'].get('memory_mode') and r.get('result')), None)
    output = {'dataset': dataset, 'total_attempts': len(rows), 'validation': 'DEVELOPMENT_ONLY',
              'execution_authority': False, 'comparisons': []}
    alpha = .05 / (2 * max(1, len(rows)))
    for mode in MODES:
        challenger = next((r for r in matched if r['spec'].get('memory_mode') == mode and r.get('result')), None)
        item = {'mode': mode, 'decision': 'INSUFFICIENT_EVIDENCE', 'alpha': alpha}
        if base and challenger:
            a, b = base['result'], challenger['result']
            x, y = a.get('daily_net_bps', {}), b.get('daily_net_bps', {})
            spec_a = {k: v for k, v in base['spec'].items() if k != 'memory_mode'}
            spec_b = {k: v for k, v in challenger['spec'].items() if k != 'memory_mode'}
            if x and x.keys() == y.keys() and spec_a == spec_b and a['code_hash'] == b['code_hash']:
                delta = np.array([y[d] - x[d] for d in sorted(x)])
                p = block_p_value(delta)
                a_sharpe, b_sharpe = a['metrics'].get('net_sharpe'), b['metrics'].get('net_sharpe')
                sharpe_delta = None if a_sharpe is None or b_sharpe is None else b_sharpe - a_sharpe
                drawdown_delta = b['risk']['max_drawdown'] - a['risk']['max_drawdown']
                passed = (len(x) >= 120 and b.get('is_simulated') is False and p <= alpha
                          and b['p_positive'] <= alpha and sharpe_delta is not None and sharpe_delta > 0
                          and drawdown_delta <= 0
                          and b['metrics']['n_trades'] >= 100
                          and b['stress_2x_costs']['net_return'] > 0)
                item.update(sessions=len(x), paired_mean_bps=float(delta.mean()), p_incremental=p,
                            sharpe_delta=sharpe_delta, drawdown_delta=drawdown_delta,
                            decision='REQUIRES_INDEPENDENT_CONFIRMATION' if passed else 'REJECTED',
                            baseline_job=base['id'], challenger_job=challenger['id'])
        output['comparisons'].append(item)
    return output
