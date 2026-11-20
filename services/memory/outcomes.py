"""Observed forecast diagnostics from recorded snapshots, never fabricated execution P&L.

Only near-horizon observations from the SAME dataset/feed/simulation namespace are paired.
Actual fills and realized account P&L remain in the authoritative ledger. Missing prices
leave outcomes pending. Estimated transaction costs are explicitly not actual commissions.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from .store import Memory, stamp


def _price(snapshot: dict) -> Decimal | None:
    q = snapshot.get('quote')
    if q and q.get('bid') and q.get('ask'):
        return (Decimal(q['bid']) + Decimal(q['ask'])) / 2
    return None


def observe_outcomes(operational, memory: Memory, limit: int = 200) -> int:
    with memory.connect() as db:
        done = {r[0] for r in db.execute("SELECT source_id FROM memories WHERE source='forecast_outcome'")}
    # Recent bounded scan; older unresolved signals remain visible in the durable journal.
    signals = operational.query('SELECT * FROM quant_signals ORDER BY generated_at DESC LIMIT ?', (limit,))
    recorded = 0
    for row in signals:
        if row['signal_id'] in done:
            continue
        signal = json.loads(row['payload_json'])
        generated = datetime.fromisoformat(stamp(signal['generated_at']))
        target = generated + timedelta(minutes=signal['horizon_minutes'])
        snapshots = operational.query(
            'SELECT payload_json FROM market_snapshots WHERE symbol=? AND julianday(data_asof) '
            'BETWEEN julianday(?) AND julianday(?) ORDER BY julianday(data_asof)',
            (row['symbol'], generated.isoformat(), (target + timedelta(minutes=2)).isoformat()))
        snapshots = [json.loads(r['payload_json']) for r in snapshots]
        snapshots = [s for s in snapshots if s.get('dataset_version') == signal['dataset_version']
                     and s.get('is_simulated') == signal['is_simulated']]
        start = next((s for s in snapshots
                      if datetime.fromisoformat(stamp(s['data_asof'])) <= generated + timedelta(seconds=30)), None)
        end = next((s for s in snapshots if datetime.fromisoformat(stamp(s['data_asof'])) >= target), None)
        if not start or not end or start.get('feed') != end.get('feed'):
            continue
        p0, p1 = _price(start), _price(end)
        if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
            continue
        observed = (p1 / p0 - 1) * 10000
        expected = Decimal(signal['expected_return_bps'])
        cost = sum(Decimal(signal[k]) for k in ('est_spread_bps', 'est_fees_bps', 'est_slippage_bps'))
        memory.append(source='forecast_outcome', source_id=row['signal_id'],
                      available_at=datetime.now(UTC).isoformat(), event_at=end['data_asof'],
                      kind='episodic', symbol=row['symbol'],
                      body={'signal_id': row['signal_id'], 'model_version': signal['model_version'],
                            'dataset_version': signal['dataset_version'], 'is_simulated': signal['is_simulated'],
                            'start_snapshot': start['snapshot_id'], 'end_snapshot': end['snapshot_id'],
                            'expected_gross_bps': str(expected), 'observed_mid_return_bps': str(observed),
                            'prediction_error_bps': str(observed - expected), 'estimated_cost_bps': str(cost),
                            'counterfactual_net_bps': str(observed - cost),
                            'measurement': 'MID_MARK_COUNTERFACTUAL_NOT_EXECUTED_PNL',
                            'was_candidate': signal['is_candidate']})
        recorded += 1
    return recorded
