"""Export immutable OOS scores for backtest-only LEAN replay; no online model or broker path."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .adapter import sha


def export_replay(artifacts: Path, model: str, destination: Path) -> dict:
    result = json.loads((artifacts / 'result.json').read_text())
    source = artifacts / 'predictions.json'
    if sha(source) != result['prediction_sha256']:
        raise ValueError('OOS predictions failed integrity check')
    rows = [r for r in json.loads(source.read_text()) if r['model'] == model]
    if not rows or len({(r['decision_ns'], r['symbol']) for r in rows}) != len(rows):
        raise ValueError('missing/duplicate OOS replay rows')
    if any(r['exit_ns'] <= r['decision_ns'] for r in rows):
        raise ValueError('invalid replay time interval')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['decision_ns', 'exit_ns', 'symbol', 'score_bps'])
        for row in sorted(rows, key=lambda r: (r['decision_ns'], -r['score'], r['symbol'])):
            writer.writerow([row['decision_ns'], row['exit_ns'], row['symbol'], row['score']])
    manifest = {'schema': 'lat-oos-replay/1', 'model': model, 'dataset_key': result['dataset_key'],
                'source_sha256': result['prediction_sha256'], 'csv_sha256': sha(destination),
                'is_simulated': result['is_simulated'], 'rows': len(rows), 'execution_authority': False,
                'note': 'LEAN limit fills and fees must be evaluated separately; research open-price fills are not LEAN parity.'}
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
