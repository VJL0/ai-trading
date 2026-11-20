"""Explicit Kronos setup and bounded local benchmark commands."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from ..autonomous.engine import Journal, prepare_cache
from ..autonomous.learners import Hypothesis
from .adapter import provision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup', help='download public pinned-revision safetensors into local storage')
    setup.add_argument('--small', action='store_true')
    bench = sub.add_parser('benchmark', help='enqueue and evaluate a paired development benchmark')
    bench.add_argument('--dataset', required=True)
    bench.add_argument('--root', type=Path, default=Path('experiments/autonomous'))
    bench.add_argument('--datasets-dir', type=Path, default=Path('datasets'))
    bench.add_argument('--variant', choices=('base', 'small'), default='base')
    bench.add_argument('--context', type=int, default=128)
    bench.add_argument('--samples', type=int, default=8)
    bench.add_argument('--rows-per-fold', type=int, default=64)
    bench.add_argument('--tune-steps', type=int, default=0)
    bench.add_argument('--timeout', type=float, default=1800)
    bench.add_argument('--memory-gb', type=float, default=6)
    replay = sub.add_parser('lean-export', help='export one model OOS score file for LEAN replay')
    replay.add_argument('--artifacts', type=Path, required=True)
    replay.add_argument('--model', default='kronos_base')
    replay.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'lean-export':
        from .lean import export_replay
        print(json.dumps(export_replay(args.artifacts, args.model, args.output), indent=2))
        return 0
    if args.command == 'setup':
        print(json.dumps(provision(variants=('base', 'small') if args.small else ('base',)), indent=2))
        return 0
    from packages.core.config import load_settings

    from ..autonomous.__main__ import run_batch
    from ..autonomous.validation import ValidationBook
    from ..cli import _quote_spreads, load_data
    from .adapter import ForecastConfig
    ForecastConfig(variant=args.variant, context=args.context, samples=args.samples)
    if not 1 <= args.rows_per_fold <= 10000 or not 0 <= args.tune_steps <= 128:
        parser.error('invalid row/step budget')
    data = load_data(source='store', dataset=args.dataset, timeframe='1Min', symbols=None,
                     prefer_timeframe='1Min', datasets_dir=args.datasets_dir)
    quotes = _quote_spreads(data, argparse.Namespace(no_quotes=False, datasets_dir=str(args.datasets_dir)))
    journal = Journal(args.root)
    key = prepare_cache(journal, data, quotes, book=ValidationBook(load_settings().experiments_dir))
    spec = Hypothesis(f'kronos_{args.variant}', kronos_context=args.context, kronos_samples=args.samples,
                      kronos_rows_per_fold=args.rows_per_fold, kronos_tune_steps=args.tune_steps)
    job = journal.enqueue(key, spec)
    print(json.dumps({'job': job, 'dataset_key': key}), flush=True)
    if not 0 < args.timeout < float('inf') or not 0 < args.memory_gb < float('inf'):
        parser.error('resource budgets must be positive and finite')
    with (args.root / 'scheduler.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('autonomous scheduler is already active; the benchmark remains queued')
        run_batch(journal, key, 1, args.timeout, args.memory_gb, lock.fileno(), job_ids={job})
    row = next(r for r in journal.rows() if r['id'] == job)
    print(json.dumps(row, indent=2))
    return 0 if row['state'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
