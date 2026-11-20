"""Local scheduler: python -m services.research.autonomous [--once]."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from .datasets import pick_history
from .engine import Journal, evaluate, prepare_cache, propose
from .evidence import screen


def worker(root: Path, jid: str) -> int:
    os.nice(10)
    journal = Journal(root)
    if not journal.claim(jid):
        return 0
    try:
        journal.finish(jid, evaluate(root, jid))
        return 0
    except Exception as exc:
        journal.finish(jid, error=f'{type(exc).__name__}: {exc}')
        return 1


def run_batch(journal: Journal, dataset: str, workers: int, timeout: float, memory_gb: float, lock_fd: int,
              job_ids: set[str] | None = None) -> None:
    pending = [r for r in journal.rows(dataset) if r['state'] == 'QUEUED'
               and (job_ids is None or r['id'] in job_ids)]
    active: dict[str, tuple[subprocess.Popen, float, TextIO]] = {}
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'SYSTEMROOT')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               VECLIB_MAXIMUM_THREADS='2', PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.7', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.5')
    try:
        while pending or active:
            while pending and len(active) < workers:
                # At most one GPU learner; CPU families can run alongside it.
                gpu_families = ('torch_', 'kronos_', 'tabm', 'temporal_', 'foundation_', 'tabiclv2', 'kumo', 'mitra')
                gpu_busy = any(r['id'] in active and r['spec']['family'].startswith(gpu_families)
                               for r in journal.rows(dataset))
                selectable = next((i for i, r in enumerate(pending)
                                   if not gpu_busy or not r['spec']['family'].startswith(gpu_families)), None)
                if selectable is None:
                    break
                row = pending.pop(selectable)
                log_dir = journal.root / 'logs'
                log_dir.mkdir(exist_ok=True)
                log = (log_dir / f"{row['id']}.log").open('w')
                proc = subprocess.Popen([sys.executable, '-m', 'services.research.autonomous', '--root',
                                         str(journal.root.resolve()), '--worker', row['id']],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,),
                                        start_new_session=True)
                active[row['id']] = (proc, time.monotonic(), log)
            for jid, (proc, started, log) in list(active.items()):
                reason = None
                if proc.poll() is None:
                    if time.monotonic() - started > timeout:
                        reason = 'worker wall-clock budget exceeded'
                    else:
                        rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(proc.pid)], capture_output=True, text=True)
                        if rss.stdout.strip() and int(rss.stdout.strip()) > memory_gb * 1024**2:
                            reason = 'worker resident-memory budget exceeded'
                    if reason:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                if proc.poll() is not None:
                    log.close()
                    row = next(r for r in journal.rows(dataset) if r['id'] == jid)
                    if reason or row['state'] in ('RUNNING', 'QUEUED'):
                        journal.finish(jid, error=reason or f'worker exited without result ({proc.returncode})')
                    del active[jid]
            time.sleep(.2)
    finally:
        for jid, (proc, _, log) in active.items():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
            journal.finish(jid, error='scheduler stopped')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('experiments/autonomous'))
    ap.add_argument('--datasets-dir', type=Path, default=Path('datasets'))
    ap.add_argument('--dataset')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--worker', help=argparse.SUPPRESS)
    ap.add_argument('--confirm', help=argparse.SUPPRESS)
    ap.add_argument('--workers', type=int, default=min(4, max(1, (os.cpu_count() or 4) // 3)))
    ap.add_argument('--rd-agent', action='store_true', help='use configured real AI transport for one bounded RD-Agent proposal')
    ap.add_argument('--budget', type=int, default=24)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--memory-gb', type=float, default=6, help='per worker RSS limit; 24 GiB default aggregate CPU ceiling')
    ap.add_argument('--interval', type=float, default=300)
    args = ap.parse_args()
    if args.worker:
        return worker(args.root, args.worker)
    if not (1 <= args.workers <= max(1, (os.cpu_count() or 4) - 2)) or args.budget < 1:
        ap.error('invalid worker count or campaign budget')
    if not all(0 < v < float('inf') for v in (args.timeout, args.memory_gb, args.interval)):
        ap.error('resource limits must be finite and positive')
    def terminate(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    from packages.core.config import load_settings

    from .validation import ValidationBook, confirm
    book = ValidationBook(load_settings().experiments_dir)
    journal = Journal(args.root)
    if args.status:
        print(json.dumps(journal.snapshot()))
        return 0
    with (args.root / ('confirmation.lock' if args.confirm else 'scheduler.lock')).open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error('a scheduler or its workers are already active')
        for row in ([] if args.confirm else journal.rows()):
            if row['state'] == 'RUNNING':
                journal.finish(row['id'], error='previous scheduler interrupted; checkpoint retained')
        while True:
            try:
                from ...market_data.store import MarketDataStore
                from ..cli import _quote_spreads, load_data
                dataset = args.dataset
                if dataset is None:
                    manifest = pick_history(MarketDataStore(args.datasets_dir))
                    if manifest is None:
                        raise ValueError('no suitable stored intraday dataset')
                    dataset = manifest['dataset_version']
                data = load_data(source='store', dataset=dataset, timeframe='1Min', symbols=None,
                                 prefer_timeframe='1Min', datasets_dir=args.datasets_dir)
                quote_args = argparse.Namespace(no_quotes=False, datasets_dir=str(args.datasets_dir))
                quotes = _quote_spreads(data, quote_args)
                if args.confirm:
                    print(json.dumps(confirm(args.root, args.confirm, data, quotes, book)))
                    return 0
                key = prepare_cache(journal, data, quotes, book=book)
                if args.rd_agent and len(journal.rows(key)) < args.budget:
                    import asyncio

                    from ..advanced.rd_agent import propose as rd_propose
                    attempt = args.root/'rd-agent'/f'{key}-{len(journal.rows(key))}'
                    asyncio.run(rd_propose(journal, key, attempt))
                propose(journal, key, args.budget)
                run_batch(journal, key, args.workers, args.timeout, args.memory_gb, lock.fileno())
                rows = journal.rows()
                from services.memory.comparison import compare
                comparison = compare(rows, key)
                (args.root / 'memory-comparison.json').write_text(json.dumps(comparison, indent=2))
                eligible = [r for r in rows if r['dataset'] == key and r['result']
                            and screen(r['result'], len(rows))['eligible_for_independent_validation']]
                attempted = {r['job'] for r in book.status()['confirmations']}
                eligible = [r for r in eligible if r['id'] not in attempted]
                if eligible:
                    champion = max(eligible, key=lambda r: r['result']['objective'])
                    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
                    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.7', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.5')
                    with (args.root / 'logs' / 'confirmation.log').open('a') as log:
                        subprocess.run([sys.executable, '-m', 'services.research.autonomous', '--root', str(args.root),
                                        '--datasets-dir', str(args.datasets_dir), '--dataset', dataset,
                                        '--confirm', champion['id'], '--once'], env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                journal.event('COMPARE', 'Development comparison complete; only independent confirmations can promote')
            except Exception as exc:
                journal.event('BLOCKED', f'{type(exc).__name__}: {exc}')
                if args.once:
                    print(json.dumps(journal.snapshot()))
                    return 2
            if args.once:
                print(json.dumps(journal.snapshot()))
                return 0
            time.sleep(args.interval)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
