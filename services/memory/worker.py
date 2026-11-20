"""Bounded background projection and evidence-linked, unvalidated reflections."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from .store import Memory

log = logging.getLogger(__name__)


def import_research(memory: Memory, root: Path) -> int:
    # Import development job records; independent validation receipts are handled separately below.
    if not (root / 'research.sqlite3').exists():
        return 0
    from services.research.autonomous.engine import Journal
    rows = Journal(root).rows()
    for row in rows:
        result = row.get('result')
        body = {'job': row['id'], 'dataset': row['dataset'], 'spec': row['spec'],
                'state': row['state'], 'error': row['error'], 'result': result,
                'validation': 'DEVELOPMENT_ONLY', 'execution_authority': False}
        memory.append(source=str(root.resolve()) + ':jobs', source_id=row['id'], body=body,
                      available_at=row['updated'], kind='research', status='DEVELOPMENT_EVIDENCE')
        if row['state'] not in ('COMPLETED', 'FAILED'):
            continue
        reasons = []
        if row['error']:
            reasons.append('Repair the recorded failure and rerun the identical specification before comparison.')
        if result:
            from services.research.autonomous.evidence import screen
            reasons += screen(result, len(rows))['reasons']
        memory.append(source=str(root.resolve()) + ':reflection', source_id=row['id'],
                      available_at=row['updated'], kind='reflection', status='HYPOTHESIS',
                      body={'job': row['id'], 'observed_reasons': reasons,
                            'hypothesis': 'A matched cost-stress and memory ablation may explain this failure.',
                            'required_test': 'Purged paired sessions; all-trial correction; independent confirmation.',
                            'established_fact': False})
    # Import validation receipts only; the validator owns all holdout access and artifact checks.
    book_path = root.parent / 'autonomous-validation.sqlite3'
    if book_path.exists():
        from contextlib import closing

        from services.research.autonomous.validation import ValidationBook
        book = ValidationBook(root.parent)
        with closing(book.connect()) as db:
            confirmed = db.execute("SELECT id FROM confirmations WHERE state='CONFIRMED'").fetchall()
        for row in confirmed:
            try:
                memory.remember_confirmation(book, row['id'])
            except (ValueError, OSError):
                log.exception('Confirmation artifact could not be verified; receipt not imported')
    return len(rows)


def tick(operational, root: Path) -> dict:
    memory = Memory(operational.path.parent / 'memory.sqlite3')
    projected = memory.sync(operational)
    research = import_research(memory, root)
    research += import_research(memory, root / "memory-benchmark")
    from .outcomes import observe_outcomes
    outcomes = observe_outcomes(operational, memory)
    return {'projected': projected, 'research_records': research, 'outcomes': outcomes, 'at': datetime.now(UTC).isoformat()}


async def run(operational, root: Path, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.to_thread(tick, operational, root)
        except Exception:
            log.exception('Memory projection failed; durable outbox retained for retry')
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=30)


def main() -> None:
    import argparse

    from packages.core.db import Database
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--database', type=Path, required=True)
    ap.add_argument('--research-root', type=Path, default=Path('experiments/autonomous'))
    args = ap.parse_args()
    with Database(args.database) as db:
        print(json.dumps(tick(db, args.research_root)))


if __name__ == '__main__':
    main()
