"""Run a fixed memory ablation on a verified, already-exposed development cache.

Never load raw datasets or holdouts. Preserve original preprocessing provenance; bind
new evaluation code independently. All six attempts share one journal and trial budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from services.research.autonomous.engine import Journal, code_hash, evaluate, identity
from services.research.autonomous.learners import Hypothesis

from .ablation import MODES
from .comparison import compare


def run(source: Path, root: Path) -> dict:
    manifest = json.loads((source / 'manifest.json').read_text())
    data = source / 'development.npz'
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != manifest['cache_sha256'] or manifest['holdout_status'] != 'UNTOUCHED_NOT_IN_WORKER_CACHE':
        raise ValueError('requires a verified development-only artifact')
    evaluation_hash = code_hash()
    key = identity({'source_manifest': manifest, 'evaluation_hash': evaluation_hash, 'protocol': 'memory-v1'})
    directory = root / 'cache' / key
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data, directory / 'development.npz')
    derived = {**manifest, 'source_preprocessing_code_hash': manifest['code_hash'],
               'source_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
               'source_dataset_key': manifest['dataset_key'], 'dataset_key': key, 'code_hash': evaluation_hash}
    (directory / 'manifest.json').write_text(json.dumps(derived, indent=2))
    journal = Journal(root)
    for mode in ('', *MODES):
        jid = journal.enqueue(key, Hypothesis('ridge', memory_mode=mode))
        if journal.claim(jid):
            try:
                result = evaluate(root, jid)
                result['promotion_blockers'] = ['Cached development memory benchmark; no independent confirmation']
                journal.finish(jid, result)
            except Exception as exc:
                journal.finish(jid, error=f'{type(exc).__name__}: {exc}')
        print(json.dumps({'mode': mode or 'baseline', 'job': jid}), flush=True)
    report = compare(journal.rows(), key)
    report['source_manifest'] = derived
    (root / 'comparison.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('experiments/autonomous/memory-benchmark'))
    args = ap.parse_args()
    print(json.dumps(run(args.cache, args.root), indent=2))


if __name__ == '__main__':
    main()
