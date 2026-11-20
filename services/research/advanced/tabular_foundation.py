"""Isolated in-context tabular prediction; fitted context contains training rows only."""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .foundation import PYTHON, ROOT, TABULAR_CATALOG, WEIGHTS


def predict(name, X, y, query):
    if name not in ('tabiclv2', 'kumo', 'mitra') or len(X) > 2048:
        raise ValueError('unregistered model or context budget exceeded')
    directory = WEIGHTS/name
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['revision'] != TABULAR_CATALOG[name][1]:
        raise ValueError('unregistered tabular checkpoint revision')
    for file, expected in manifest['files'].items():
        path = directory/file
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('invalid artifact path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint integrity mismatch')
    with tempfile.TemporaryDirectory(prefix='lat-tabular-') as temp:
        run = Path(temp)
        np.savez(run/'input.npz', X=X, y=y, query=query)
        (run/'request.json').write_text(json.dumps({'name': name, 'weights': str(directory)}))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
        env.update(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        done = subprocess.run([str(PYTHON), '-m', 'services.research.advanced.tabular_foundation_worker', str(run)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=900, check=False)
        if done.returncode:
            raise RuntimeError(done.stderr[-3000:])
        pred = np.load(run/'output.npy', allow_pickle=False)
    if pred.shape != (len(query), y.shape[1]) or not np.isfinite(pred).all():
        raise ValueError('invalid tabular foundation predictions')
    return pred
