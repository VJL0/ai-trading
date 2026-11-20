"""Pinned, local-only foundation-model artifacts and isolated inference contract."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / 'services/research/foundation_env/.venv/bin/python'
WEIGHTS = ROOT / 'var/models/foundation'
CATALOG = {
    'fincast': ('Vincent05R/FinCast', '2d7d90b159db8961d27c2cf165d51195902ef92b', 'apache-2.0'),
    'timestral': ('mistralai/Timestral-209M', '9874d252ab0aae16f7f7c999f98f98d6ad7d8365', 'modified-mit-revenue-limit'),
    'chronos2': ('amazon/chronos-2', '29ec3766d36d6f73f0696f85560a422f50e8498c', 'apache-2.0'),
    't0_gamma': ('theforecastingcompany/t0-gamma', 'fef67abad3eed4f227c7b536a6e8ee4e9ab1080d', 'apache-2.0'),
    'simons': ('abel-lab/Simons-297M', '335dc22e4ce26ca0a8bc567de9e93dbb1bd2ce3a', 'openmdw-1.1'),
    'tirex2': ('NX-AI/TiRex-2', '352636173653592b6dceb3fbeee9224ca97952a9', 'apache-2.0'),
    'timesfm25': ('google/timesfm-2.5-200m-pytorch', '1d952420fba87f3c6dee4f240de0f1a0fbc790e3', 'apache-2.0'),
    'fintext': ('FinText/Chronos_Tiny_2023_US', 'af8e48bae6d33f96b5b9bb4c077cbadfddafe840', 'apache-2.0'),
}


TABULAR_CATALOG = {
    'tabiclv2': ('jingang/TabICL', '4dcd344ece2c00be9e831fdd35bed57b5ad83e19', 'bsd-3-clause'),
    'kumo': ('nvidia/Kumo-Tabular', 'c5893d418e162b9f841ae179dc767afefff73831', 'openmdw-1.1'),
    'mitra': ('autogluon/mitra-regressor-2', '246c7d310fff04296a2388df5f448e3ffe809283', 'apache-2.0'),
}

def setup(name: str) -> dict:
    from huggingface_hub import snapshot_download
    repo, revision, license_name = (CATALOG | TABULAR_CATALOG)[name]
    directory = WEIGHTS / name
    patterns = ['*.json', '*.safetensors', 'README.md', 'LICENSE*', 'NOTICE*']
    if name == 'tabiclv2':
        patterns.append('tabicl-regressor-v2-20260212.ckpt')
    if name == 'kumo':
        patterns.append('small/regressor.pt')
    if name == 'fincast':
        patterns.append('v1.pth')
    snapshot_download(repo, revision=revision, local_dir=directory, token=False, allow_patterns=patterns)
    files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*')
             if p.is_file() and p.name != 'manifest.json' and '.cache' not in p.parts}
    manifest = {'model': name, 'repository': repo, 'revision': revision, 'license': license_name, 'files': files}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def forecast(name: str, histories: np.ndarray, horizon: int = 60) -> dict[str, np.ndarray]:
    x = np.asarray(histories, dtype=np.float32)
    if name not in CATALOG or x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all() or not 1 <= horizon <= 256:
        raise ValueError('finite [series,time] histories and supported horizon required')
    directory = WEIGHTS/name
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['revision'] != CATALOG[name][1]:
        raise ValueError('unregistered checkpoint revision')
    for file, expected in manifest['files'].items():
        p = directory/file
        if Path(file).name != file or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint integrity mismatch')
    with tempfile.TemporaryDirectory(prefix='lat-foundation-') as temp:
        run = Path(temp)
        np.save(run/'histories.npy', x, allow_pickle=False)
        (run/'request.json').write_text(json.dumps({'name': name, 'weights': str(directory), 'horizon': horizon}))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
        env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'  # noqa: S105 -- disables credentials for public checkpoints
        env.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.5', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.4',
                   HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
        completed = subprocess.run([str(PYTHON), '-m', 'services.research.advanced.foundation_worker', str(run)],
                                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
        if completed.returncode:
            raise RuntimeError(f'{name} inference failed: {completed.stderr[-3000:]}')
        with np.load(run/'forecast.npz', allow_pickle=False) as archive:
            result = dict(archive)
    q, levels = result['quantiles'], result['levels']
    if q.shape != (len(x), horizon, len(levels)) or not np.isfinite(q).all():
        raise ValueError('invalid probabilistic forecast')
    # Quantile rearrangement removes crossings without inventing probabilistic paths.
    result['quantiles'] = np.sort(q, axis=-1)
    return result
