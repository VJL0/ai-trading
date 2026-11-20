"""Foundation evidence participates in the existing journal and research registry."""
import hashlib
from pathlib import Path

from ..autonomous.engine import code_hash
from ..models.registry import register_research_challenger
from ..provenance import dumps
from .benchmark import benchmark


def evaluate_foundation(root, job, spec, cache, manifest):
    from .foundation import WEIGHTS
    fingerprint = hashlib.sha256((WEIGHTS/spec.family.removeprefix('foundation_')/'manifest.json').read_bytes()).hexdigest()
    if fingerprint != spec.foundation_artifact_hash:
        raise ValueError('checkpoint manifest changed after enqueue')
    directory = root/'runs'/job
    directory.mkdir(parents=True, exist_ok=True)
    tournament = directory/'paired'
    if tournament.exists():
        raise ValueError('immutable tournament already exists; submit a distinct hypothesis to rerun')
    report = benchmark(cache, tournament, ['ridge', 'lightgbm', 'catboost', spec.family], train_rows=128, test_rows=16)
    model = report['models'][spec.family]
    if model['status'] != 'COMPLETED':
        raise RuntimeError(f'foundation worker failed: {model["log"]}')
    result = {'dataset_key': manifest['dataset_key'], 'code_hash': code_hash(), 'metrics': model['metrics'],
              'risk': model['risk'], 'stress_2x_costs': model['stress_2x'], 'folds': model['fold_metrics'],
              'objective': -1e9, 'sessions': 0, 'p_positive': 1., 'p_incremental': 1.,
              'is_simulated': manifest['is_simulated'], 'validation': 'DEVELOPMENT_ONLY',
              'promotion_eligible': False, 'promotion_blockers': model['promotion_blockers'],
              'paired_report': str(tournament/'report.json')}
    (directory/'result.json').write_text(dumps(result))
    register_research_challenger(Path(root).parent, job, spec.describe(), result, directory)
    return result
