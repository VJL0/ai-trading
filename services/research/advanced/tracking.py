"""Count bounded tournaments as research trials; they cannot bypass promotion screening."""
import json
from pathlib import Path

from ..autonomous.engine import Journal, code_hash
from ..autonomous.learners import Hypothesis
from ..models.registry import register_research_challenger
from ..provenance import dumps
from .foundation import ROOT


def record_tournament(output: Path):
    report = json.loads((output/'report.json').read_text())
    journal = Journal(ROOT/'experiments/autonomous')
    for family, model in report['models'].items():
        spec = Hypothesis(family, capacity=32, epochs=3, diagnostic_rows=len(report['folds'][0]['train']),
                          diagnostic_test_rows=len(report['folds'][0]['test']))
        job = journal.enqueue(report['dataset'], spec)
        if not journal.claim(job):
            continue
        if model['status'] != 'COMPLETED':
            journal.finish(job, error=f'bounded tournament worker failed; see {model["log"]}')
            continue
        result = {'dataset_key': report['dataset'], 'code_hash': code_hash(), 'metrics': model['metrics'],
                  'risk': model['risk'], 'stress_2x_costs': model['stress_2x'], 'folds': model['fold_metrics'],
                  'is_simulated': report['is_simulated'], 'objective': -1e9, 'p_positive': 1., 'p_incremental': 1.,
                  'sessions': 0, 'validation': 'BOUNDED_DEVELOPMENT_DIAGNOSTIC', 'promotion_eligible': False,
                  'promotion_blockers': model['promotion_blockers'], 'paired_report': str(output.resolve()/'report.json')}
        directory = journal.root/'runs'/job
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'result.json').write_text(dumps(result))
        register_research_challenger(journal.root.parent, job, spec.describe(), result, directory)
        journal.finish(job, result=result)
        model['journal_job'] = job
    (output/'report.json').write_text(dumps(report))
