"""Explicit local setup and reproducible bounded development tournaments."""
import argparse
import json
from pathlib import Path

from .benchmark import benchmark
from .foundation import CATALOG, TABULAR_CATALOG, setup

parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest='command', required=True)
download = commands.add_parser('setup')
download.add_argument('model', choices=sorted(CATALOG | TABULAR_CATALOG))
bench = commands.add_parser('benchmark')
bench.add_argument('--cache', type=Path, required=True)
bench.add_argument('--output', type=Path, required=True)
bench.add_argument('--families', nargs='+', default=['ridge', 'elastic_net', 'lightgbm', 'catboost', 'xgboost',
    'double_ensemble', 'tabm', 'symbolic_gp', 'temporal_patchtst', 'temporal_itransformer',
    'temporal_moderntcn', 'temporal_multitask_transformer'])
bench.add_argument('--train-rows', type=int, default=256)
bench.add_argument('--test-rows', type=int, default=32)
statistics = commands.add_parser('statistics')
statistics.add_argument('--cache', type=Path, required=True)
statistics.add_argument('--paired-report', type=Path, required=True)
statistics.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.command == 'setup':
    print(json.dumps(setup(args.model), indent=2))
elif args.command == 'statistics':
    from .statistical_benchmark import benchmark_statistics
    print(json.dumps(benchmark_statistics(args.cache, args.paired_report, args.output), indent=2))
else:
    print(json.dumps(benchmark(args.cache, args.output, args.families, args.train_rows, args.test_rows), indent=2))
