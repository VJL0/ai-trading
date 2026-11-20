"""Paired volatility forecasts and filtered regime features on development-only windows."""
import json
from pathlib import Path

import numpy as np

from ..models.features import VolSpec
from ..models.vol import ewma_forecast, fit_har, vol_metrics
from ..provenance import dumps
from .statistical import FilteredHMM, GJRGarch, MarkovSwitching


def benchmark_statistics(cache: Path, paired_report: Path, output: Path):
    manifest = json.loads((cache/'manifest.json').read_text())
    folds = json.loads(paired_report.read_text())['folds']
    with np.load(cache/'development.npz', allow_pickle=False) as archive:
        a = dict(archive)
    actual, predictions, regimes = [], {'har_rv': [], 'ewma': [], 'gjr_garch': []}, []
    for fold in folds:
        train, test = np.asarray(fold['train']), np.asarray(fold['test'])
        vs = VolSpec(**manifest['vol_spec'])
        har = fit_har(a['vol_X'][train], a['y'][train, 1]**2, vs)
        predictions['har_rv'].extend(har.forecast(a['vol_X'][test]))
        predictions['ewma'].extend(ewma_forecast(a['vol_X'][test], vs))
        actual.extend(a['y'][test, 1]**2)
        fitted = {}
        for symbol in np.unique(a['symbol_id'][test]):
            cutoff = a['decision'][train].max()
            stop = np.searchsorted(a[f'avail_{symbol}'], cutoff, side='right')
            # Keep a bounded recent training window; reject overnight/non-minute jumps.
            prices = a[f'bars_{symbol}'][max(0, stop-4001):stop, 3].astype(float)
            starts = a[f'start_{symbol}'][max(0, stop-4001):stop]
            returns = np.diff(np.log(prices))[np.diff(starts) == 60_000_000_000]
            gjr = GJRGarch().fit(returns)
            hmm = FilteredHMM().fit(np.column_stack([returns, abs(returns)]))
            markov = MarkovSwitching().fit(returns[-1000:])
            fitted[int(symbol)] = gjr, hmm, markov
        for row in test:
            symbol = int(a['symbol_id'][row])
            stop = np.searchsorted(a[f'avail_{symbol}'], a['decision'][row], side='right')
            prices = a[f'bars_{symbol}'][stop-256:stop, 3].astype(float)
            starts = a[f'start_{symbol}'][stop-256:stop]
            returns = np.diff(np.log(prices))[np.diff(starts) == 60_000_000_000]
            gjr, hmm, markov = fitted[symbol]
            predictions['gjr_garch'].append(gjr.forecast(returns)[60])
            regimes.append({'row': int(row), 'hmm': hmm.filtered(np.column_stack([returns, abs(returns)]))[-1].tolist(),
                            'markov_switching': markov.filtered(returns)[-1].tolist()})
    report = {'dataset': manifest['dataset_key'], 'scope': 'development only; filtered regime features, not tradable edge',
              'volatility': {name: vol_metrics(np.asarray(actual), np.asarray(values)) for name, values in predictions.items()},
              'regimes': regimes, 'promotion_eligible': False}
    output.write_text(dumps(report))
    return report
