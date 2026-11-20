"""Run one challenger on the parent's identical purged folds, in an isolated interpreter."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..autonomous.learners import Hypothesis, fit
from .inputs import model_inputs

root = Path(sys.argv[1])
request = json.loads((root/'request.json').read_text())
with np.load(request['cache'], allow_pickle=False) as archive:
    arrays = dict(archive)
family = request['family']
spec = Hypothesis(family, capacity=32, epochs=3)
all_rows = np.unique(np.concatenate([np.asarray(f[k], int) for f in request['folds'] for k in ('train', 'cal', 'test')]))
if family.startswith('foundation_'):
    from .foundation import forecast
    if family == 'foundation_fintext' and np.any(arrays['decision'][all_rows] < 1704067200000000000):
        raise ValueError('FinText 2023 checkpoint requires post-2023 evaluation rows')
    histories = []
    for row in all_rows:
        symbol = int(arrays['symbol_id'][row])
        stop = np.searchsorted(arrays[f'avail_{symbol}'], arrays['decision'][row], side='right')
        history = arrays[f'bars_{symbol}'][stop-128:stop, 3]
        if len(history) != 128:
            raise ValueError('incomplete foundation context')
        histories.append(history)
    histories = np.asarray(histories)
    outputs = []
    for offset in range(0, len(histories), 512):
        batch = histories[offset:offset+512]
        name = family.removeprefix('foundation_')
        model_history = np.diff(np.log(batch), axis=1) if name == 'fintext' else batch
        out = forecast(name, model_history, 60)['quantiles']
        if name == 'fintext':
            # Marginal return quantiles are features, not a fabricated joint-price path.
            features = out[:, [4, 14, 59], :].reshape(len(batch), -1)
        else:
            features = (out[:, [4, 14, 59], :]/batch[:, -1, None, None]-1).reshape(len(batch), -1)*10000
        outputs.append(features)
    embedding = np.concatenate(outputs)
    row_positions = {int(row): pos for pos, row in enumerate(all_rows)}

    def embedded_inputs(rows):
        return np.column_stack([arrays['X'][rows], embedding[[row_positions[int(i)] for i in rows]]])
    inputs = embedded_inputs
    spec = Hypothesis('ridge', alpha=10.)
else:
    def tabular_inputs(rows):
        return model_inputs(spec, arrays, rows)
    inputs = tabular_inputs

results = []
for fold in request['folds']:
    train, cal, test = (np.asarray(fold[k], int) for k in ('train', 'cal', 'test'))
    started = time.monotonic()
    if family in ('tabiclv2', 'kumo', 'mitra'):
        from .tabular_foundation import predict as context_predict
        combined = np.concatenate([cal, test])
        predicted = context_predict(family, arrays['X'][train], arrays['y'][train], arrays['X'][combined])
        calibration, pred = predicted[:len(cal)], predicted[len(cal):]
        seconds = time.monotonic()-started
        allowance = max(0., float(np.quantile(calibration[:, 0]-arrays['y'][cal, 0], .9)))
        results.append({'rows': test.tolist(), 'predictions': pred.tolist(), 'allowance': allowance,
                        'train_seconds': seconds, 'head_latency_ms_per_row': seconds*1000/len(combined)})
        continue
    model = fit(spec, inputs(train), arrays['y'][train])
    train_seconds = time.monotonic()-started
    calibration = model.predict(inputs(cal))
    allowance = max(0., float(np.quantile(calibration[:, 0]-arrays['y'][cal, 0], .9)))
    query = inputs(test)
    started = time.monotonic()
    pred = model.predict(query)
    seconds = time.monotonic()-started
    results.append({'rows': test.tolist(), 'predictions': pred.tolist(), 'allowance': allowance,
                    'train_seconds': train_seconds, 'head_latency_ms_per_row': seconds*1000/len(test)})
(root/'output.json').write_text(json.dumps(results, allow_nan=False))
