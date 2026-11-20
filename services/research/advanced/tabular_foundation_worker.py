# pyright: reportMissingImports=false
"""Bounded local in-context learning with real TabICLv2 and Kumo checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
request = json.loads((root/'request.json').read_text())
with np.load(root/'input.npz', allow_pickle=False) as a:
    X, y, query = a['X'], a['y'], a['query']
path = Path(request['weights'])
torch.set_num_threads(2)
torch.manual_seed(42)
# CPU avoids unsupported mixed-precision/attention kernels in these upstream tabular runtimes.
results = []
for col in range(y.shape[1]):
    if request['name'] == 'tabiclv2':
        from tabicl import TabICLRegressor
        model = TabICLRegressor(n_estimators=2, model_path=path/'tabicl-regressor-v2-20260212.ckpt',
                               allow_auto_download=False, device='cpu', use_amp=False, use_fa3=False, n_jobs=2)
        model.fit(X, y[:, col])
        pred = model.predict(query)
    elif request['name'] == 'mitra':
        from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
        from safetensors.torch import load_file
        config = json.loads((path/'config.json').read_text())
        model = Tab2D(**config, use_pretrained_weights=False, path_to_weights='', device='cpu')
        model.load_state_dict(load_file(path/'model.safetensors'), strict=True)
        model.eval()
        low, scale = float(y[:, col].min()), max(float(np.ptp(y[:, col])), 1e-8)
        edges = torch.linspace(-.5, 1.5, 1001)
        labels = (torch.bucketize(torch.tensor((y[:, col]-low)/scale, dtype=torch.float32), edges)-1).clamp(0, 999)
        centers = (edges[:-1]+edges[1:])/2
        with torch.inference_mode():
            logits = model(torch.tensor(X[None], dtype=torch.float32), labels[None],
                           torch.tensor(query[None], dtype=torch.float32),
                           torch.zeros((1, X.shape[1]), dtype=torch.bool),
                           torch.zeros((1, len(X)), dtype=torch.bool),
                           torch.zeros((1, len(query)), dtype=torch.bool))
            pred = (logits.softmax(-1)*centers).sum(-1)[0].numpy()*scale+low
    elif request['name'] == 'kumo':
        from sdm.models import KumoTabular
        model = KumoTabular(task='regression', size='small', pretrained=False, device='cpu')
        state = torch.load(path/'small/regressor.pt', map_location='cpu', weights_only=True)
        for network in model.models.values():
            network.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            model.fit(torch.tensor(X, dtype=torch.float32), torch.tensor(y[:, col:col+1], dtype=torch.float32),
                      num_estimators=2)
            output = model.predict(torch.tensor(query, dtype=torch.float32))
            quantiles = output.to_pandas().to_numpy().reshape(len(query), -1)
            pred = quantiles.mean(axis=1)
    else:
        raise ValueError('unknown tabular checkpoint')
    results.append(np.asarray(pred))
np.save(root/'output.npy', np.column_stack(results), allow_pickle=False)
