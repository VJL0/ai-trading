# pyright: reportMissingImports=false
"""One checkpoint per disposable process; never imports foundation runtimes in trading."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

run = Path(sys.argv[1])
r = json.loads((run/'request.json').read_text())
x = np.load(run/'histories.npy', allow_pickle=False)
name, path, horizon = r['name'], r['weights'], r['horizon']
torch.set_num_threads(2)
torch.manual_seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
levels = np.arange(1, 10)/10
with torch.inference_mode():
    if name == 'fincast':
        from .vendor.fincast.decoder import FFMConfig, PatchedTimeSeriesDecoder_MOE
        model = PatchedTimeSeriesDecoder_MOE(FFMConfig(num_experts=4))
        state = torch.load(Path(path)/'v1.pth', map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model.to(device).eval()
        batches = []
        for values in torch.from_numpy(x).split(8):
            values = values.to(device)
            # No padding: upstream masks use 1 for padded values.
            result, _ = model(values, torch.zeros_like(values),
                              torch.zeros((len(values), 1), dtype=torch.long, device=device))
            batches.append(result[:, -1, :horizon, 1:].cpu().numpy())
        q = np.concatenate(batches)
    elif name == 'timestral':
        device = 'cpu'  # Upstream causal scaler requires float64.
        from timestral import TimestralForecaster
        model = TimestralForecaster.from_pretrained(path, device=device, local_files_only=True)
        batches = []
        result = None
        for offset in range(0, len(x), 16):
            result = model.forecast(x[offset:offset+16, None, :], prediction_length=horizon)
            batches.append(np.asarray(result.quantiles)[:, 0])
        q = np.concatenate(batches)
        if result is None:
            raise ValueError('empty forecast batch')
        levels = np.asarray(result.quantile_levels)
    elif name == 'chronos2':
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(path, device_map=device, dtype=torch.float32, local_files_only=True)
        model.model.eval()
        quantiles, _ = model.predict_quantiles(x[:, None, :], prediction_length=horizon,
                                               quantile_levels=levels.tolist(), batch_size=32)
        q = np.stack([v.cpu().numpy().reshape(horizon, -1) for v in quantiles])
    elif name == 'fintext':
        from chronos import ChronosPipeline
        model = ChronosPipeline.from_pretrained(path, device_map=device, dtype=torch.float32, local_files_only=True)
        model.model.eval()
        batches = []
        for batch in torch.from_numpy(x).split(8):
            samples = model.predict(batch, prediction_length=horizon, num_samples=64)
            batches.append(np.quantile(samples.cpu().numpy(), levels, axis=1).transpose(1, 2, 0))
        q = np.concatenate(batches)
    elif name == 't0_gamma':
        from t0.mlx import T0Forecaster
        model = T0Forecaster.from_pretrained(path, local_files_only=True).eval()
        q = np.concatenate([np.asarray(model.predict(x[i:i+16], horizon=horizon,
                            quantile_levels=levels.tolist()).quantiles) for i in range(0, len(x), 16)])
        device = 'mlx'
    elif name == 'simons':
        # Official sliding normalization uses float64, unsupported by Metal.
        device = 'cpu'
        from simons import Simons
        model = Simons.from_pretrained(path, device=device).eval()
        q = np.concatenate([model.forecast(b.to(device), horizon=horizon).cpu().numpy()
                            for b in torch.from_numpy(x).split(16)])
    elif name == 'tirex2':
        from tirex2 import TimeseriesType, load_model
        model = load_model(path, device=device, use_flex_attention=False, compile=False, cuda_graphs=False)
        series = [TimeseriesType(torch.from_numpy(v[None]), None, None) for v in x]
        forecasts = model.forecast(series, prediction_length=horizon, output_type='numpy')
        q = np.stack([v[0].T for v in forecasts])
    elif name == 'timesfm25':
        import timesfm
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(path, local_files_only=True, torch_compile=False)
        model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                      use_continuous_quantile_head=True, fix_quantile_crossing=True))
        q = np.concatenate([model.forecast(horizon=horizon, inputs=list(x[i:i+16]))[1][..., 1:]
                            for i in range(0, len(x), 16)])
    else:
        raise ValueError('unregistered foundation model')
np.savez(run/'forecast.npz', quantiles=q, levels=levels, device=np.array(device))
