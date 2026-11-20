"""Local, numeric-weight-only Kronos adapter with reproducible retained sample paths."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

UPSTREAM_COMMIT = '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
DEFAULT_ROOT = Path('var/models/kronos')
COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ForecastConfig:
    variant: str = 'base'
    context: int = 128
    samples: int = 8
    temperature: float = 1.0
    top_p: float = .9
    seed: int = 42
    device: str = 'auto'

    def __post_init__(self) -> None:
        if self.variant not in ('base', 'small') or not 2 <= self.context <= 512:
            raise ValueError('Kronos base/small require context in [2,512]')
        if not 2 <= self.samples <= 128 or not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError('invalid sampling configuration (at least two paths required)')
        if self.device not in ('auto', 'cpu', 'mps'):
            raise ValueError('device must be auto, cpu or mps')


@dataclass
class Forecast:
    paths: np.ndarray  # sample, future bar, OHLCVA; raw model draws, never averaged upstream
    timestamps: pd.DatetimeIndex
    metadata: dict[str, Any]

    def features(self, horizons: tuple[int, ...], cost_bps: float = 0) -> dict[str, float]:
        """Next-open to horizon-open returns match the suite's execution convention.

        h requires h+1 forecast bars. Dispersion is sampling uncertainty, not a calibrated confidence interval.
        """
        result = {}
        for h in horizons:
            if h < 1 or h >= self.paths.shape[1]:
                raise ValueError('horizon requires h+1 predicted bars')
            r = (self.paths[:, h, 0] / self.paths[:, 0, 0] - 1) * 10000 - cost_bps
            last_close = float(self.metadata.get('last_close', self.paths[0, 0, 3]))
            closes = np.column_stack([np.full(len(self.paths), last_close), self.paths[:, :h, 3]])
            log_returns = np.diff(np.log(closes), axis=1)
            for name, value in {'mean_bps': r.mean(), 'std_bps': r.std(ddof=1),
                                'q10_bps': np.quantile(r, .1), 'q50_bps': np.quantile(r, .5),
                                'q90_bps': np.quantile(r, .9), 'p_positive': (r > 0).mean(),
                                'variance': np.mean(np.sum(log_returns**2, axis=1))}.items():
                result[f'h{h}_{name}'] = float(value)
        return result


def provision(root: Path = DEFAULT_ROOT, variants: tuple[str, ...] = ('base',)) -> dict[str, Any]:
    """Explicit setup only. Workers never download weights or execute Hub code."""
    from huggingface_hub import HfApi, hf_hub_download
    if not set(variants) <= {'base', 'small'}:
        raise ValueError('unsupported model variant')
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in ('tokenizer', *variants):
        repo = 'NeoQuasar/Kronos-Tokenizer-base' if name == 'tokenizer' else f'NeoQuasar/Kronos-{name}'
        revision = HfApi(token=False).model_info(repo).sha
        files = {}
        for filename in ('config.json', 'model.safetensors'):
            path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=root / name, token=False))
            files[filename] = sha(path)
        entries[name] = {'repo': repo, 'revision': revision, 'files': files}
    manifest = {'upstream_commit': UPSTREAM_COMMIT, 'models': entries, 'pretraining_cutoff': None,
                'pretraining_overlap': 'UNKNOWN; historical OOS is not proven pretraining-independent'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class KronosAdapter:
    model: Any
    tokenizer: Any
    def __init__(self, config: ForecastConfig = ForecastConfig(), root: Path = DEFAULT_ROOT) -> None:
        import torch
        from safetensors.torch import load_file

        from .vendor.kronos import Kronos, KronosTokenizer
        self.config = config
        self.root = root.resolve()
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest['upstream_commit'] != UPSTREAM_COMMIT:
            raise ValueError('Kronos source pin mismatch')
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if config.device == 'auto' else config.device
        self.fallback_reason = None
        for name, cls in [('tokenizer', KronosTokenizer), (config.variant, Kronos)]:
            entry = self.manifest['models'][name]
            for filename in ('config.json', 'model.safetensors'):
                if sha(root / name / filename) != entry['files'][filename]:
                    raise ValueError(f'Kronos artifact integrity mismatch: {name}/{filename}')
            model = cls(**json.loads((root / name / 'config.json').read_text()))
            model.load_state_dict(load_file(str(root / name / 'model.safetensors')), strict=True)
            model = model.float().to(self.device).eval()
            setattr(self, 'tokenizer' if name == 'tokenizer' else 'model', model)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def _cpu(self, reason: str) -> None:
        self.device = 'cpu'
        self.fallback_reason = reason
        for model in (self.model, self.tokenizer):
            model.to('cpu').eval()
            # Upstream RoPE caches are ordinary tensors, not registered buffers.
            for module in model.modules():
                if hasattr(module, 'seq_len_cached'):
                    module.seq_len_cached = None

    def forecast(self, frame: pd.DataFrame, future: pd.DatetimeIndex, *, decision: pd.Timestamp,
                 seed: int | None = None) -> Forecast:
        import torch
        frame = frame.tail(self.config.context).copy()
        if len(frame) < 2 or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError('at least two timezone-aware historical bars required')
        if decision.tzinfo is None or future.tz is None or not 1 <= len(future) <= 512:
            raise ValueError('timezone-aware decision/future and 1..512 future bars required')
        if (not frame.index.is_monotonic_increasing or frame.index.has_duplicates
                or cast(pd.Timestamp, frame.index[-1]) >= cast(pd.Timestamp, future[0])):
            raise ValueError('historical timestamps must be unique and precede forecasts')
        if not future.is_monotonic_increasing or future.has_duplicates or cast(pd.Timestamp, future[0]) < decision:
            raise ValueError('future calendar must be strictly ordered and start at/after decision')
        if cast(pd.Timestamp, frame.index[-1]) >= decision:
            raise ValueError('context timestamps must precede decision')
        if 'available_at_ts' not in frame or (pd.to_datetime(frame.available_at_ts, utc=True) > decision).any():
            raise ValueError('context contains unpublished bars or lacks availability timestamps')
        if 'amount' not in frame:
            frame['amount'] = frame['volume'] * frame[['open', 'high', 'low', 'close']].mean(axis=1)
        x = frame[COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or (x[:, :4] <= 0).any() or (x[:, 4:] < 0).any():
            raise ValueError('nonfinite/invalid OHLCVA context')
        if (x[:, 1] < x[:, [0, 2, 3]].max(axis=1)).any() or (x[:, 2] > x[:, [0, 1, 3]].min(axis=1)).any():
            raise ValueError('invalid historical OHLC geometry')
        chosen_seed = self.config.seed if seed is None else seed
        try:
            paths = self._generate(x, frame.index, future, chosen_seed)
        except (NotImplementedError, RuntimeError) as exc:
            if self.device != 'mps' or not any(s in str(exc).lower() for s in ('mps', 'metal', 'not implemented')):
                raise
            self._cpu(str(exc))
            paths = self._generate(x, frame.index, future, chosen_seed)
        if not np.isfinite(paths).all() or (paths[:, :, :4] <= 0).any():
            raise ValueError('invalid generated prices; refusing forecast')
        # Preserve raw paths; report invalid candle/volume draws, never silently manufacture bars.
        invalid = ((paths[:, :, 1] < paths[:, :, [0, 2, 3]].max(axis=2))
                   | (paths[:, :, 2] > paths[:, :, [0, 1, 3]].min(axis=2)) | (paths[:, :, 4:] < 0).any(axis=2))
        return Forecast(paths, future, {'config': asdict(self.config), 'seed': chosen_seed, 'device': self.device,
                                      'last_close': float(x[-1, 3]), 'cpu_fallback': self.fallback_reason,
                                      'parameter_count': self.parameter_count,
                                      'invalid_candle_fraction': float(invalid.mean()),
                                      'weights': self.manifest, 'torch_version': torch.__version__})

    def _generate(self, x: np.ndarray, history: pd.DatetimeIndex, future: pd.DatetimeIndex, seed: int) -> np.ndarray:
        import torch

        from .vendor.kronos import auto_regressive_inference, calc_time_stamps
        self.model.eval()
        self.tokenizer.eval()
        mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
        # Samples are independent BATCH rows, with sample_count=1, avoiding upstream path averaging.
        batch = np.repeat(((x-mean)/std)[None], self.config.samples, axis=0)
        def stamps(index: pd.DatetimeIndex) -> Any:
            values = calc_time_stamps(pd.Series(index.tz_convert('America/New_York'))).to_numpy(dtype=np.float32)
            return torch.tensor(np.repeat(values[None], self.config.samples, axis=0), device=self.device)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(seed)
            if self.device == 'mps':
                torch.mps.manual_seed(seed)
            paths = auto_regressive_inference(self.tokenizer, self.model,
                                             torch.tensor(batch, dtype=torch.float32, device=self.device),
                                             stamps(history), stamps(future), 512, len(future),
                                             T=self.config.temperature, top_k=0, top_p=self.config.top_p,
                                             sample_count=1, verbose=False)
        return paths[:, -len(future):] * std + mean

    def forecast_many(self, requests: list[tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp]]) -> list[Forecast]:
        # Ragged sequences are never zero padded: normalization/tokenization sees only valid bars.
        # This also avoids device-dependent padded attention behavior during autoregressive inference.
        return [self.forecast(frame, future, decision=decision) for frame, future, decision in requests]
