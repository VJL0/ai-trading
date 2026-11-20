"""Bounded, fold-local token-head fine-tuning; tokenizer/backbone remain frozen."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .adapter import COLUMNS, KronosAdapter
from .data import context


def tune_heads(adapter: KronosAdapter, arrays: dict[str, np.ndarray], fit_idx: np.ndarray,
               steps: int, validation_start: int) -> dict[str, Any]:
    import torch

    from .vendor.kronos import calc_time_stamps
    if not 1 <= steps <= 128 or not len(fit_idx):
        raise ValueError('fine-tuning requires 1..128 bounded steps and training rows')
    if arrays['end'][fit_idx].max() >= validation_start:
        raise ValueError('fine-tuning label intervals overlap calibration/test')
    torch.manual_seed(adapter.config.seed)
    adapter.tokenizer.eval()
    adapter.model.requires_grad_(False)
    adapter.model.head.requires_grad_(True)
    adapter.model.train()
    optimizer = torch.optim.AdamW(adapter.model.head.parameters(), lr=1e-5, weight_decay=.01)
    rng = np.random.default_rng(adapter.config.seed)
    losses = []
    try:
        for row in rng.choice(fit_idx, steps, replace=True):
            frame = context(arrays, int(row), adapter.config.context)
            x = frame[COLUMNS].to_numpy(dtype=np.float32)
            # Window is wholly inside fit data. Prefix-only statistics avoid target normalization leakage.
            mean, std = x[:-1].mean(axis=0), x[:-1].std(axis=0) + 1e-5
            values = torch.tensor(np.clip((x-mean)/std, -5, 5)[None], device=adapter.device)
            local = pd.Series(pd.DatetimeIndex(frame.index).tz_convert('America/New_York'))
            stamp = calc_time_stamps(local).to_numpy(dtype=np.float32)
            stamp = torch.tensor(stamp[None, :-1], device=adapter.device)
            with torch.no_grad():
                s1, s2 = adapter.tokenizer.encode(values, half=True)
            optimizer.zero_grad()
            l1, l2 = adapter.model(s1[:, :-1], s2[:, :-1], stamp, use_teacher_forcing=True, s1_targets=s1[:, 1:])
            loss, _, _ = adapter.model.head.compute_loss(l1, l2, s1[:, 1:], s2[:, 1:])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite Kronos fine-tuning loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.head.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    finally:
        adapter.model.eval()
        adapter.tokenizer.eval()
    return {'method': 'token_head_fine_tuning', 'steps': steps, 'losses': losses,
            'fit_end_ns': int(arrays['end'][fit_idx].max()), 'validation_start_ns': int(validation_start),
            'learning_rate': 1e-5, 'backbone_frozen': True, 'tokenizer_frozen': True}
