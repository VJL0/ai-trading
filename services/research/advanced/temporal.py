"""Trainable temporal architectures over real [batch,time,channel] histories, not feature tokens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ARCHITECTURES = ('patchtst', 'itransformer', 'moderntcn', 'multitask_transformer')


def network(family: str, length: int, channels: int, outputs: int, width: int = 32) -> Any:
    import torch
    from torch import nn
    if family not in ARCHITECTURES or width % 4 or length < 16:
        raise ValueError('invalid temporal architecture/shape')

    class TemporalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.family = family
            layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0, batch_first=True, norm_first=True)
            if family == 'patchtst':
                self.embed = nn.Linear(16, width)
                self.position = nn.Parameter(torch.randn(1, (length-16)//8+1, width)*.01)
                self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                hidden = channels*width
            elif family == 'itransformer':
                self.embed = nn.Linear(length, width)
                self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                hidden = channels*width
            elif family == 'moderntcn':
                # Large-kernel depthwise temporal mixing; channel/group pointwise mixing.
                hidden = channels*width
                self.embed = nn.Conv1d(channels, hidden, 3, padding=1, groups=channels)
                self.encoder = nn.Sequential(nn.Conv1d(hidden, hidden, 31, padding=15, groups=hidden),
                                             nn.GELU(), nn.Conv1d(hidden, hidden*2, 1, groups=channels), nn.GELU(),
                                             nn.Conv1d(hidden*2, hidden, 1, groups=channels), nn.GELU(),
                                             nn.Conv1d(hidden, hidden, 1))
            else:
                self.embed = nn.Linear(channels, width)
                self.position = nn.Parameter(torch.randn(1, length, width)*.01)
                self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                hidden = width
            self.heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(outputs))
            self.reconstruct = nn.Linear(hidden, length*channels)

        def representation(self, x):
            b, _, c = x.shape
            if self.family == 'patchtst':
                z = x.transpose(1, 2).unfold(-1, 16, 8).reshape(b*c, -1, 16)
                z = self.encoder(self.embed(z)+self.position).mean(dim=1).reshape(b, -1)
            elif self.family == 'itransformer':
                z = self.encoder(self.embed(x.transpose(1, 2))).flatten(1)
            elif self.family == 'moderntcn':
                z = self.embed(x.transpose(1, 2))
                z = (z+self.encoder(z)).mean(dim=-1)
            else:
                z = self.encoder(self.embed(x)+self.position).mean(dim=1)
            return z

        def forward(self, x):
            z = self.representation(x)
            return torch.cat([head(z) for head in self.heads], dim=1)

    return TemporalNetwork()


@dataclass
class TemporalLearner:
    family: str
    width: int = 32
    epochs: int = 8
    seed: int = 42
    pretrain_epochs: int = 0
    meta_steps: int = 0
    batch_size: int = 64

    def fit(self, X: np.ndarray, y: np.ndarray) -> TemporalLearner:
        import torch
        if X.ndim != 3 or len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('finite aligned temporal histories/targets required')
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.mean, self.scale = X.mean(axis=(0, 1)), np.maximum(X.std(axis=(0, 1)), 1e-6)
        self.ymean, self.yscale = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6)
        self.shape = X.shape[1:]
        self.model = network(self.family, int(self.shape[0]), int(self.shape[1]), y.shape[1], self.width).to(self.device)
        x = torch.as_tensor((X-self.mean)/self.scale, dtype=torch.float32)
        target = torch.as_tensor((y-self.ymean)/self.yscale, dtype=torch.float32)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.01)
        self.model.train()
        for _ in range(self.pretrain_epochs):
            for batch in x.split(self.batch_size):
                batch = batch.to(self.device)
                mask = torch.rand_like(batch) < .2
                decoded = self.model.reconstruct(self.model.representation(batch.masked_fill(mask, 0))).reshape_as(batch)
                loss = torch.nn.functional.mse_loss(decoded[mask], batch[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Reptile initialization across chronological, non-overlapping training episodes.
        for episode in range(self.meta_steps):
            sections = x.split(max(16, len(x)//4))
            targets = target.split(max(16, len(x)//4))
            k = episode % len(sections)
            before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
            inner = torch.optim.SGD(self.model.parameters(), lr=.01)
            for _ in range(3):
                loss = torch.nn.functional.mse_loss(self.model(sections[k].to(self.device)), targets[k].to(self.device))
                inner.zero_grad()
                loss.backward()
                inner.step()
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    parameter.copy_(before[name]+.1*(parameter-before[name]))
        self.losses = []
        for _ in range(self.epochs):
            epoch = []
            for indices in torch.randperm(len(x)).split(self.batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.huber_loss(self.model(x[indices].to(self.device)), target[indices].to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite temporal training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                epoch.append(float(loss.detach().cpu()))
            self.losses.append(float(np.mean(epoch)))
        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        if X.ndim != 3 or X.shape[1:] != self.shape:
            raise ValueError('temporal shape mismatch')
        self.model.eval()
        with torch.inference_mode():
            x = torch.as_tensor((X-self.mean)/self.scale, dtype=torch.float32)
            pred = np.concatenate([self.model(b.to(self.device)).cpu().numpy() for b in x.split(self.batch_size)])
        return pred*self.yscale+self.ymean
