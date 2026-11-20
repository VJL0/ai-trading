"""Chronological walk-forward splits with purging, embargo and a frozen final test interval (spec §3).

Definitions (all times are int64 ns; every sample ``i`` has a decision time ``d_i`` and a label interval
``[d_i, e_i]`` where ``e_i`` is when its forward-return label becomes known):

* **Frozen final test**: the last ``final_test_fraction`` of the unique decision times. Samples with
  ``d_i >= T_f`` are never used for fitting, calibration or model selection. Development samples are
  purged/embargoed against it exactly like any other test window. A sha256 of the frozen-test sample keys
  is recorded so later runs can prove which interval was (and was not) evaluated.
* **Walk-forward folds**: after an initial training block (``min_train_fraction`` of the development
  times) the remaining development times are cut into ``n_folds`` contiguous test windows ``[a_k, b_k]``.
  Training for fold ``k`` uses only development samples with ``d_i < a_k`` (expanding window).
* **Purging**: a training sample is dropped when its label interval overlaps the test window,
  i.e. ``d_i <= b_k and e_i >= a_k``.
* **Embargo**: additionally drop training samples whose decision time lies within ``embargo_steps``
  unique decision times of the test window on either side (``U[ia - E] <= d_i <= U[ib + E]``). In a
  forward-only walk-forward this is the gap between the last training decision and the test window,
  which limits leakage through serially-correlated features.
* **Calibration fold**: inside each training set the last ``calib_fraction`` of its decision times is held
  out for probability calibration; the fitting part is purged/embargoed against it the same way.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class InsufficientDataError(ValueError):
    """Not enough distinct decision times to build the requested splits."""


@dataclass(frozen=True)
class Fold:
    fold: int
    fit_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    test_start_ns: int
    test_end_ns: int
    n_purged: int
    n_embargoed: int

    @property
    def train_idx(self) -> np.ndarray:
        return np.union1d(self.fit_idx, self.calib_idx)

    def describe(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "fit": len(self.fit_idx),
            "calib": len(self.calib_idx),
            "test": len(self.test_idx),
            "test_start_ns": self.test_start_ns,
            "test_end_ns": self.test_end_ns,
            "purged": self.n_purged,
            "embargoed": self.n_embargoed,
        }


@dataclass(frozen=True)
class SplitPlan:
    folds: tuple[Fold, ...]
    dev_idx: np.ndarray
    final_fit_idx: np.ndarray
    final_calib_idx: np.ndarray
    final_test_idx: np.ndarray
    final_test_start_ns: int
    final_test_hash: str
    dev_purged_vs_final: int
    dev_embargoed_vs_final: int
    spec: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "dev_samples": len(self.dev_idx),
            "dev_purged_vs_final_test": self.dev_purged_vs_final,
            "dev_embargoed_vs_final_test": self.dev_embargoed_vs_final,
            "final_fit_samples": len(self.final_fit_idx),
            "final_calib_samples": len(self.final_calib_idx),
            "final_test_samples": len(self.final_test_idx),
            "final_test_start_ns": self.final_test_start_ns,
            "final_test_hash": self.final_test_hash,
            "folds": [f.describe() for f in self.folds],
        }


def purge_and_embargo(
    candidate_idx: np.ndarray,
    decision_ns: np.ndarray,
    label_end_ns: np.ndarray,
    *,
    test_start_ns: int,
    test_end_ns: int,
    embargo_lo_ns: int,
    embargo_hi_ns: int,
) -> tuple[np.ndarray, int, int]:
    """Remove purged (label overlaps test) and embargoed samples. Returns (kept, n_purged, n_embargoed)."""
    c = np.asarray(candidate_idx, dtype=np.int64)
    d = decision_ns[c]
    e = label_end_ns[c]
    overlap = (d <= test_end_ns) & (e >= test_start_ns)
    embargo = (d >= embargo_lo_ns) & (d <= embargo_hi_ns) & ~overlap
    keep = ~overlap & ~embargo
    return c[keep], int(overlap.sum()), int(embargo.sum())


def _embargo_bounds(unique_ns: np.ndarray, start_ns: int, end_ns: int, steps: int) -> tuple[int, int]:
    ia = int(np.searchsorted(unique_ns, start_ns, side="left"))
    ib = int(np.searchsorted(unique_ns, end_ns, side="right")) - 1
    lo = int(unique_ns[max(0, ia - steps)])
    hi = int(unique_ns[min(len(unique_ns) - 1, max(ib, 0) + steps)])
    return lo, hi


def holdout_split(
    train_idx: np.ndarray,
    decision_ns: np.ndarray,
    label_end_ns: np.ndarray,
    unique_ns: np.ndarray,
    calib_fraction: float,
    embargo_steps: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Hold out the last ``calib_fraction`` of ``train_idx``'s decision times; purge/embargo the rest against it.

    Returns ``(fit_idx, holdout_idx, n_purged, n_embargoed)``.
    """
    times = np.unique(decision_ns[train_idx])
    if len(times) < 4 or calib_fraction <= 0:
        return train_idx, np.empty(0, dtype=np.int64), 0, 0
    c0 = int(np.floor(len(times) * (1.0 - calib_fraction)))
    c0 = min(max(c0, 1), len(times) - 1)
    calib_start = int(times[c0])
    calib_end = int(times[-1])
    calib_idx = train_idx[decision_ns[train_idx] >= calib_start]
    fit_cand = train_idx[decision_ns[train_idx] < calib_start]
    lo, hi = _embargo_bounds(unique_ns, calib_start, calib_end, embargo_steps)
    fit_idx, n_p, n_e = purge_and_embargo(
        fit_cand, decision_ns, label_end_ns, test_start_ns=calib_start, test_end_ns=calib_end, embargo_lo_ns=lo,
        embargo_hi_ns=hi,
    )
    if len(fit_idx) == 0:
        # purge/embargo consumed the whole fit block: no calibration hold-out (probabilities stay UNCALIBRATED)
        return train_idx, np.empty(0, dtype=np.int64), 0, 0
    return fit_idx, calib_idx, n_p, n_e


def frozen_test_hash(keys: Sequence[str], start_ns: int) -> str:
    payload = json.dumps({"final_test_start_ns": start_ns, "n": len(keys), "keys": sorted(keys)}, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def plan_walk_forward(
    decision_ns: np.ndarray,
    label_end_ns: np.ndarray,
    *,
    keys: Sequence[str] | None = None,
    n_folds: int = 4,
    final_test_fraction: float = 0.2,
    embargo_steps: int = 5,
    min_train_fraction: float = 0.4,
    calib_fraction: float = 0.25,
    min_unique_times: int = 40,
) -> SplitPlan:
    """Build the walk-forward plan (see module docstring). Arrays are aligned per sample."""
    d = np.asarray(decision_ns, dtype=np.int64)
    e = np.asarray(label_end_ns, dtype=np.int64)
    if d.shape != e.shape or d.ndim != 1:
        raise ValueError("decision_ns and label_end_ns must be aligned 1-d arrays")
    if (e < d).any():
        raise ValueError("label_end precedes decision time for some samples")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0 < final_test_fraction < 0.5:
        raise ValueError("final_test_fraction must be in (0, 0.5)")
    if not 0 < min_train_fraction < 0.9:
        raise ValueError("min_train_fraction must be in (0, 0.9)")
    if embargo_steps < 0:
        raise ValueError("embargo_steps must be >= 0")
    unique_ns = np.unique(d)
    m = len(unique_ns)
    if m < min_unique_times:
        raise InsufficientDataError(f"only {m} distinct decision times (< {min_unique_times})")
    all_idx = np.arange(len(d), dtype=np.int64)

    # ---- frozen final test
    f0 = int(np.floor(m * (1.0 - final_test_fraction)))
    final_start = int(unique_ns[f0])
    final_test_idx = all_idx[d >= final_start]
    lo, _ = _embargo_bounds(unique_ns, final_start, int(unique_ns[-1]), embargo_steps)
    dev_idx, dev_p, dev_e = purge_and_embargo(
        all_idx[d < final_start], d, e, test_start_ns=final_start, test_end_ns=int(max(e.max(), unique_ns[-1])),
        embargo_lo_ns=lo, embargo_hi_ns=int(unique_ns[-1]),
    )
    dev_times = np.unique(d[dev_idx])
    md = len(dev_times)
    first = int(np.ceil(md * min_train_fraction))
    if md - first < n_folds * 2 or first < 2:
        raise InsufficientDataError(f"{md} development decision times cannot support {n_folds} folds")

    folds: list[Fold] = []
    for k, block in enumerate(np.array_split(np.arange(first, md), n_folds)):
        a = int(dev_times[block[0]])
        b = int(dev_times[block[-1]])
        test_idx = dev_idx[(d[dev_idx] >= a) & (d[dev_idx] <= b)]
        lo_k, hi_k = _embargo_bounds(unique_ns, a, b, embargo_steps)
        train_idx, n_p, n_e = purge_and_embargo(
            dev_idx[d[dev_idx] < a], d, e, test_start_ns=a, test_end_ns=b, embargo_lo_ns=lo_k, embargo_hi_ns=hi_k
        )
        if len(train_idx) == 0:
            raise InsufficientDataError(f"fold {k} has no training samples left after purging/embargo")
        fit_idx, calib_idx, _, _ = holdout_split(train_idx, d, e, unique_ns, calib_fraction, embargo_steps)
        folds.append(Fold(k, fit_idx, calib_idx, test_idx, a, b, n_p, n_e))

    final_fit, final_calib, _, _ = holdout_split(dev_idx, d, e, unique_ns, calib_fraction, embargo_steps)
    key_list = [str(keys[i]) for i in final_test_idx] if keys is not None else [str(int(i)) for i in final_test_idx]
    return SplitPlan(
        folds=tuple(folds),
        dev_idx=dev_idx,
        final_fit_idx=final_fit,
        final_calib_idx=final_calib,
        final_test_idx=final_test_idx,
        final_test_start_ns=final_start,
        final_test_hash=frozen_test_hash(key_list, final_start),
        dev_purged_vs_final=dev_p,
        dev_embargoed_vs_final=dev_e,
        spec={
            "n_folds": n_folds,
            "final_test_fraction": final_test_fraction,
            "embargo_steps": embargo_steps,
            "min_train_fraction": min_train_fraction,
            "calib_fraction": calib_fraction,
            "unique_decision_times": m,
            "scheme": "expanding walk-forward; purge label overlap; embargo +/- E decision steps; frozen final test",
        },
    )
