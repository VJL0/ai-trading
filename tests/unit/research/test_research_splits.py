"""Walk-forward splits: purging, embargo, calibration hold-out and the frozen final test."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from services.research.splits import InsufficientDataError, plan_walk_forward


def _samples(n_times: int, n_symbols: int, horizon: int, step: int = 60) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n_times, dtype=np.int64) * step
    d = np.repeat(t, n_symbols)
    e = d + horizon * step
    return d, e


def _check_plan(d: np.ndarray, e: np.ndarray, plan_kwargs: dict[str, object]) -> None:
    plan = plan_walk_forward(d, e, **plan_kwargs)  # type: ignore[arg-type]
    unique = np.unique(d)
    E = int(plan_kwargs.get("embargo_steps", 5))  # type: ignore[arg-type]
    final = set(plan.final_test_idx.tolist())
    fs = plan.final_test_start_ns
    # frozen final test = every sample at/after its start; dev labels end strictly before it
    assert final == set(np.nonzero(d >= fs)[0].tolist())
    assert (e[plan.dev_idx] < fs).all()
    assert not final.intersection(plan.dev_idx.tolist())
    for f in plan.folds:
        a, b = f.test_start_ns, f.test_end_ns
        train = f.train_idx
        assert len(train) and len(f.test_idx)
        # purging: no training label interval [d_i, e_i] overlaps the test window [a, b]
        assert not ((d[train] <= b) & (e[train] >= a)).any()
        assert (e[train] < a).all() and (d[train] < a).all()
        # embargo: no training decision within E unique steps before the test window
        ia = int(np.searchsorted(unique, a))
        lo = unique[max(0, ia - E)]
        assert not ((d[train] >= lo) & (d[train] <= b)).any()
        # calibration block is later than the fit block and the fit block is purged against it
        assert len(f.fit_idx)
        if len(f.calib_idx):
            c0 = d[f.calib_idx].min()
            assert (e[f.fit_idx] < c0).all()
            assert d[f.fit_idx].max() < c0
        for idx in (f.fit_idx, f.calib_idx, f.test_idx):
            assert not final.intersection(idx.tolist())
        assert set(f.test_idx.tolist()) <= set(plan.dev_idx.tolist())
    starts = [f.test_start_ns for f in plan.folds]
    assert starts == sorted(starts)
    for f1, f2 in zip(plan.folds, plan.folds[1:], strict=False):
        assert f1.test_end_ns < f2.test_start_ns  # contiguous, non-overlapping test windows
    assert not set(plan.final_fit_idx.tolist()).intersection(plan.final_calib_idx.tolist())
    assert (e[plan.final_fit_idx] < d[plan.final_calib_idx].min()).all()


def test_basic_plan_is_purged_and_embargoed() -> None:
    d, e = _samples(300, 3, horizon=5)
    _check_plan(d, e, {"n_folds": 4, "embargo_steps": 5})


def test_purging_removes_long_labels_overlapping_test() -> None:
    d, e = _samples(200, 1, horizon=1)
    e = e.copy()
    # one early sample whose label reaches far into the future must be purged from every later fold
    e[10] = d[-1]
    plan = plan_walk_forward(d, e, n_folds=3, embargo_steps=0)
    assert 10 not in plan.dev_idx  # its label overlaps the frozen final test
    assert plan.dev_purged_vs_final >= 1
    for f in plan.folds:
        assert 10 not in f.train_idx


def test_embargo_counts_reported() -> None:
    d, e = _samples(200, 2, horizon=3)
    plan = plan_walk_forward(d, e, n_folds=3, embargo_steps=4)
    assert all(f.n_purged > 0 for f in plan.folds)
    assert all(f.n_embargoed > 0 for f in plan.folds)
    desc = plan.describe()
    assert desc["final_test_hash"].startswith("sha256:")
    assert len(desc["folds"]) == 3


def test_final_test_hash_depends_on_keys_only() -> None:
    d, e = _samples(120, 2, horizon=2)
    keys = [f"S{i % 2}|{int(x)}" for i, x in enumerate(d)]
    h1 = plan_walk_forward(d, e, keys=keys, n_folds=2).final_test_hash
    h2 = plan_walk_forward(d, e, keys=keys, n_folds=3, embargo_steps=1).final_test_hash
    assert h1 == h2
    keys2 = [*keys[:-1], "other"]
    assert plan_walk_forward(d, e, keys=keys2, n_folds=2).final_test_hash != h1


def test_insufficient_data_rejected() -> None:
    d, e = _samples(20, 1, horizon=1)
    with pytest.raises(InsufficientDataError):
        plan_walk_forward(d, e)
    with pytest.raises(ValueError, match="label_end precedes"):
        plan_walk_forward(np.array([5, 6]), np.array([4, 7]))


@settings(max_examples=60, deadline=None)
@given(
    n_times=st.integers(min_value=80, max_value=260),
    n_symbols=st.integers(min_value=1, max_value=3),
    horizon=st.integers(min_value=1, max_value=12),
    embargo=st.integers(min_value=0, max_value=10),
    n_folds=st.integers(min_value=1, max_value=5),
    jitter_seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_no_train_test_overlap(
    n_times: int, n_symbols: int, horizon: int, embargo: int, n_folds: int, jitter_seed: int
) -> None:
    d, e = _samples(n_times, n_symbols, horizon)
    rng = np.random.default_rng(jitter_seed)
    e = e + rng.integers(0, 3 * 60, size=len(e))  # irregular label lengths (e.g. overnight gaps)
    with contextlib.suppress(InsufficientDataError):
        _check_plan(d, e, {"n_folds": n_folds, "embargo_steps": embargo})
