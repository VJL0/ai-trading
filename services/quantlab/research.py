"""Development-stage family evaluation: every variant backtested after costs and recorded, walk-forward selection,
deflated Sharpe with all family trials, Sharpe-differential SPA vs the passive benchmark, PBO, 2x cost stress,
alpha vs benchmark, and the pre-registered gate verdict. The protected holdout is never touched here.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.research.stats import benjamini_hochberg

from .backtest import BacktestConfig, BacktestResult, performance_metrics, run_weights
from .gates import DEV_GATE, dev_verdict
from .ledger import TrialLedger, code_hash, spec_hash, variant_name
from .panel import EquityPanel
from .validation import alpha_beta, deflated_sharpe, pbo_cscv, spa_test, trial_sharpe_variance, walk_forward_select

NO_EDGE_NOTE = ("Backtested development results on historical data under a cost model. They are not realized "
                "trading results and are not evidence of future profitability.")


@dataclass
class FamilySpec:
    family: str
    asset_class: str
    description: str
    variants: Sequence[dict[str, Any]]
    make_targets: Callable[[dict[str, Any]], np.ndarray]
    benchmark_name: str
    benchmark_targets: np.ndarray
    config: BacktestConfig = field(default_factory=BacktestConfig)
    config_for: Callable[[dict[str, Any]], BacktestConfig] | None = None
    first_test_year: int = 2019
    periods_per_year: float = 252.0
    execution_notes: list[str] = field(default_factory=list)
    wf_period: str = "year"
    wf_min_train_days: int = 504
    runner: Callable[[dict[str, Any], float], BacktestResult] | None = None
    """Custom simulator (e.g. intraday event backtests): (params, cost_multiplier) -> BacktestResult."""
    benchmark_result: BacktestResult | None = None
    policy_runner: Callable[[dict[str, Any], Any, float], BacktestResult] | None = None
    """Execution-tier simulation of a variant under TODAY's risk policy (custom-simulator families)."""

    def run(self, panel: EquityPanel, p: dict[str, Any], cost_multiplier: float = 1.0) -> BacktestResult:
        if self.runner is not None:
            return self.runner(p, cost_multiplier)
        cfg = self.config_for(p) if self.config_for else self.config
        if cost_multiplier != 1.0:
            cfg = replace(cfg, costs=replace(cfg.costs, cost_multiplier=cfg.costs.cost_multiplier * cost_multiplier))
            if cfg.flat_cost_bps is not None:
                cfg = replace(cfg, flat_cost_bps=tuple(x * cost_multiplier for x in cfg.flat_cost_bps))
        return run_weights(panel.f, panel.dates, self.make_targets(p), cfg)


def _variant_name(p: dict[str, Any]) -> str:
    return variant_name(p)


def _vol_unit(r: np.ndarray) -> np.ndarray:
    sd = np.nanstd(r)
    return r / sd if sd > 0 else r * 0


def evaluate_family(spec: FamilySpec, panel: EquityPanel, ledger: TrialLedger, *, fingerprint: str,
                    out_dir: Path | None = None, keep_series: bool = False) -> dict[str, Any]:
    t0 = time.time()
    ppy = spec.periods_per_year
    bench_res = spec.benchmark_result or run_weights(panel.f, panel.dates, spec.benchmark_targets, spec.config)
    bench = pd.Series(bench_res.net, index=bench_res.dates)
    series: dict[str, pd.Series] = {}
    results: dict[str, BacktestResult] = {}
    rows = []
    for p in spec.variants:
        name = _variant_name(p)
        try:
            res = spec.run(panel, p)
        except Exception as exc:  # failures are recorded, never silently dropped
            ledger.record(family=spec.family, variant=name, spec=p, data_fingerprint=fingerprint, stage="DEV",
                          metrics={}, status="ERROR", note=f"{type(exc).__name__}: {exc}"[:500])
            rows.append({"variant": name, "status": "ERROR", "error": str(exc)[:200]})
            continue
        m = res.metrics(ppy)
        ledger.record(family=spec.family, variant=name, spec=p, data_fingerprint=fingerprint, stage="DEV",
                      metrics=m, status="OK")
        series[name] = res.series()
        results[name] = res
        rows.append({"variant": name, "params": p, "status": "OK",
                     **{k: m.get(k) for k in ("sharpe", "cagr", "max_drawdown", "ann_vol", "turnover_annual",
                                              "cost_drag_annual", "trades", "avg_positions", "avg_exposure",
                                              "sortino", "cvar95_daily")}})
    report: dict[str, Any] = {
        "family": spec.family, "asset_class": spec.asset_class, "description": spec.description,
        "generated_at": datetime.now(UTC).isoformat(), "data_fingerprint": fingerprint, "code_hash": code_hash(),
        "dev_period": [str(panel.dates[0])[:10], str(panel.dates[-1])[:10]],
        "benchmark": {"name": spec.benchmark_name, "metrics": bench_res.metrics(ppy)},
        "variants": rows, "n_variants": len(rows), "execution_notes": spec.execution_notes,
        "no_edge_note": NO_EDGE_NOTE, "gate": DEV_GATE,
    }
    if not series:
        report.update({"passed": False, "fail_reasons": ["no variant evaluated"]})
        return report
    frame = pd.DataFrame(series)
    wf = walk_forward_select(series, first_test_year=spec.first_test_year, period=spec.wf_period,
                             min_train_days=spec.wf_min_train_days)
    if wf.oos.empty:
        report.update({"passed": False, "fail_reasons": ["walk-forward produced no out-of-sample period"]})
        return report
    oos = wf.oos.to_numpy()
    wf_metrics = performance_metrics(oos, pd.DatetimeIndex(wf.oos.index), periods_per_year=ppy)
    n_trials = max(len(series), ledger.count(spec.family))
    var_sr = trial_sharpe_variance(ledger.sharpes(spec.family)) or float(np.var(
        [_sr_pp(s.to_numpy()) for s in series.values()], ddof=1))
    dsr = deflated_sharpe(oos, n_trials, var_sr)
    common = frame.index.intersection(bench.index)
    X = np.column_stack([_vol_unit(frame.loc[common, c].to_numpy()) for c in frame.columns])
    b_unit = _vol_unit(bench.loc[common].to_numpy())
    spa: dict[str, Any] = dict(spa_test(X, b_unit))
    spa["best_variant_name"] = str(list(frame.columns)[int(spa["best_variant"])]) if "best_variant" in spa else None
    pbo = pbo_cscv(frame.dropna().to_numpy()) if frame.shape[1] >= 4 else {"pbo": float("nan"), "splits": 0.0}
    oos_bench = bench.reindex(wf.oos.index).to_numpy()
    ab = alpha_beta(oos, oos_bench, ppy)
    selected = wf.choices[-1]["selected"]
    sel_params = next(p for p in spec.variants if _variant_name(p) == selected)
    stress_res = spec.run(panel, sel_params, DEV_GATE["stress_cost_multiplier"])
    stress_wf = stress_res.series().reindex(wf.oos.index).to_numpy()
    report.update({
        "walk_forward": {"metrics": wf_metrics, "choices": wf.choices,
                         "benchmark_same_period": performance_metrics(np.nan_to_num(oos_bench),
                                                                      pd.DatetimeIndex(wf.oos.index),
                                                                      periods_per_year=ppy)},
        "dsr": dsr, "spa_vs_benchmark": spa, "pbo": pbo, "alpha_vs_benchmark": ab,
        "stress": {"cost_multiplier": DEV_GATE["stress_cost_multiplier"],
                   "sharpe": performance_metrics(np.nan_to_num(stress_wf), periods_per_year=ppy).get("sharpe", 0.0),
                   "full_dev": stress_res.metrics(ppy)},
        "selected_variant": selected, "selected_params": sel_params, "selected_spec_hash": spec_hash(sel_params),
        "selected_trades": int(results[selected].trades),
        "selected_full_dev": results[selected].metrics(ppy),
        "elapsed_s": round(time.time() - t0, 1),
    })
    passed, fails = dev_verdict(report)
    report["passed"], report["fail_reasons"] = passed, fails
    ledger.record(family=spec.family, variant="WALK_FORWARD:" + selected, spec={"wf_of": spec.family},
                  data_fingerprint=fingerprint, stage="WF", metrics=wf_metrics,
                  status="PASS" if passed else "FAIL", note="; ".join(fails)[:500])
    if keep_series:
        report["_series"] = {"wf_oos": wf.oos, "benchmark": bench, "variants": frame}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{spec.family}.json").write_text(json.dumps(_public(report), indent=1, default=_json))
        pd.DataFrame({"wf_oos": wf.oos, "benchmark": bench.reindex(wf.oos.index)}).to_csv(
            out_dir / f"{spec.family}_wf_oos.csv")
    return report


def family_fdr(reports: Sequence[dict[str, Any]], q: float = 0.10,
               peers: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """Benjamini-Hochberg across families on p = 1 - DSR (a family passes only if it also survives FDR).
    ``peers`` adds the latest recorded p-values of families NOT evaluated in this run (skipped as unchanged, or
    evaluated in other runs), so running a family alone or skipping its peers never shrinks the correction."""
    p = {r["family"]: 1.0 - float(r.get("dsr", {}).get("dsr", 0.0)) for r in reports if "dsr" in r}
    for fam, pv in (peers or {}).items():
        p.setdefault(fam, float(pv))
    return benjamini_hochberg(p, q=q) if p else {}


class TierNotApplicable(Exception):
    """Raised by a ``policy_runner`` when today's policy or lifecycle cannot execute the variant at all."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


def _sr_pp(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0
    return float(r.mean() / sd) if sd > 0 else 0.0


def _public(report: dict[str, Any]) -> dict[str, Any]:
    return clean_json({k: v for k, v in report.items() if not k.startswith("_")})


def clean_json(o: Any) -> Any:
    """Strict-JSON-safe copy: non-finite floats become None, numpy scalars become Python numbers."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean_json(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def _json(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp | datetime):
        return o.isoformat()
    return str(o)
