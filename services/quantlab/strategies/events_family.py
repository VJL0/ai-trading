"""L_events: a pre-registered EDGAR feature ablation on family K's purged folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from services.market_data.edgar.store import EdgarStore, store_path

from ..backtest import BacktestConfig, performance_metrics
from ..costs import EquityCosts
from ..events import (
    EARNINGS_FIELDS,
    EVENT_FIELDS,
    FUNDAMENTAL_FIELDS,
    INSIDER_FIELDS,
    attach_events,
    evidence_fingerprint,
    missing_fields,
)
from ..gates import HOLDOUT_START
from ..panel import EquityPanel
from ..ranking import pct_rank
from ..research import FamilySpec, clean_json
from . import xs_rank as K
from .momentum import buy_and_hold

if TYPE_CHECKING:
    from ..pipeline import Lab

FAMILY = "L_events"
ARMS = {
    "price_only": (),
    "price_fundamentals": FUNDAMENTAL_FIELDS,
    "price_insider": INSIDER_FIELDS,
    "all_features": EVENT_FIELDS,
}
_PORTFOLIO = {"model": "ridge", "horizon": 5, "construction": "cost_aware"}


class _EventRank(K.XsRankLab):
    def __init__(self, panel: EquityPanel, fields: tuple[str, ...], *, costs: EquityCosts) -> None:
        super().__init__(panel, costs=costs)
        self.event_fields = fields

    def prepare(self) -> None:
        if self._ready:  # pyright: ignore[reportPrivateUsage]
            return
        super().prepare()
        if self.event_fields:
            values = []
            for name in self.event_fields:
                ranked = pct_rank(self.sub.f[name], self.mask)
                # Center finite cross-section scores and give missing disclosures a neutral score.
                counts = np.isfinite(ranked).sum(axis=1, keepdims=True)
                center = 0.5 + 0.5 / np.maximum(1, counts)
                values.append(np.nan_to_num(ranked - center).astype(np.float32))
            self.X = np.concatenate((self.X, np.stack(values, axis=-1)), axis=-1)
            self.names.extend(self.event_fields)


def events_family(dev: EquityPanel) -> FamilySpec:
    if not len(dev.dates) or dev.dates.date[-1] >= HOLDOUT_START:
        raise ValueError("L_events requires a nonempty development-only panel")
    missing = set(missing_fields(dev, EVENT_FIELDS))
    for name, fields in (("earnings", EARNINGS_FIELDS), ("fundamentals", FUNDAMENTAL_FIELDS), ("insider", INSIDER_FIELDS)):
        if missing.issuperset(fields):
            raise ValueError(f"L_events needs SEC EDGAR {name} coverage on the development panel")
    costs = EquityCosts()
    cached: dict[str, np.ndarray] = {}

    def targets(params: dict[str, Any]) -> np.ndarray:
        arm = str(params["features"])
        if arm not in cached:
            cached[arm] = _EventRank(dev, ARMS[arm], costs=costs).targets(params)
        return cached[arm]

    return FamilySpec(
        FAMILY, "us_equity", "SEC filing feature ablation against the same price-only ridge model",
        [{**_PORTFOLIO, "features": name} for name in ARMS], targets, "SPY buy-and-hold", buy_and_hold(dev, "SPY"),
        BacktestConfig(capital=K.CAPITAL, execution="open", costs=costs), first_test_year=K.FIRST_TEST_YEAR,
        execution_notes=["purged yearly predictions and cost-aware 5-session portfolios as in K_xs_rank",
                         "positions hold overnight and require execution-policy authorization"],
    )


def compare_after_costs(spec: FamilySpec, dev: EquityPanel) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if not (np.asarray(dev.dates.year)[1:] >= spec.first_test_year).any():
        return {"status": "NO_OUT_OF_SAMPLE_SESSIONS", "family": FAMILY}
    for variant in spec.variants:
        result = spec.run(dev, variant)
        coverage = np.asarray(result.dates.year) >= spec.first_test_year
        metrics = performance_metrics(result.net[coverage], result.dates[coverage], cost=result.cost[coverage],
                                      turnover=result.turnover[coverage])
        rows[str(variant["features"])] = {key: metrics.get(key) for key in (
            "days", "start", "end", "cagr", "sharpe", "max_drawdown", "cost_drag_annual", "turnover_annual",
        )}
    base = rows["price_only"]
    for row in rows.values():
        for key in ("cagr", "sharpe", "cost_drag_annual"):
            row[f"delta_{key}_vs_price_only"] = (row[key] - base[key]) if row[key] is not None else None
    return clean_json({
        "status": "OK", "family": FAMILY, "source": "SEC EDGAR", "classification": "development_backtest",
        "arms": rows, "missing_development_fields": missing_fields(dev, EVENT_FIELDS),
    })


def run_events_research(lab: Lab, dev: EquityPanel, store: EdgarStore, *, force: bool = False,
                        run_id: str | None = None) -> dict[str, Any]:
    if not len(dev.dates) or dev.dates.date[-1] >= HOLDOUT_START:
        raise ValueError("L_events requires a nonempty development-only panel")
    dev = attach_events(dev, store)
    spec = events_family(dev)
    summary = lab.run_research([spec], dev, f"edgar:{evidence_fingerprint(dev)}", force=force, run_id=run_id)
    if FAMILY not in summary["skipped_no_new_evidence"]:
        comparison = compare_after_costs(spec, dev)
        summary["events_comparison"] = comparison
        report = lab.out / "runs" / str(summary["run_id"]) / "events_comparison.json"
        report.write_text(json.dumps(comparison, indent=1, allow_nan=False))
    return summary


def run_from_project(root: Path, *, force: bool = False) -> dict[str, Any]:
    """Entry for the parent research CLI; the protected panel rows are sliced before the EDGAR join."""
    from ..pipeline import Lab

    lab = Lab(root)
    panel, _ = lab.load_panel("equity_")
    return run_events_research(lab, lab.dev_slice(panel), EdgarStore(store_path(root)), force=force)
