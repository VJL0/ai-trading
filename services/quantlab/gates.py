"""Pre-registered validation gates (code constants — changing them is a reviewed code change, never a runtime
setting, and research agents have no write path to them). Standards are never lowered to admit a strategy."""

from __future__ import annotations

from datetime import date
from math import isfinite
from numbers import Real
from typing import Any, Final

HOLDOUT_NAME: Final = "quantlab-holdout-v1"
HOLDOUT_START: Final = date(2025, 11, 21)
"""Protected holdout: the final 12 months of history available when quantlab was built (2025-11-21..2026-11-20).
Development panels end strictly before this date. The vault stores the interval permanently on first use."""
HOLDOUT_END: Final = date(2026, 11, 20)

DEV_GATE: Final[dict[str, float]] = {
    "dsr_min": 0.95,  # deflated Sharpe probability on the walk-forward OOS series, all family trials counted
    "wf_oos_sharpe_min": 0.5,  # annualized, after costs
    "benchmark_spa_p_max": 0.10,  # Sharpe-differential SPA vs the family's passive benchmark ...
    "alpha_t_min": 2.0,  # ... OR Newey-West alpha t-stat vs the benchmark
    "pbo_max": 0.5,
    "stress_cost_multiplier": 2.0,
    "stress_sharpe_min": 0.0,
    "max_drawdown_floor": -0.35,
    "min_trades": 50,
}

HOLDOUT_GATE: Final[dict[str, float]] = {
    "net_return_min": 0.0,
    "sharpe_min": 0.0,
    "drawdown_multiple_of_dev": 1.5,  # holdout max drawdown may not exceed 1.5x the development one ...
    "max_drawdown_floor": -0.35,  # ... nor this absolute floor
}

SHADOW_GATE: Final[dict[str, float]] = {
    "min_sessions": 20,  # prospective shadow sessions before a PAPER promotion can be requested
    "max_tracking_gap_bps": 50.0,  # mean |shadow realized - backtest-replayed| per decision
}


def _metric_errors(values: dict[str, Any], *, probabilities: tuple[str, ...] = (),
                   drawdowns: tuple[str, ...] = ()) -> list[str]:
    """Missing/non-finite evidence never passes a numerical comparison (NaN compares false both ways)."""
    fails = []
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            fails.append(f"{name}: finite numeric evidence required")
            continue
        number = float(value)
        if name in probabilities and not 0 <= number <= 1:
            fails.append(f"{name}: probability outside [0, 1]")
        elif name in drawdowns and not -1 <= number <= 0:
            fails.append(f"{name}: drawdown outside [-1, 0]")
        elif name == "selected_trades" and (number < 0 or int(number) != number):
            fails.append("selected_trades: non-negative integer required")
    return fails


def _get(report: dict[str, Any], *path: str) -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def dev_verdict(report: dict[str, Any]) -> tuple[bool, list[str]]:
    g = DEV_GATE
    invalid = _metric_errors({
        "dsr": _get(report, "dsr", "dsr"),
        "sharpe": _get(report, "walk_forward", "metrics", "sharpe"),
        "spa_p": _get(report, "spa_vs_benchmark", "p_value"),
        "alpha_t": _get(report, "alpha_vs_benchmark", "alpha_t"),
        "pbo": _get(report, "pbo", "pbo"),
        "stress_sharpe": _get(report, "stress", "sharpe"),
        "max_drawdown": _get(report, "walk_forward", "metrics", "max_drawdown"),
        "selected_trades": report.get("selected_trades"),
    }, probabilities=("dsr", "spa_p", "pbo"), drawdowns=("max_drawdown",))
    if invalid:
        return False, invalid
    fails: list[str] = []
    wf = report.get("walk_forward", {}).get("metrics", {})
    if report.get("dsr", {}).get("dsr", 0.0) < g["dsr_min"]:
        fails.append(f"DSR {report.get('dsr', {}).get('dsr', 0):.3f} < {g['dsr_min']}")
    if wf.get("sharpe", 0.0) < g["wf_oos_sharpe_min"]:
        fails.append(f"walk-forward OOS Sharpe {wf.get('sharpe', 0):.2f} < {g['wf_oos_sharpe_min']}")
    spa_p = report.get("spa_vs_benchmark", {}).get("p_value", 1.0)
    alpha_t = report.get("alpha_vs_benchmark", {}).get("alpha_t", 0.0)
    if spa_p > g["benchmark_spa_p_max"] and alpha_t < g["alpha_t_min"]:
        fails.append(f"no edge vs passive benchmark (Sharpe-diff SPA p={spa_p:.2f}, alpha t={alpha_t:.2f})")
    pbo = report.get("pbo", {}).get("pbo")
    if pbo > g["pbo_max"]:
        fails.append(f"PBO {pbo:.2f} > {g['pbo_max']}")
    if report.get("stress", {}).get("sharpe", 0.0) <= g["stress_sharpe_min"]:
        fails.append("Sharpe not positive at 2x costs")
    if wf.get("max_drawdown", -1.0) < g["max_drawdown_floor"]:
        fails.append(f"walk-forward max drawdown {wf.get('max_drawdown', 0):.1%} below {g['max_drawdown_floor']:.0%}")
    if report.get("selected_trades", 0) < g["min_trades"]:
        fails.append(f"fewer than {int(g['min_trades'])} trades")
    return (not fails), fails


def holdout_verdict(holdout: dict[str, Any], dev_mdd: float) -> tuple[bool, list[str]]:
    g = HOLDOUT_GATE
    invalid = _metric_errors({
        "total_return": holdout.get("total_return"), "sharpe": holdout.get("sharpe"),
        "max_drawdown": holdout.get("max_drawdown"), "dev_mdd": dev_mdd,
    }, drawdowns=("max_drawdown", "dev_mdd"))
    if invalid:
        return False, invalid
    fails = []
    if holdout.get("total_return", -1.0) <= g["net_return_min"]:
        fails.append(f"holdout net return {holdout.get('total_return', 0):.2%} <= 0")
    if holdout.get("sharpe", -1.0) <= g["sharpe_min"]:
        fails.append(f"holdout Sharpe {holdout.get('sharpe', 0):.2f} <= 0")
    floor = max(g["max_drawdown_floor"], g["drawdown_multiple_of_dev"] * min(dev_mdd, -0.01))
    if holdout.get("max_drawdown", -1.0) < floor:
        fails.append(f"holdout drawdown {holdout.get('max_drawdown', 0):.1%} below {floor:.1%}")
    return (not fails), fails
