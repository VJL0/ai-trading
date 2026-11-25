"""Originality screen for agent-proposed factors (AlphaAgent-style, KDD 2025: originality, complexity bounds).

An agent factor is only worth a trial if it is not an existing signal in disguise. Before an agent spec is
evaluated, its cross-sectional score is compared with

* a **reference library** — the signals the built-in families already search (momentum horizons, residual
  momentum, short-term reversal, low volatility, RSI, 52-week high, beta, trend distance, relative volume, the
  family-A blend), written in the same declarative language, and
* the most recent distinct **earlier agent factors** recorded in the trial ledger,

by the mean per-date Spearman rank correlation within the spec's universe on development data (one date every
21 sessions). A candidate whose ``|correlation|`` with any of them reaches :data:`MAX_ABS_CORR` is refused as
``REDUNDANT_HYPOTHESIS`` (not recorded, so it does not consume the family's deflated-Sharpe budget); the sign is
ignored because a top-N portfolio of ``-x`` tests the other tail of the same hypothesis. The report (nearest
signal, all correlations, node count) is returned with every accepted experiment.

The screen never loosens a gate: an original factor still faces the same development gate as every family.
"""

from __future__ import annotations

import ast
from typing import Any, Final

import numpy as np

from .factors import evaluate_expression, validate_expression
from .panel import EquityPanel
from .ranking import rank_ic

MAX_ABS_CORR: Final = 0.9
SAMPLE_EVERY: Final = 21
WARMUP_ROWS: Final = 252
MAX_PRIOR: Final = 25

REFERENCE_FACTORS: Final[dict[str, str]] = {
    "A:mom_12_1": "mom(252, 21)",
    "A:mom_6_1": "mom(126, 21)",
    "A:mom_3": "mom(63)",
    "mom_1": "mom(21)",
    "A:resid_12_1": "resid_mom(252, 21)",
    "A:blend": "rank(mom(21)) + rank(mom(63)) + rank(mom(126)) + rank(mom(252))",
    "C1:reversal_5": "-mom(5)",
    "C2:rsi2": "-rsi(2)",
    "rsi14": "rsi(14)",
    "low_vol_63": "-vol(63)",
    "G:high52": "high52()",
    "G:low_beta": "-beta(252)",
    "G:dist200": "dist_sma(200)",
    "G:dist50": "dist_sma(50)",
    "G:relvol": "relvol(20)",
}
"""Signals already searched by built-in families (A momentum, C reversal/pullback, G ML features)."""


def node_count(expression: str) -> int:
    return len(list(ast.walk(validate_expression(expression))))


def _reduced(panel: EquityPanel, mask: np.ndarray) -> tuple[EquityPanel, np.ndarray]:
    """Columns ever eligible (plus SPY, which beta/residual momentum need): identical per-name values, less work."""
    cols = set(np.where(mask.any(axis=0))[0].tolist())
    if "SPY" in set(panel.symbols.tolist()):
        cols.add(panel.col("SPY"))
    idx = np.array(sorted(cols), dtype=int)
    sub = EquityPanel(panel.dates, panel.symbols[idx], {k: v[:, idx] for k, v in panel.f.items()}, panel.kind[idx],
                      dict(panel.meta))
    return sub, mask[:, idx]


def novelty_report(expression: str, panel: EquityPanel, mask: np.ndarray,
                   prior: dict[str, str] | None = None) -> dict[str, Any]:
    """Mean rank correlation of ``expression`` with every reference and prior factor (development data only)."""
    sub, m = _reduced(panel, mask)
    rows = np.arange(WARMUP_ROWS, len(sub.dates), SAMPLE_EVERY)
    cand = evaluate_expression(expression, sub, m)
    others = {**REFERENCE_FACTORS, **{f"agent:{k}": v for k, v in (prior or {}).items()}}
    corrs: dict[str, float | None] = {}
    for name, expr in others.items():
        try:
            ref = evaluate_expression(expr, sub, m)
        except Exception:  # a malformed legacy prior expression is skipped, never fatal
            corrs[name] = None
            continue
        ic = rank_ic(cand, ref, m, rows)
        corrs[name] = float(np.nanmean(ic)) if np.isfinite(ic).any() else None
    finite = {k: v for k, v in corrs.items() if v is not None}
    nearest = max(finite, key=lambda k: abs(finite[k])) if finite else None
    max_abs = abs(finite[nearest]) if nearest else 0.0
    return {"max_abs_corr": round(max_abs, 4), "nearest": nearest,
            "nearest_corr": round(finite[nearest], 4) if nearest else None,
            "redundant": bool(max_abs >= MAX_ABS_CORR), "threshold": MAX_ABS_CORR,
            "complexity_nodes": node_count(expression), "compared": len(finite), "sample_dates": len(rows),
            "correlations": {k: (round(v, 4) if v is not None else None) for k, v in corrs.items()}}


def prior_agent_expressions(rows: list[dict[str, Any]], exclude: str, canonical: Any) -> dict[str, str]:
    """Distinct canonical expressions of earlier agent trials (newest first, at most :data:`MAX_PRIOR`), without
    the candidate's own (a re-test on new evidence is not redundant with itself)."""
    import json

    out: dict[str, str] = {}
    for r in rows:
        if r.get("stage") != "DEV":
            continue
        try:
            expr = canonical(json.loads(r.get("spec") or "{}"))["expression"]
        except Exception:  # noqa: S112 - unparsable legacy rows are not comparable
            continue
        if expr != exclude and expr not in out.values():
            out[str(r.get("id", len(out)))] = expr
        if len(out) >= MAX_PRIOR:
            break
    return out
