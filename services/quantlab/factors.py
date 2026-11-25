"""Strategy F — agent-proposed factor strategies through a closed, declarative expression language.

Research agents (Claude Code / Codex / ChatGPT via the MCP tool surface) may submit a *spec*, never code:

    {"expression": "rank(mom(126,21)) - rank(vol(63))", "universe": "liquid1000", "n": 20, "rebalance": 21,
     "regime": "none"}

The expression is parsed with :mod:`ast` and only these nodes are accepted: numeric constants, + - * /, unary -,
and calls to the whitelisted feature/transform functions below with numeric literal arguments in bounded ranges.
No names, attributes, subscripts, comprehensions or keyword arguments exist in the language, so a spec cannot
read files, the network, the protected holdout or anything outside the development panel it is evaluated on.
Every submission is recorded in the trial ledger under family ``F_agent``, so the deflated-Sharpe bar rises with
each attempt, and it must pass exactly the same gates as every other family.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

import numpy as np

from .panel import EquityPanel, trailing_mean
from .strategies import features as F

FAMILY = "F_agent"
MAX_EXPRESSION_CHARS = 300
MAX_NODES = 40


class SpecError(ValueError):
    pass


def _rng(name: str, v: float, lo: float, hi: float) -> int:
    if not (lo <= v <= hi) or int(v) != v:
        raise SpecError(f"{name} argument must be an integer in [{lo}, {hi}]")
    return int(v)


def _build_functions(panel: EquityPanel, mask: np.ndarray) -> dict[str, Callable[..., np.ndarray]]:
    def mom(lb: float, skip: float = 0) -> np.ndarray:
        return F.momentum(panel, _rng("mom", lb, 5, 504), _rng("mom skip", skip, 0, 63))

    def resid_mom(lb: float, skip: float = 21) -> np.ndarray:
        return F.residual_momentum(panel, _rng("resid_mom", lb, 21, 504), _rng("resid skip", skip, 0, 63))

    def vol(w: float) -> np.ndarray:
        return F.realized_vol(panel, _rng("vol", w, 5, 252))

    def rsi(w: float) -> np.ndarray:
        return F.rsi(panel, _rng("rsi", w, 2, 30))

    def relvol(w: float = 20) -> np.ndarray:
        v = panel.f["volume"]
        return v / np.where(trailing_mean(v, _rng("relvol", w, 5, 63), 5) > 0, trailing_mean(v, int(w), 5), np.nan)

    def high52() -> np.ndarray:
        return F.high_52w_ratio(panel)

    def beta(w: float = 252) -> np.ndarray:
        return F.rolling_beta(panel, _rng("beta", w, 63, 504))

    def dist_sma(w: float) -> np.ndarray:
        s = F.sma(panel, _rng("dist_sma", w, 5, 252))
        return panel.f["adj_close"] / s - 1

    def rank(x: np.ndarray) -> np.ndarray:
        return F.cross_rank(x, mask)

    def zscore(x: np.ndarray) -> np.ndarray:
        v = np.where(mask & np.isfinite(x), x, np.nan)
        mu = np.nanmean(v, axis=1, keepdims=True)
        sd = np.nanstd(v, axis=1, keepdims=True)
        return (v - mu) / np.where(sd > 0, sd, np.nan)

    def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(x, float(lo), float(hi))

    return {"mom": mom, "resid_mom": resid_mom, "vol": vol, "rsi": rsi, "relvol": relvol, "high52": high52,
            "beta": beta, "dist_sma": dist_sma, "rank": rank, "zscore": zscore, "clip": clip}


FUNCTION_NAMES = frozenset({"mom", "resid_mom", "vol", "rsi", "relvol", "high52", "beta", "dist_sma", "rank",
                            "zscore", "clip"})
NUMERIC_ARG_FUNCS = frozenset({"mom", "resid_mom", "vol", "rsi", "relvol", "beta", "dist_sma", "high52"})


def validate_expression(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPRESSION_CHARS:
        raise SpecError(f"expression must be 1..{MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise SpecError(f"syntax error: {exc.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise SpecError(f"expression too complex (> {MAX_NODES} nodes)")
    for node in nodes:
        if isinstance(node, ast.Expression | ast.Load | ast.Add | ast.Sub | ast.Mult | ast.Div | ast.USub):
            continue
        if isinstance(node, ast.BinOp | ast.UnaryOp):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTION_NAMES or node.keywords:
                raise SpecError("only whitelisted functions with positional arguments are allowed")
            if node.func.id in NUMERIC_ARG_FUNCS and not all(
                    isinstance(a, ast.Constant) and isinstance(a.value, int | float) for a in node.args):
                raise SpecError(f"{node.func.id}() takes numeric literal arguments only")
            continue
        if isinstance(node, ast.Name) and node.id in FUNCTION_NAMES:
            continue
        raise SpecError(f"disallowed syntax: {type(node).__name__}")
    return tree


def evaluate_expression(expr: str, panel: EquityPanel, mask: np.ndarray) -> np.ndarray:
    tree = validate_expression(expr)
    fns = _build_functions(panel, mask)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            assert isinstance(node.value, int | float)
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            with np.errstate(divide="ignore", invalid="ignore"):
                return a / np.where(np.asarray(b) == 0, np.nan, b)
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            return fns[node.func.id](*[ev(a) for a in node.args])
        raise SpecError(f"unsupported node {type(node).__name__}")

    out = ev(tree)
    if not isinstance(out, np.ndarray) or out.shape != panel.shape:
        raise SpecError("expression must produce a cross-sectional score (use at least one feature)")
    return out


def canonical_expression(tree: ast.Expression) -> str:
    """One spelling per expression (spacing, redundant parentheses and 21 vs 21.0 normalized), so the hypothesis
    memory recognizes a resubmitted factor."""
    class _Num(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            v = node.value
            if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
                return ast.copy_location(ast.Constant(int(v)), node)
            return node

    out = ast.unparse(_Num().visit(ast.parse(ast.unparse(tree), mode="eval")))
    validate_expression(out)  # the canonical spelling must itself be a valid spec (length, node budget)
    return out


def agent_family(dev: EquityPanel, spec: dict[str, Any], actor: str) -> Any:
    """The single-variant development family an agent spec is evaluated as (targets, config, SPY benchmark).
    Lives here so that :func:`services.quantlab.ledger.methodology_hash` covers it."""
    from .backtest import BacktestConfig, apply_rebalance_schedule, hold_schedule
    from .research import FamilySpec
    from .strategies import momentum as A

    cache = A.ScoreCache(dev)

    def targets(p: dict[str, Any]) -> np.ndarray:
        uni = cache.universe(p["universe"])
        sc = evaluate_expression(p["expression"], dev, uni)
        w = A.top_n_weights(sc, uni, int(p["n"])) * cache.regime(p["regime"])[:, None]
        return apply_rebalance_schedule(w, hold_schedule(len(dev.dates), int(p["rebalance"])))

    return FamilySpec(FAMILY, "us_equity", f"agent factor ({actor}): {spec['expression']}", [spec], targets,
                      "SPY buy-and-hold", A.buy_and_hold(dev, "SPY"), BacktestConfig(),
                      execution_notes=["agent-proposed; same gates as every family"])


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    allowed = {"expression", "universe", "n", "rebalance", "regime"}
    extra = set(spec) - allowed
    if extra:
        raise SpecError(f"unknown spec fields: {sorted(extra)}")
    tree = validate_expression(str(spec.get("expression", "")))
    clean = {"expression": canonical_expression(tree),
             "universe": spec.get("universe", "liquid1000"), "n": int(spec.get("n", 20)),
             "rebalance": int(spec.get("rebalance", 21)), "regime": spec.get("regime", "none")}
    if clean["universe"] not in ("liquid1000", "large300"):
        raise SpecError("universe must be liquid1000 or large300")
    if not 5 <= clean["n"] <= 100:
        raise SpecError("n must be in [5, 100]")
    if clean["rebalance"] not in (1, 5, 10, 21, 63):
        raise SpecError("rebalance must be one of 1, 5, 10, 21, 63 sessions")
    if clean["regime"] not in ("none", "sma200", "volscale"):
        raise SpecError("regime must be none, sma200 or volscale")
    return clean
