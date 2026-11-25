"""Agent-factor originality screen on a SIMULATED panel (correlation arithmetic and refusal rules only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from services.quantlab import novelty as Nv
from services.quantlab.factors import validate_spec
from services.quantlab.panel import EquityPanel
from tests.unit.quantlab import synth


@pytest.fixture(scope="module")
def panel_mask() -> tuple[EquityPanel, np.ndarray]:
    rng = np.random.default_rng(2)
    syms = {"SPY": {"drift": 0.0003, "vol": 0.01, "price": 300, "volume": 8e7}}
    for i in range(40):
        syms[f"S{i:02d}"] = {"drift": float(rng.normal(0.0002, 0.0006)), "vol": float(rng.uniform(0.01, 0.03)),
                             "price": 50, "volume": 3e6}
    raw, adj = synth.bars(syms, n=800, start="2016-01-04")
    panel = EquityPanel.build(raw, adj, synth.assets(list(syms)), min_obs=50)
    return panel, panel.universe(min_adv=1e6, min_history=60, kinds=("COMMON_STOCK",))


def test_reference_library_is_valid_dsl() -> None:
    for expr in Nv.REFERENCE_FACTORS.values():
        validate_spec({"expression": expr})
        assert Nv.node_count(expr) <= 40


def test_rescaled_or_negated_existing_signal_is_redundant(panel_mask: tuple[EquityPanel, np.ndarray]) -> None:
    panel, mask = panel_mask
    rep = Nv.novelty_report("2 * mom(252, 21)", panel, mask)
    assert rep["redundant"] and rep["nearest"] == "A:mom_12_1" and rep["nearest_corr"] > 0.99
    neg = Nv.novelty_report("-mom(126, 21)", panel, mask)
    assert neg["redundant"] and neg["nearest_corr"] < -0.99  # the other tail of the same hypothesis
    assert set(rep["correlations"]) == set(Nv.REFERENCE_FACTORS)
    assert rep["correlations"]["G:relvol"] is None  # constant synthetic volume: no cross-sectional variation
    assert rep["compared"] == len(Nv.REFERENCE_FACTORS) - 1
    json.dumps(rep, allow_nan=False)


def test_a_different_signal_passes_the_screen(panel_mask: tuple[EquityPanel, np.ndarray]) -> None:
    panel, mask = panel_mask
    rep = Nv.novelty_report("rank(vol(21)) - rank(vol(126))", panel, mask)
    assert not rep["redundant"] and rep["max_abs_corr"] < Nv.MAX_ABS_CORR
    assert rep["complexity_nodes"] == Nv.node_count("rank(vol(21)) - rank(vol(126))")


def test_prior_agent_factors_count_but_never_the_candidate_itself(panel_mask: tuple[EquityPanel, np.ndarray]) -> None:
    panel, mask = panel_mask
    rows = [{"id": 3, "stage": "DEV", "spec": json.dumps({"expression": "rank(vol(21)) - rank(vol(126))"})},
            {"id": 2, "stage": "WF", "spec": json.dumps({"expression": "mom(30)"})},
            {"id": 1, "stage": "DEV", "spec": "not json"}]
    prior = Nv.prior_agent_expressions(rows, exclude="rank(vol(21)) - rank(vol(126))", canonical=validate_spec)
    assert prior == {}  # own expression excluded, WF rows and unparsable rows ignored
    prior2 = Nv.prior_agent_expressions(rows, exclude="rank(vol(20)) - rank(vol(126))", canonical=validate_spec)
    assert list(prior2.values()) == ["rank(vol(21)) - rank(vol(126))"]
    rep = Nv.novelty_report("rank(vol(20)) - rank(vol(126))", panel, mask, prior2)
    assert rep["redundant"] and rep["nearest"] == "agent:3"
