"""Read-only research view for family K: the diagnostics digest rides on the family detail only (files only)."""

from __future__ import annotations

import json

from services.quantlab import views


def test_family_detail_carries_a_compact_diagnostics_digest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "experiments" / "quantlab"
    (out / "runs" / "ql_x").mkdir(parents=True)
    fam = {"family": "K_xs_rank", "run_id": "ql_x", "passed_dev_gate": False, "dsr": 0.4}
    (out / "latest_research.json").write_text(json.dumps({"families": {"K_xs_rank": fam}, "updated_at": "now"}))
    (out / "runs" / "ql_x" / "K_xs_rank.json").write_text(json.dumps({"dsr": {"dsr": 0.4}, "variants": [1, 2]}))
    diag = {"run_id": "ql_x", "per_model_horizon": {"ridge|h5": {
        "rank_ic": {"mean": 0.01, "t": 1.2, "pct_positive_years": 0.6, "yearly": {"2020": 0.02}},
        "decile_excess_bps": {"top_minus_bottom_mean": 12.0, "top_mean": 6.0, "profile": [1] * 10},
        "top_decile_median_round_trip_cost_bps": 9.0, "top_decile_minus_cost_bps": -3.0,
        "top20_turnover_per_rebalance": 0.4, "calibration_by_year": {"2020": {}}}},
        "portfolios": {"ridge|5|top_n": {"oos": {"sharpe": 0.5, "cagr": 0.08}, "full_dev": {"turnover_annual": 9.0}}},
        "paired_ic_tests": {}, "benchmarks_oos": {"cash": {"sharpe": 0.0}}, "fold_log": [{"big": 1}] * 50}
    (out / "xs_rank_diagnostics.json").write_text(json.dumps(diag))
    res = views.research(tmp_path, "K_xs_rank")
    assert res["status"] == "OK" and "variants" not in res["detail"]
    d = res["diagnostics"]
    assert d["per_model_horizon"]["ridge|h5"] == {
        "ic_mean": 0.01, "ic_t": 1.2, "ic_pct_positive_years": 0.6, "top_minus_bottom_bps": 12.0,
        "top_decile_bps": 6.0, "top_decile_round_trip_cost_bps": 9.0, "top_decile_minus_cost_bps": -3.0,
        "top20_turnover": 0.4}
    assert d["portfolios"]["ridge|5|top_n"]["oos_sharpe"] == 0.5
    assert "fold_log" not in d  # bulky internals stay in the file
    assert "diagnostics" not in views.research(tmp_path)  # summary listing unchanged
