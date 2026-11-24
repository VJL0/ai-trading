"""Phase 4: every NO_TRADE cycle is attributed to predictions, costs, uncertainty, threshold, data, locks or risk."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from services.research.opportunity import CAUSES, attribute_no_trades, classify_signal


def sig(gross: str, cost: str = "6", unc: str = "0", reason: str = "not candidate") -> dict[str, Any]:
    return {"expected_return_bps": gross, "est_spread_bps": cost, "est_fees_bps": "0", "est_slippage_bps": "0",
            "uncertainty_bps": unc, "reason": reason}


def test_classification_follows_the_decision_contract() -> None:
    assert classify_signal(sig("-1"), threshold_bps=5) == "PREDICTION"
    assert classify_signal(sig("8", cost="6"), threshold_bps=5) == "COSTS"  # 8 > 5 but 8 - 6 = 2
    assert classify_signal(sig("14", cost="6", unc="4"), threshold_bps=5) == "UNCERTAINTY"  # net 8 > 5, 8 - 4 = 4
    assert classify_signal(sig("4", cost="1"), threshold_bps=5) == "THRESHOLD"
    assert classify_signal(sig("40", reason="not candidate: stale data: last bar ended"), threshold_bps=5) == "DATA"
    assert classify_signal(sig("0", reason="model produced no prediction -> not eligible"), threshold_bps=5) == "NO_FORECAST"


def test_runtime_cycles_are_all_attributed(tmp_path: Path) -> None:
    from services.api.runtime import Runtime
    from tests.e2e.api.test_e2e_pipeline import _settings, _startup

    rt = Runtime(_settings(tmp_path))
    try:
        asyncio.run(_startup(rt))
        for _ in range(3):
            asyncio.run(rt.run_cycle(symbols=None, use_ai=False, actor="test"))
            rt.sim_clock.step(120)
        report = attribute_no_trades(rt.db, threshold_bps=float(rt.signal_engine.min_net_edge_bps))
        assert report["cycles"] == 3 and report["no_trade"] + report["submitted"] == 3
        assert sum(report["causes"].values()) == report["no_trade"] and set(report["causes"]) == set(CAUSES)
        assert "never lowers" in report["note"]
    finally:
        asyncio.run(rt.aclose())
