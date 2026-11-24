from __future__ import annotations

from typing import Any

import pytest

from services.quantlab.gates import dev_verdict, holdout_verdict


def valid_report() -> dict[str, Any]:
    return {"dsr": {"dsr": 0.97}, "walk_forward": {"metrics": {"sharpe": 1.1, "max_drawdown": -0.1}},
            "spa_vs_benchmark": {"p_value": 0.02}, "alpha_vs_benchmark": {"alpha_t": 2.1},
            "pbo": {"pbo": 0.2}, "stress": {"sharpe": 0.8}, "selected_trades": 80}


@pytest.mark.parametrize("path", [
    ("dsr", "dsr"), ("walk_forward", "metrics", "sharpe"), ("walk_forward", "metrics", "max_drawdown"),
    ("spa_vs_benchmark", "p_value"), ("alpha_vs_benchmark", "alpha_t"), ("pbo", "pbo"),
    ("stress", "sharpe"), ("selected_trades",),
])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.99", True])
def test_invalid_development_evidence_fails_closed(path: tuple[str, ...], bad: Any) -> None:
    report = valid_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    passed, failures = dev_verdict(report)
    assert not passed and failures


@pytest.mark.parametrize("field", ["total_return", "sharpe", "max_drawdown", "dev_mdd"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "1", True])
def test_invalid_holdout_evidence_fails_closed(field: str, bad: Any) -> None:
    metrics = {"total_return": 0.1, "sharpe": 1.1, "max_drawdown": -0.12, "dev_mdd": -0.1}
    metrics[field] = bad
    passed, failures = holdout_verdict(metrics, metrics.pop("dev_mdd"))
    assert not passed and failures


def test_valid_evidence_keeps_existing_gate_thresholds() -> None:
    assert dev_verdict(valid_report()) == (True, [])
    assert holdout_verdict({"total_return": 0.1, "sharpe": 1.0, "max_drawdown": -0.12}, -0.1) == (True, [])
    assert not dev_verdict({})[0]
    assert not holdout_verdict({}, -0.1)[0]


@pytest.mark.parametrize(("section", "field", "bad"), [
    ("dsr", "dsr", 1.1), ("spa_vs_benchmark", "p_value", -0.1), ("pbo", "pbo", -0.1),
])
def test_impossible_probability_fails_closed(section: str, field: str, bad: float) -> None:
    report = valid_report()
    report[section][field] = bad
    assert not dev_verdict(report)[0]
