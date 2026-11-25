"""Scripted providers for tests and the ``--fake`` CLI demo. No network, no CLI, no quota."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import mechanisms
from .contracts import Invocation, McpServerSpec, Provider, RunOutcome, RunResult, TaskKind
from .schemas import kind_of_schema

ScriptItem = RunResult | Callable[[Invocation], RunResult]


class FakeProvider:
    """Returns the scripted results in order (a callable receives the invocation); then ``default`` or ERROR."""

    def __init__(self, name: Provider, script: Sequence[ScriptItem] = (), *,
                 default: ScriptItem | None = None, available: tuple[bool, str] = (True, "fake provider")) -> None:
        self._name = Provider(name)
        self.script: list[ScriptItem] = list(script)
        self.default = default
        self.available = available
        self.calls: list[Invocation] = []

    @property
    def name(self) -> Provider:
        return self._name

    def availability(self) -> tuple[bool, str]:
        return self.available

    def run(self, invocation: Invocation) -> RunResult:
        self.calls.append(invocation)
        item = self.script.pop(0) if self.script else self.default
        if item is None:
            return RunResult(RunOutcome.ERROR, detail="fake script exhausted")
        return item(invocation) if callable(item) else item


def fake_mcp_spec(root: Path, agent_name: str) -> McpServerSpec:
    return McpServerSpec(name="lat-research-fake", command="true", args=(), cwd=root)


def demo_output(kind: TaskKind, invocation: Invocation) -> dict[str, Any]:
    """A schema-valid answer per task kind (demo only: clearly labelled as simulated)."""
    if kind is TaskKind.HYPOTHESIS:
        mech = next((m for m in mechanisms.EXPRESSIBLE if f"Assigned mechanism: {m} " in invocation.prompt),
                    "momentum")
        exprs = {"momentum": "rank(mom(252,21)) - rank(vol(63))", "reversal": "-rank(mom(5))",
                 "attention": "rank(relvol(20)) + rank(mom(126,21))", "volatility_risk": "-rank(beta(252))"}
        return {"mechanism": mech, "economic_rationale": "[fake demo] simulated rationale",
                "why_novel": "[fake demo] simulated novelty statement",
                "falsification": "[fake demo] refuted if the walk-forward Sharpe does not beat SPY after costs",
                "specs": [{"expression": exprs.get(mech, "rank(mom(126,21))"), "universe": "liquid1000", "n": 20,
                           "rebalance": 21, "regime": "none", "intent": "[fake demo]"}],
                "data_requests": []}
    if kind is TaskKind.REVIEW:
        return {"verdict": "INCONCLUSIVE", "concerns": ["[fake demo] simulated review"], "leakage_or_bias_risks": [],
                "mechanism_plausibility": "[fake demo]", "suggested_tests": []}
    if kind is TaskKind.MARKET_ANALYSIS:
        return {"regime_summary": "[fake demo] simulated", "observations": [], "hypothesis_seeds": [],
                "data_gaps": []}
    return {"proposals": []}


def demo_provider(name: Provider) -> FakeProvider:
    def answer(inv: Invocation) -> RunResult:
        kind = kind_of_schema(dict(inv.output_schema))
        if kind is None:
            return RunResult(RunOutcome.INVALID_OUTPUT, detail="unknown schema")
        return RunResult(RunOutcome.OK, output=demo_output(kind, inv), usage={"simulated": True})

    return FakeProvider(name, default=answer, available=(True, "fake demo provider (simulated)"))


def demo_experiment_runner(root: Path, spec: dict[str, Any], actor: str) -> dict[str, Any]:
    """Validates like the real runner but never evaluates or writes the trial ledger."""
    from services.quantlab.factors import SpecError, validate_spec

    try:
        clean = validate_spec(spec)
    except SpecError as exc:
        return {"accepted": False, "error": str(exc)}
    return {"accepted": True, "status": "DONE", "spec": clean,
            "result": {"passed_dev_gate": False, "fail_reasons": ["SIMULATED: fake demo, nothing evaluated"],
                       "execution_label": "SIMULATED", "is_simulated": True, "actor": actor}}
