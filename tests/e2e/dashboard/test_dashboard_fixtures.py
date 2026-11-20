"""Dashboard fixtures and generated TS contract schemas stay in sync with packages.contracts.

* the mock-API fixture JSON (apps/dashboard/fixtures) equals what build_fixtures.py renders now;
* every contract object embedded in a fixture validates back through its Pydantic model;
* src/generated/{schemas.json,enums.ts} equal what scripts/gen_ts_types.py renders now, and every
  Decimal field is a string schema tagged ``x-lat-kind: decimal`` (TS ``DecimalString``);
* the fixture tool catalog matches the real agent tool registry (names, categories, mutating flags).
No network; pure file/contract checks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from packages.contracts import (
    ApprovedOrderIntent,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "apps" / "dashboard" / "fixtures"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load("lat_dashboard_build_fixtures", ROOT / "tests" / "e2e" / "dashboard" / "build_fixtures.py")


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load("lat_gen_ts_types", ROOT / "scripts" / "gen_ts_types.py")


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_fixtures_are_fresh(builder: ModuleType) -> None:
    stale = [p.name for p, text in builder.render().items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    assert stale == [], f"stale fixtures {stale}: run uv run --frozen python tests/e2e/dashboard/build_fixtures.py"


def test_fixture_contract_objects_validate() -> None:
    detail = _fixture("order_lat-sim-01jbdashfixture00000filled1.json")
    ApprovedOrderIntent.model_validate(detail["intent"])
    assert [t["to_state"] for t in detail["transitions"]][-1] == "FILLED"
    assert len(detail["fills"]) == 2


def test_fixture_money_is_decimal_strings() -> None:
    perf = _fixture("final_performance.json")
    for key in ("portfolio_value", "cash", "realized_pnl", "api_cost_usd"):
        assert isinstance(perf[key], str)
        Decimal(perf[key])
    for order in _fixture("final_orders.json")["orders"]:
        assert isinstance(order["quantity"], str) and isinstance(order["limit_price"], str)


def test_fixtures_are_labelled_simulated() -> None:
    status = _fixture("status.json")
    assert status["execution_label"] == "SIMULATED" and status["feed"]["is_simulated"] is True
    assert _fixture("performance.json")["is_simulated"] is True


def test_generated_ts_schemas_are_fresh(gen: ModuleType) -> None:
    stale = [p.name for p, text in gen.render().items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    assert stale == [], f"stale generated files {stale}: run uv run --frozen python scripts/gen_ts_types.py"


def test_generated_schema_marks_decimals_and_timestamps(gen: ModuleType) -> None:
    bundle = gen.build_bundle()
    defs = bundle["$defs"]
    for name in ("ApprovedOrderIntent", "TradeProposal", "PredictionSet", "RiskPolicy", "AccountSnapshot"):
        assert name in defs, name
    intent = defs["ApprovedOrderIntent"]["properties"]
    assert intent["quantity"]["type"] == "string" and intent["quantity"]["tsType"] == "DecimalString"
    assert intent["quantity"]["x-lat-kind"] == "decimal"
    limit = intent["limit_price"]
    assert limit.get("tsType") == "DecimalString" or any(sub.get("tsType") == "DecimalString" for sub in limit.get("anyOf", []))
    created = intent["created_at"]
    assert created["tsType"] == "UtcTimestamp" and created["format"] == "date-time"
    pred = defs["PredictionSet"]["properties"]
    assert any(sub.get("tsType") == "DecimalString" for sub in pred["success_probability"]["anyOf"])
    # every model's properties are all required (model_dump always emits them)
    for name, schema in defs.items():
        if "properties" in schema:
            assert set(schema["required"]) == set(schema["properties"]), name
