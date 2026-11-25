"""Trading runs independently of the research crew, and the crew has no execution authority:

* importing the crew (loop, CLI, fake providers, and the quantlab modules it calls lazily) never loads the API,
  order ledger, risk, positions, sessions or a broker SDK;
* nothing in the trading runtime imports ``services.agents``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FORBIDDEN = ("services.api", "services.order_ledger", "services.risk", "services.positions", "services.sessions")
TRADING_PACKAGES = ("api", "risk", "order_ledger", "positions", "sessions")

_PROBE = """
import json, sys
import services.agents.crew, services.agents.__main__, services.agents.fake, services.agents.planner
import services.agents.context, services.agents.store, services.agents.schemas
import services.quantlab.factors, services.quantlab.views  # what the crew imports lazily at run time
print(json.dumps(sorted(sys.modules)))
"""


def test_importing_crew_does_not_load_trading_runtime() -> None:
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, capture_output=True, text=True, timeout=120,
                          check=False, env={"PATH": "/usr/bin:/bin", "LAT_USE_KEYCHAIN": "false",
                                            "LAT_FEED_MODE": "SIMULATED", "PYTHONPATH": str(REPO)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    mods = json.loads(proc.stdout.strip().splitlines()[-1])
    loaded = [m for m in mods if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    assert loaded == []
    assert not [m for m in mods if m == "alpaca" or m.startswith("alpaca.")]
    assert not [m for m in mods if m.startswith("services.market_data.") and m != "services.market_data.calendar"]


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in ("__import__", "import_module"):  # dynamic imports count too
                out.add(str(node.args[0].value))
    return out


def test_trading_runtime_never_imports_agents() -> None:
    offenders: list[str] = []
    scanned = 0
    for pkg in TRADING_PACKAGES:
        for path in (REPO / "services" / pkg).rglob("*.py"):
            scanned += 1
            for mod in _imports(path):
                if mod == "services.agents" or mod.startswith("services.agents."):  # incl. `from services import agents`
                    offenders.append(f"{path.relative_to(REPO)}: {mod}")
    assert scanned > 10
    assert offenders == []
