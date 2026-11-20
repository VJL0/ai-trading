"""LEAN run helpers for the LAT trading component (used by scripts/lean_*.sh and tests/*/lean).

* Per-run LEAN ``config.json`` generation (never edits the upstream ``Launcher/config.json``).
* Backtest result parsing into ``experiments/lean/<run_id>/summary.json``.
* PAPER (Alpaca) config generation with secrets read through ``packages.core.secrets.SecretStore``.

Every config is FLAT (no ``environments`` section) so what LEAN runs is exactly what is written here.
CLI: ``uv run --frozen python trading/lean/tools/lat_lean.py <command> ...`` (see ``main``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LEAN_DIR = ROOT / "trading" / "lean" / "Lean"
LAUNCHER_DIR = LEAN_DIR / "Launcher" / "bin" / "Release"
LAUNCHER_DLL = LAUNCHER_DIR / "QuantConnect.Lean.Launcher.dll"
BUNDLE_DIR = ROOT / "trading" / "lean" / "bin" / "plugins"
LAT_DLL = BUNDLE_DIR / "LatStrategies.dll"
TEST_ALGOS_DLL = ROOT / "trading" / "lean" / "LatTestAlgorithms" / "bin" / "Release" / "LatTestAlgorithms.dll"
RUNS_DIR = ROOT / "trading" / "lean" / "runs"
EXPERIMENTS_DIR = ROOT / "experiments" / "lean"
LEAN_SAMPLE_DATA = LEAN_DIR / "Data"
DATA_LOCAL = ROOT / "trading" / "lean" / "data-local"
PINS = ROOT / "trading" / "PINS.json"
PAPER_TEMPLATE = ROOT / "trading" / "lean" / "config" / "live-alpaca-paper.template.json"

_RUN_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")

# Engine plumbing identical to the upstream Launcher defaults (Launcher/config.json at the pinned commit).
_BASE: dict[str, Any] = {
    "algorithm-language": "CSharp",
    "debugging": False,
    "debugging-method": "LocalCmdline",
    "log-handler": "QuantConnect.Logging.CompositeLogHandler",
    "messaging-handler": "QuantConnect.Messaging.Messaging",
    "job-queue-handler": "QuantConnect.Queues.JobQueue",
    "api-handler": "QuantConnect.Api.Api",
    "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
    "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
    "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
    "data-channel-provider": "DataChannelProvider",
    "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
    "data-aggregator": "QuantConnect.Lean.Engine.DataFeeds.AggregationManager",
    "symbol-minute-limit": 10000,
    "symbol-second-limit": 10000,
    "symbol-tick-limit": 10000,
    "show-missing-data-logs": False,
    "maximum-warmup-history-days-look-back": 5,
    "maximum-data-points-per-chart-series": 1000000,
    "maximum-chart-series": 30,
    "transaction-log": "",
    "reserved-words-prefix": "@",
    # no QuantConnect cloud account: user id 0 / empty token (only the Alpaca plugin's licence check uses them)
    "job-user-id": "0",
    "api-access-token": "",
    "job-organization-id": "",
    "close-automatically": True,
    "ignore-unknown-asset-holdings": True,
}

_BACKTEST_HANDLERS: dict[str, Any] = {
    "live-mode": False,
    "setup-handler": "QuantConnect.Lean.Engine.Setup.BacktestingSetupHandler",
    "result-handler": "QuantConnect.Lean.Engine.Results.BacktestingResultHandler",
    "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.FileSystemDataFeed",
    "real-time-handler": "QuantConnect.Lean.Engine.RealTime.BacktestingRealTimeHandler",
    "history-provider": ["QuantConnect.Lean.Engine.HistoricalData.SubscriptionDataReaderHistoryProvider"],
    "transaction-handler": "QuantConnect.Lean.Engine.TransactionHandlers.BacktestingTransactionHandler",
}

# Live handlers as in the upstream "live-paper" environment, with the GATED paper brokerage and the
# deterministic SIMULATED data queue instead of QuantConnect's cloud LiveDataQueue.
_LIVE_SIM_HANDLERS: dict[str, Any] = {
    "live-mode": True,
    "live-mode-brokerage": "GatedPaperBrokerage",
    "setup-handler": "QuantConnect.Lean.Engine.Setup.BrokerageSetupHandler",
    "result-handler": "QuantConnect.Lean.Engine.Results.LiveTradingResultHandler",
    "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.LiveTradingDataFeed",
    "data-queue-handler": ["Lat.Lean.Data.LatSimulatedDataQueue"],
    "real-time-handler": "QuantConnect.Lean.Engine.RealTime.LiveTradingRealTimeHandler",
    "transaction-handler": "QuantConnect.Lean.Engine.TransactionHandlers.BacktestingTransactionHandler",
    "history-provider": ["QuantConnect.Lean.Engine.HistoricalData.SubscriptionDataReaderHistoryProvider"],
    # SIMULATED only: the wall clock may be outside US market hours; the gate still requires an open
    # exchange, which this flag makes true for every symbol. Never set by the Alpaca PAPER config.
    "force-exchange-always-open": True,
    "live-holdings": "[]",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def check_run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id):
        raise ValueError(f"illegal run id {run_id!r}")
    return run_id


def pins() -> dict[str, Any]:
    return json.loads(PINS.read_text())


def lean_built() -> bool:
    return LAUNCHER_DLL.is_file() and LAT_DLL.is_file()


def _write_config(run_dir: Path, cfg: dict[str, Any], *, private: bool = False) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    data = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    if private:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(data)
        path.chmod(0o600)
    else:
        path.write_text(data)
    return path


# ------------------------------------------------------------------------------------------ backtest


@dataclass(frozen=True)
class BacktestSpec:
    run_id: str
    data_folder: Path = LEAN_SAMPLE_DATA
    dataset: str = "LEAN_SAMPLE"
    parameters: dict[str, str] = field(default_factory=dict)
    algorithm: str = "Lat.Lean.Algorithms.LatBaselineAlgorithm"


def backtest_config(spec: BacktestSpec) -> dict[str, Any]:
    check_run_id(spec.run_id)
    run_dir = RUNS_DIR / spec.run_id
    cfg = dict(_BASE)
    cfg.update(_BACKTEST_HANDLERS)
    cfg.update({
        "algorithm-type-name": spec.algorithm,
        "algorithm-location": str(LAT_DLL),
        "algorithm-id": spec.run_id,
        "backtest-name": spec.run_id,
        "data-folder": str(spec.data_folder.resolve()) + "/",
        "results-destination-folder": str(run_dir),
        "object-store-root": str(run_dir / "storage"),
        "plugin-directory": str(BUNDLE_DIR),
        "parameters": {k: str(v) for k, v in spec.parameters.items()},
        "lat-dataset": spec.dataset,
    })
    return cfg


def data_range(data_folder: Path, symbol: str, resolution: str) -> tuple[str, str]:
    """First/last trading date (yyyy-MM-dd) available for ``symbol`` in a LEAN equity data folder."""
    sym = symbol.lower()
    if resolution.lower() == "minute":
        days = sorted(p.name[:8] for p in (data_folder / "equity" / "usa" / "minute" / sym).glob("*_trade.zip"))
    else:
        import zipfile

        with zipfile.ZipFile(data_folder / "equity" / "usa" / "daily" / f"{sym}.zip") as zf:
            text = zf.read(zf.namelist()[0]).decode()
        days = sorted(line[:8] for line in text.splitlines() if line[:8].isdigit())
    if not days:
        raise FileNotFoundError(f"no {resolution} data for {symbol} under {data_folder}")
    fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}"  # noqa: E731
    return fmt(days[0]), fmt(days[-1])


def write_backtest_config(spec: BacktestSpec) -> Path:
    return _write_config(RUNS_DIR / spec.run_id, backtest_config(spec))


def _dec(text: Any) -> Decimal | None:
    if text is None:
        return None
    s = str(text).strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _pct_to_fraction(text: Any) -> str | None:
    d = _dec(text)
    return None if d is None else format(d / Decimal(100), "f")


def _equity_curve(result: dict[str, Any]) -> list[dict[str, Any]]:
    charts = result.get("charts") or result.get("Charts") or {}
    chart = charts.get("Strategy Equity") or {}
    series = (chart.get("series") or chart.get("Series") or {}).get("Equity") or {}
    values = series.get("values") or series.get("Values") or []
    out: list[dict[str, Any]] = []
    for v in values:
        if isinstance(v, list) and len(v) >= 2:
            t, close = int(v[0]), v[-1]  # candlestick [t, o, h, l, c] or line [t, y]
        elif isinstance(v, dict):
            t, close = int(v.get("x", 0)), v.get("y")
        else:
            continue
        if close is None:
            continue
        out.append({"time": t, "equity": format(Decimal(str(close)), "f")})
    return _downsample_daily(sorted(out, key=lambda p: p["time"]))


def _downsample_daily(points: list[dict[str, Any]], max_points: int = 3000) -> list[dict[str, Any]]:
    """Last equity sample per UTC day (LEAN samples the equity chart intraday); keeps first/last points."""
    if len(points) <= 2:
        return points
    by_day: dict[str, dict[str, Any]] = {}
    for p in points:
        by_day[datetime.fromtimestamp(p["time"], UTC).strftime("%Y-%m-%d")] = p
    daily = list(by_day.values())
    if len(daily) < 30:  # short (intraday) runs keep LEAN's own samples
        daily = points
    if daily[0] is not points[0]:
        daily.insert(0, points[0])
    if len(daily) > max_points:
        step = len(daily) / max_points
        daily = [daily[int(i * step)] for i in range(max_points - 1)] + [daily[-1]]
    return daily


def summarize_backtest(run_id: str, *, strategy: str, dataset: str, started_at: str, notes: str,
                       is_simulated: bool = False, out_dir: Path = EXPERIMENTS_DIR,
                       data_folder: Path | None = None) -> dict[str, Any]:
    """Parse LEAN's own result files for ``run_id`` into the dashboard/API summary contract.

    ``data_folder``: when it holds the ``lat-export-manifest.json`` written by ``services.market_data`` export-lean,
    its provenance is copied into the summary and a synthetic export is always labelled ``is_simulated``."""
    check_run_id(run_id)
    provenance: dict[str, Any] | None = None
    if data_folder is not None and (data_folder / "lat-export-manifest.json").is_file():
        raw = json.loads((data_folder / "lat-export-manifest.json").read_text())
        provenance = {k: raw.get(k) for k in ("dataset_version", "feed", "is_synthetic", "source_id", "limitations",
                                              "exported_at", "symbols")}
        if raw.get("is_synthetic") is not False:
            is_simulated = True  # fail-safe labelling: unknown or synthetic provenance is never presented as real
    run_dir = RUNS_DIR / run_id
    result_path = run_dir / f"{run_id}.json"
    log_path = run_dir / f"{run_id}-log.txt"
    if not result_path.is_file():
        raise FileNotFoundError(f"LEAN result file missing: {result_path}")
    result = json.loads(result_path.read_text())
    stats: dict[str, Any] = result.get("statistics") or result.get("Statistics") or {}
    runtime: dict[str, Any] = result.get("runtimeStatistics") or result.get("RuntimeStatistics") or {}
    curve = _equity_curve(result)
    if len(curve) < 2:
        raise ValueError("LEAN result has no equity curve; refusing to write an empty summary")
    start_eq, end_eq = Decimal(curve[0]["equity"]), Decimal(curve[-1]["equity"])
    total_return = (end_eq / start_eq - 1) if start_eq else Decimal(0)
    fees = _dec(stats.get("Total Fees")) or Decimal(0)
    orders = int(_dec(stats.get("Total Orders")) or 0)
    out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_excerpt = out_dir / "lean-log-excerpt.txt"
    if log_path.is_file():
        lines = log_path.read_text(errors="replace").splitlines()
        keep = [ln for ln in lines if any(k in ln for k in (
            "Engine.Main", "LAT ", "LAT-", "Algorithm.Run", "BacktestingResultHandler", "Total Orders", "STATISTICS::"))]
        log_excerpt.write_text("\n".join([*lines[:15], "...", *keep[-120:]]) + "\n")
    summary = {
        "run_id": run_id,
        "engine": "LEAN",
        "lean_commit": pins()["lean"]["commit"],
        "strategy": strategy,
        "dataset": dataset,
        "is_simulated": is_simulated,
        "started_at": started_at,
        "metrics": {
            "total_return": format(total_return.quantize(Decimal("0.000001")), "f"),
            "sharpe": str(stats.get("Sharpe Ratio", "")),
            "max_drawdown": _pct_to_fraction(stats.get("Drawdown")) or "0",
            "trades": orders,
            "fees": format(fees, "f"),
            "net_profit": _pct_to_fraction(stats.get("Net Profit")),
            "compounding_annual_return": _pct_to_fraction(stats.get("Compounding Annual Return")),
            "end_equity": format(end_eq, "f"),
            "start_equity": format(start_eq, "f"),
        },
        "lean_statistics": stats,
        "lean_runtime_statistics": runtime,
        "equity_curve": curve,
        "notes": notes,
        "data_provenance": provenance,
        "source_path": str(result_path.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)) if log_path.is_file() else None,
        "log_excerpt_path": str(log_excerpt.relative_to(ROOT)) if log_excerpt.is_file() else None,
        "execution_label": "SIMULATED fills (LEAN backtest fill model) on " + (
            "SYNTHETIC data" if is_simulated else "REAL historical LEAN sample data"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ------------------------------------------------------------------------------------------ live (simulated)


@dataclass(frozen=True)
class LiveSimSpec:
    run_id: str
    bridge_url: str
    pubkey_path: Path
    symbols: tuple[str, ...] = ("SPY",)
    sim_quotes: dict[str, dict[str, str]] = field(
        default_factory=lambda: {"SPY": {"bid": "449.95", "ask": "450.05", "last": "450.00"}})
    exit_after_seconds: int = 60
    rogue: bool = False
    cash_usd: str = "100000"
    engine_id: str | None = None
    journal_path: Path | None = None


def live_sim_config(spec: LiveSimSpec) -> dict[str, Any]:
    check_run_id(spec.run_id)
    run_dir = RUNS_DIR / spec.run_id
    cash = Decimal(spec.cash_usd)
    cfg = dict(_BASE)
    cfg.update(_LIVE_SIM_HANDLERS)
    algorithm, location = (
        ("Lat.Lean.TestAlgorithms.LatRogueStrategyAlgorithm", TEST_ALGOS_DLL) if spec.rogue
        else ("Lat.Lean.Algorithms.LatIntentBridgeAlgorithm", LAT_DLL))
    cfg.update({
        "algorithm-type-name": algorithm,
        "algorithm-location": str(location),
        "algorithm-id": spec.run_id,
        "data-folder": str(LEAN_SAMPLE_DATA.resolve()) + "/",
        "results-destination-folder": str(run_dir),
        "object-store-root": str(run_dir / "storage"),
        "plugin-directory": str(BUNDLE_DIR),
        "live-cash-balance": json.dumps([{"Amount": int(cash) if cash == cash.to_integral() else str(cash), "Currency": "USD"}]),
        "lat-gateway": "LEAN_PAPER_SIM",
        "lat-mode": "PAPER",
        "lat-run-id": spec.run_id,
        "lat-bridge-url": spec.bridge_url,
        "lat-intent-pubkey-path": str(spec.pubkey_path),
        "lat-engine-id": spec.engine_id or f"lean-sim-{spec.run_id}"[:100],
        "lat-symbols": ",".join(spec.symbols),
        "lat-sim-quotes": json.dumps(spec.sim_quotes, sort_keys=True),
        "lat-sim-tick-ms": "500",
        "lat-resolution": "Second",
        "lat-exit-after-seconds": str(spec.exit_after_seconds),
        "lat-lean-commit": pins()["lean"]["commit"],
        "lat-bridge-outbox-path": str(run_dir / "bridge-outbox.jsonl"),
        "lat-gate-journal-path": str(spec.journal_path or (run_dir / "used-intents.tsv")),
        "lat-execution-label": "SIMULATED",
    })
    return cfg


def write_live_sim_config(spec: LiveSimSpec) -> Path:
    return _write_config(RUNS_DIR / spec.run_id, live_sim_config(spec))


def write_api_live_sim_config(run_id: str, *, bridge_url: str | None = None, exit_after_seconds: int = 0,
                              env: dict[str, str] | None = None) -> tuple[Path, Path]:
    """SIMULATED LEAN engine (GatedPaperBrokerage + LatSimulatedDataQueue) for the REAL local API, which must run with
    ``LAT_MODE=PAPER LAT_GATEWAY=LEAN_PAPER_SIM`` (same environment as this call). Uses that namespace's PUBLIC intent
    key and engine-token file (path returned, never the value) and the namespace's durable used-intent journal
    (``var/<ns>/lean/used-intents.tsv``) so an engine restart can never re-execute a consumed intent.

    Returns ``(config_path, engine_token_path)``. Raises Refused (exit 2) / Blocked (exit 3)."""
    environ = dict(os.environ) if env is None else env
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from packages.contracts.enums import GatewayKind, TradingMode
    from packages.core.config import Settings
    from packages.core.signing import PUBLIC_KEY_FILE
    from services.order_ledger.bridge import ENGINE_TOKEN_FILE

    lat_env = {k[4:].lower(): v for k, v in environ.items() if k.startswith("LAT_") and k[4:].lower() in
               ("mode", "gateway", "project_root", "api_host", "api_port")}
    try:
        settings = Settings(**lat_env)  # type: ignore[arg-type]
    except ValueError as exc:
        raise Refused(f"invalid LAT_* settings: {exc}") from exc
    if settings.mode != TradingMode.PAPER:
        raise Refused(f"LAT_MODE={settings.mode.value}: the simulated LEAN engine serves the PAPER API only")
    if settings.gateway != GatewayKind.LEAN_PAPER_SIM:
        raise Refused(f"LAT_GATEWAY={settings.gateway.value}: start the API with LAT_GATEWAY=LEAN_PAPER_SIM for this engine "
                      "(Alpaca PAPER: scripts/lean_paper_alpaca.sh)")
    pub = settings.keys_dir / PUBLIC_KEY_FILE
    token_path = settings.keys_dir / ENGINE_TOKEN_FILE
    if not pub.is_file() or not token_path.is_file():
        raise Blocked(f"BLOCKED: {pub} / {token_path} missing: start the API once (make dev, or "
                      "uv run --frozen python -m services.api.cli init) so it creates its signing key and engine token")
    url = bridge_url or f"http://{settings.api_host}:{settings.api_port}"
    spec = LiveSimSpec(run_id=check_run_id(run_id), bridge_url=url, pubkey_path=pub, symbols=tuple(settings.symbols),
                       sim_quotes=_sim_quotes(tuple(settings.symbols)), exit_after_seconds=exit_after_seconds,
                       engine_id=f"lean-sim-{settings.namespace}-{run_id}"[:100],
                       journal_path=settings.var_dir / "lean" / "used-intents.tsv")
    return write_live_sim_config(spec), token_path


def _sim_quotes(symbols: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Fixed, explicitly SIMULATED quotes (one cent half-spread around a per-symbol constant). Not market data."""
    base = {"SPY": "450.00", "QQQ": "390.00", "AAPL": "225.00", "MSFT": "410.00"}
    out: dict[str, dict[str, str]] = {}
    for s in symbols:
        last = Decimal(base.get(s, "100.00"))
        out[s] = {"bid": str(last - Decimal("0.05")), "ask": str(last + Decimal("0.05")), "last": str(last)}
    return out


# ------------------------------------------------------------------------------------------ Alpaca PAPER


class Blocked(RuntimeError):
    """A precondition outside this codebase is missing (credentials, licence). Exit code 3."""


class Refused(RuntimeError):
    """A safety rule refused the request. Exit code 2."""


# Template keys that may hold secrets: must be EMPTY in the committed template (values only in the 0600 run config).
_SECRET_KEY_HINTS = ("alpaca-api", "access-token", "secret", "password", "api-key")
# The official plugin's ValidateSubscription() authenticates these QuantConnect credentials (see trading/README.md).
QC_USER_ENV, QC_TOKEN_ENV = "LAT_QC_USER_ID", "LAT_QC_API_TOKEN"
QC_ORG_ENV, QC_ACK_ENV = "LAT_QC_ORGANIZATION_ID", "LAT_QC_LICENSE_ACK"


def _check_template(template: dict[str, Any]) -> None:
    if template.get("alpaca-paper-trading") is not True:
        raise Refused('paper template must hard-code "alpaca-paper-trading": true')
    if template.get("live-mode-brokerage") != "GatedAlpacaBrokerage" or template.get("lat-gateway") != "LEAN_ALPACA_PAPER":
        raise Refused("paper template must use GatedAlpacaBrokerage / LEAN_ALPACA_PAPER")
    for k, v in template.items():
        if any(h in k for h in _SECRET_KEY_HINTS) and v not in ("", None):
            raise Refused(f"template must not contain secrets ({k} is set)")


def paper_config(run_id: str, *, key_id: str, secret: str, bridge_url: str, pubkey_path: Path,
                 symbols: tuple[str, ...] = ("SPY",), template_path: Path = PAPER_TEMPLATE,
                 qc_user_id: str = "", qc_api_token: str = "", qc_organization_id: str = "") -> dict[str, Any]:
    """Merge the committed PAPER template with per-run values. Live trading cannot be expressed: there is no
    parameter for it, the template's ``alpaca-paper-trading`` must be literal ``true`` and it is re-asserted after
    the merge (and GatedAlpacaBrokerage hard-codes isPaperTrading=true regardless)."""
    check_run_id(run_id)
    if not key_id or not secret:
        raise Blocked("BLOCKED: Alpaca PAPER key id/secret missing")
    if key_id.startswith("AK"):
        raise Refused("refusing a key id starting with 'AK' (Alpaca LIVE key format); PAPER keys start with 'PK'")
    template = json.loads(template_path.read_text())
    _check_template(template)
    run_dir = RUNS_DIR / run_id
    cfg = dict(_BASE)
    cfg.update({k: v for k, v in template.items() if not k.startswith("_")})
    cfg.update({
        "algorithm-location": str(LAT_DLL),
        "algorithm-id": run_id,
        "data-folder": str(LEAN_SAMPLE_DATA.resolve()) + "/",
        "results-destination-folder": str(run_dir),
        "object-store-root": str(run_dir / "storage"),
        "plugin-directory": str(BUNDLE_DIR),
        "lat-run-id": run_id,
        "lat-bridge-url": bridge_url,
        "lat-intent-pubkey-path": str(pubkey_path),
        "lat-engine-id": f"lean-paper-{run_id}"[:100],
        "lat-symbols": ",".join(symbols),
        "lat-lean-commit": pins()["lean"]["commit"],
        "lat-bridge-outbox-path": str(run_dir / "bridge-outbox.jsonl"),
        # durable across runs: an intent consumed by any earlier PAPER engine run is refused forever
        "lat-gate-journal-path": str(ROOT / "var" / "paper" / "lean" / "used-intents.tsv"),
        "alpaca-api-key": key_id,
        "alpaca-api-secret": secret,
        "alpaca-paper-trading": True,
        "force-exchange-always-open": False,
        "job-user-id": qc_user_id or "0",
        "api-access-token": qc_api_token,
        "job-organization-id": qc_organization_id,
    })
    assert_paper_only(cfg)
    return cfg


def assert_paper_only(cfg: dict[str, Any]) -> None:
    """Final structural check on any Alpaca config this module emits."""
    if cfg.get("alpaca-paper-trading") is not True:
        raise Refused("alpaca-paper-trading must be true")
    if cfg.get("live-mode-brokerage") != "GatedAlpacaBrokerage":
        raise Refused("only the gated Alpaca brokerage may be configured")
    if any(h != "GatedAlpacaBrokerage" for h in (cfg.get("data-queue-handler") or [])):
        raise Refused("only GatedAlpacaBrokerage may be the data queue handler (never the ungated AlpacaBrokerage)")
    if cfg.get("lat-gateway") != "LEAN_ALPACA_PAPER" or cfg.get("lat-mode") != "PAPER":
        raise Refused("gateway/mode must be LEAN_ALPACA_PAPER/PAPER")
    if cfg.get("algorithm-type-name") != "Lat.Lean.Algorithms.LatIntentBridgeAlgorithm":
        raise Refused("only LatIntentBridgeAlgorithm may run against the Alpaca PAPER endpoint")
    if cfg.get("force-exchange-always-open"):
        raise Refused("force-exchange-always-open is a SIMULATION-only setting")
    if (key := cfg.get("alpaca-api-key")) and str(key).startswith("AK"):
        raise Refused("refusing a key id starting with 'AK' (Alpaca LIVE key format)")


def write_paper_config_from_secret_store(run_id: str, *, bridge_url: str, pubkey_path: Path,
                                         symbols: tuple[str, ...] = ("SPY",), env: dict[str, str] | None = None,
                                         use_keychain: bool = True) -> Path:
    """Reads ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY via SecretStore (Keychain -> env -> .env) and writes a
    0600 per-run config (run dir 0700). Never prints secret values. Raises Refused (exit 2) or Blocked (exit 3)."""
    environ = dict(os.environ) if env is None else env
    mode = environ.get("LAT_MODE", "PAPER")
    if mode != "PAPER":
        raise Refused(f"LAT_MODE={mode!r}: the Alpaca LEAN launcher runs in PAPER mode only")
    check_run_id(run_id)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from packages.core.secrets import SecretName, SecretStore

    store = SecretStore(ROOT, env=environ, use_keychain=use_keychain)
    key = store.get(SecretName.ALPACA_PAPER_KEY_ID)
    sec = store.get(SecretName.ALPACA_PAPER_SECRET_KEY)
    if key is None or sec is None:
        raise Blocked(
            "BLOCKED: Alpaca PAPER credentials not found (ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY in macOS "
            "Keychain service 'local-ai-trader', the environment, or the git-ignored .env). Create PAPER keys in the "
            "Alpaca dashboard (paper account) and store them, e.g.: security add-generic-password -s local-ai-trader "
            "-a ALPACA_PAPER_KEY_ID -w   (and the same for ALPACA_PAPER_SECRET_KEY). Note: the official LEAN Alpaca "
            "plugin additionally validates a QuantConnect module licence at start-up (see trading/README.md).")
    if key.get_secret_value().startswith("AK"):
        raise Refused("refusing a key id starting with 'AK' (Alpaca LIVE key format); PAPER keys start with 'PK'")
    qc_user, qc_token = environ.get(QC_USER_ENV, ""), environ.get(QC_TOKEN_ENV, "")
    if not qc_user or not qc_token:
        raise Blocked(
            "BLOCKED: the official QuantConnect LEAN Alpaca plugin (pinned source, AlpacaBrokerage.ValidateSubscription) "
            "authenticates a QuantConnect account and its Alpaca brokerage module licence (product 347) against "
            f"api.quantconnect.com at start-up and exits the engine on failure. Provide {QC_USER_ENV} and "
            f"{QC_TOKEN_ENV} (QuantConnect user id / API token; optionally {QC_ORG_ENV}) for an account that holds "
            "that licence, or decide to use a patched plugin (owner decision, not done here).")
    if environ.get(QC_ACK_ENV) != "1":
        raise Blocked(
            "BLOCKED: the plugin's licence check sends this Mac's machine name, user name, OS version and network "
            f"interface MAC/IP addresses to api.quantconnect.com. Set {QC_ACK_ENV}=1 to accept that for this run.")
    if not pubkey_path.is_file():
        raise Blocked(f"BLOCKED: intent public key {pubkey_path} missing (run: make setup / services.api.cli init)")
    cfg = paper_config(run_id, key_id=key.get_secret_value(), secret=sec.get_secret_value(),
                       bridge_url=bridge_url, pubkey_path=pubkey_path, symbols=symbols or ("SPY",),
                       qc_user_id=qc_user, qc_api_token=qc_token,
                       qc_organization_id=environ.get(QC_ORG_ENV, ""))
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.chmod(0o700)
    return _write_config(run_dir, cfg, private=True)


# ------------------------------------------------------------------------------------------ CLI


def dotnet_path() -> str:
    root = os.environ.get("DOTNET_ROOT", str(ROOT / ".tools" / "dotnet"))
    exe = Path(root) / "dotnet"
    return str(exe) if exe.is_file() else (shutil.which("dotnet") or "dotnet")


def _kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"bad --param {it!r} (expected key=value)")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lat_lean")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest-config")
    b.add_argument("--run-id", required=True)
    b.add_argument("--data-folder", default=str(LEAN_SAMPLE_DATA))
    b.add_argument("--dataset", default="LEAN_SAMPLE")
    b.add_argument("--algorithm", default="Lat.Lean.Algorithms.LatBaselineAlgorithm",
                   choices=("Lat.Lean.Algorithms.LatBaselineAlgorithm", "Lat.Lean.Algorithms.LatResearchReplayAlgorithm"))
    b.add_argument("--param", action="append", default=[])
    s = sub.add_parser("summarize")
    s.add_argument("--run-id", required=True)
    s.add_argument("--strategy", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--started-at", required=True)
    s.add_argument("--notes", default="")
    s.add_argument("--simulated", action="store_true")
    s.add_argument("--data-folder", default=None, help="dataset folder (reads lat-export-manifest.json provenance)")
    pp = sub.add_parser("paper-config")
    pp.add_argument("--run-id", required=True)
    pp.add_argument("--bridge-url", default="http://127.0.0.1:8765")
    pp.add_argument("--pubkey", required=True)
    pp.add_argument("--symbols", default="SPY")
    la = sub.add_parser("live-sim-api-config")
    la.add_argument("--run-id", required=True)
    la.add_argument("--bridge-url", default=None)
    la.add_argument("--exit-after", type=int, default=0)
    sub.add_parser("pin").add_argument("path", nargs="+")
    dr = sub.add_parser("data-range")
    dr.add_argument("--data-folder", required=True)
    dr.add_argument("--symbol", required=True)
    dr.add_argument("--resolution", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "backtest-config":
            spec = BacktestSpec(run_id=args.run_id, data_folder=Path(args.data_folder), dataset=args.dataset,
                                parameters=_kv(args.param), algorithm=args.algorithm)
            print(write_backtest_config(spec))
        elif args.cmd == "summarize":
            out = summarize_backtest(args.run_id, strategy=args.strategy, dataset=args.dataset,
                                     started_at=args.started_at, notes=args.notes, is_simulated=args.simulated,
                                     data_folder=Path(args.data_folder) if args.data_folder else None)
            m = out["metrics"]
            print(f"summary experiments/lean/{args.run_id}/summary.json: return={m['total_return']} "
                  f"sharpe={m['sharpe']} maxdd={m['max_drawdown']} orders={m['trades']} fees={m['fees']}")
        elif args.cmd == "paper-config":
            raise Refused("LEAN paper execution is retired; use make paper (official alpaca-py SDK)")
        elif args.cmd == "live-sim-api-config":
            cfg_path, token_path = write_api_live_sim_config(args.run_id, bridge_url=args.bridge_url,
                                                             exit_after_seconds=args.exit_after)
            print(f"{cfg_path}\t{token_path}")
        elif args.cmd == "data-range":
            print(" ".join(data_range(Path(args.data_folder), args.symbol, args.resolution)))
        elif args.cmd == "pin":
            node: Any = pins()
            for k in args.path:
                node = node[k]
            print(node)
    except Blocked as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
