"""Research CLI (no LLM, no orders): ``uv run --frozen python -m services.research.cli <command>``.

Commands
--------
``datasets``                        list stored datasets (MarketDataStore registry)
``baselines   [--dataset VER]``     no-LLM baselines + PIT backtests -> experiments/baselines/<run_id>/
``signals     --asof ISO [--dataset VER]``  score the universe at ``asof`` -> candidate or NO_TRADE
``walkforward [--dataset VER]``     LightGBM/CatBoost walk-forward -> experiments/ml/<run_id>/
``qlib        [--dataset VER]``     Qlib v0.9.8 LightGBM workflow (isolated env) -> experiments/qlib/<run_id>/
``train-models [--dataset VER]``    local quant model suite (walk-forward, net-Sharpe selection) -> experiments/models/<ver>/
                                    (default dataset: newest contiguous 1Min one; the LEAN sample only when named)
``predict      --asof ISO [--dataset VER]``  point-in-time PredictionSets from the latest valid model registry

Data comes from ``services.market_data.store.MarketDataStore`` (imported lazily) when it exists. With
``--source auto`` (default) and no stored dataset, the commands fall back to the explicitly SYNTHETIC research
fixture and say so in every output; ``--source store`` refuses to fall back; ``--source synthetic`` forces it.
Stored datasets with injected defects are never picked implicitly (pass ``--dataset`` to test failure handling).
RAW stored datasets get their point-in-time corporate actions applied (``adjustments.apply_corporate_actions``;
``signals`` only uses records published by ``--asof``). Every result is labelled with its dataset version, the
adjustments applied and whether it is simulated. NO EDGE IS CLAIMED.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from packages.core.config import load_settings
from packages.core.ids import new_causal_id

from .adjustments import apply_corporate_actions
from .costs import CostModel
from .frames import BarDataError, is_intraday, utc_ts
from .provenance import NO_EDGE_NOTE, dumps, new_run_id, relpath, utc_now_iso, write_json
from .splits import InsufficientDataError
from .synthetic import DEFAULT_SYNTHETIC_VERSION, make_synthetic_bars

SYNTHETIC_SYMBOLS = ("AAA", "BBB", "CCC")


class CliError(RuntimeError):
    """User-facing failure (exit code 2)."""


@dataclass
class LoadedData:
    bars: pd.DataFrame
    dataset_version: str
    timeframe: str
    symbols: list[str]
    source: str  # "store" | "synthetic"
    manifest: dict[str, Any] | None
    note: str
    dividends: dict[str, dict[date, Decimal]] = field(default_factory=lambda: {})
    adjustments: dict[str, Any] = field(default_factory=lambda: {})


def _adjust(store: Any, df: pd.DataFrame, manifest: dict[str, Any], asof: datetime | None
            ) -> tuple[pd.DataFrame, dict[str, dict[date, Decimal]], dict[str, Any]]:
    """Apply the dataset's point-in-time corporate actions to RAW bars (``adjustments`` module)."""
    policy = str(manifest.get("adjustment") or "raw")
    loader = getattr(store, "load_corporate_actions", None)
    if loader is None:
        return df, {}, {"input_adjustment": policy, "applied": "none: store exposes no corporate actions",
                        "warning": "RAW prices: a split inside the period would corrupt results" if policy == "raw"
                        else ""}
    symbols = sorted(set(df["symbol"].astype(str)))
    actions = loader(symbols, asof=asof, dataset_version=str(manifest["dataset_version"]))
    res = apply_corporate_actions(df, actions, adjustment=policy, asof=asof)
    return res.bars, res.dividends, res.report


def _experiments_dir(args: argparse.Namespace) -> Path:
    return Path(args.out) if getattr(args, "out", None) else load_settings().experiments_dir


def _store(datasets_dir: Path | None = None) -> Any | None:
    """MarketDataStore when the market-data service is installed, else None (lazy import by design)."""
    try:
        from services.market_data.store import MarketDataStore
    except ImportError:
        return None
    try:
        return MarketDataStore(datasets_dir or load_settings().datasets_dir)
    except Exception as exc:  # a broken store must not silently turn into synthetic data
        raise CliError(f"market data store unavailable: {type(exc).__name__}: {exc}") from exc


def _pick_manifest(store: Any, dataset: str | None, prefer_tf: str | None) -> dict[str, Any] | None:
    if dataset:
        m = store.get_manifest(dataset)
        if m is None:
            known = [x.get("dataset_version") for x in store.list_manifests()]
            raise CliError(f"dataset {dataset!r} not found; stored datasets: {known}")
        return dict(m)
    manifests = [dict(m) for m in store.list_manifests(include_defective=False)]
    if not manifests:
        return None
    if prefer_tf:
        preferred = [m for m in manifests if prefer_tf in (m.get("timeframes") or [])]
        if preferred:
            return preferred[-1]
    return manifests[-1]


def load_data(
    *,
    source: str,
    dataset: str | None,
    timeframe: str | None,
    symbols: Sequence[str] | None,
    prefer_timeframe: str,
    asof: datetime | None = None,
    synthetic_sessions: int = 500,
    datasets_dir: Path | None = None,
) -> LoadedData:
    """Resolve and load bars according to ``--source`` (see module docstring)."""
    if source not in {"auto", "store", "synthetic"}:
        raise CliError(f"unknown --source {source!r}")
    if dataset == DEFAULT_SYNTHETIC_VERSION:
        if source == "store":
            raise CliError(f"{DEFAULT_SYNTHETIC_VERSION} is the in-memory research fixture, not a stored dataset")
        source = "synthetic"
    if source != "synthetic":
        store = _store(datasets_dir)
        if store is None:
            if source == "store":
                raise CliError("services.market_data.store is not available; cannot use --source store")
        else:
            manifest = _pick_manifest(store, dataset, timeframe or prefer_timeframe)
            if manifest is not None:
                tfs = list(manifest.get("timeframes") or [])
                tf = timeframe or (prefer_timeframe if prefer_timeframe in tfs else (tfs[0] if tfs else prefer_timeframe))
                if tfs and tf not in tfs:
                    raise CliError(f"dataset {manifest['dataset_version']} has timeframes {tfs}, not {tf}")
                syms = list(symbols) if symbols else list(manifest.get("symbols") or [])
                ver = str(manifest["dataset_version"])
                df = store.load_bars_df(syms, None, None, tf, asof=asof, dataset_version=ver, prices="decimal")
                if df.empty:
                    raise CliError(f"dataset {ver} returned no {tf} bars for {syms}" + (f" at asof {asof}" if asof else ""))
                df, dividends, adj = _adjust(store, df, manifest, asof)
                return LoadedData(df, ver, tf, sorted(set(df["symbol"].astype(str))), "store", manifest,
                                  f"stored dataset {ver} (feed {manifest.get('feed')}, synthetic={manifest.get('is_synthetic')})",
                                  dividends, adj)
            if source == "store" or dataset:
                raise CliError("no stored dataset found (run: make fixture) and --source store/--dataset was requested")
    if dataset and dataset != DEFAULT_SYNTHETIC_VERSION:
        raise CliError(f"--dataset {dataset!r} requested but only the synthetic research fixture is available")
    tf = timeframe or prefer_timeframe
    syms = tuple(symbols) if symbols else SYNTHETIC_SYMBOLS
    sessions = synthetic_sessions if not is_intraday(tf) else 10
    # one drifting and one high-volatility symbol so the momentum and volatility filters actually bind
    df = make_synthetic_bars(syms, timeframe=tf, n_sessions=sessions, drifts={syms[0]: 0.0008}, vols={syms[-1]: 0.04})
    return LoadedData(df, DEFAULT_SYNTHETIC_VERSION, tf, sorted(syms), "synthetic", None,
                      "SYNTHETIC research fixture (no stored dataset used) - not market data", {},
                      {"input_adjustment": "n/a", "applied": "none (synthetic fixture has no corporate actions)"})


def _symbols(arg: str | None) -> list[str] | None:
    return [s.strip().upper() for s in arg.split(",") if s.strip()] if arg else None


def _data(args: argparse.Namespace, prefer_tf: str, asof: datetime | None = None) -> LoadedData:
    return load_data(source=args.source, dataset=args.dataset, timeframe=args.timeframe, symbols=_symbols(args.symbols),
                     prefer_timeframe=prefer_tf, asof=asof,
                     datasets_dir=Path(args.datasets_dir) if args.datasets_dir else None)


def _adj_summary(adj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in adj.items() if k != "splits"} | {
        "split_events": sum(len(v) for v in (adj.get("splits") or {}).values())
    }


def _emit(obj: Any) -> None:
    sys.stdout.write(dumps(obj) + "\n")


# --------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------


def cmd_datasets(args: argparse.Namespace) -> int:
    store = _store(Path(args.datasets_dir) if args.datasets_dir else None)
    if store is None:
        _emit({"available": False, "reason": "services.market_data.store not importable"})
        return 0
    rows = [
        {k: m.get(k) for k in ("dataset_version", "kind", "feed", "is_synthetic", "symbols", "timeframes", "start", "end")}
        | {"defects_injected": bool(m.get("defects_injected"))}
        for m in store.list_manifests()
    ]
    _emit({"datasets": rows})
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .baselines import run_all_baselines

    data = _data(args, "1Day")
    out = run_all_baselines(data.bars, CostModel(), _experiments_dir(args) / "baselines", symbols=data.symbols,
                            dividends=data.dividends, data_adjustments=data.adjustments)
    _emit({
        "command": "baselines",
        "run_dir": out["run_dir"],
        "data": data.note,
        "dataset_version": out["dataset_version"],
        "is_simulated": out["is_simulated"],
        "timeframe": out["timeframe"],
        "aggregate": out["aggregate"],
        "data_adjustments": _adj_summary(data.adjustments),
        "summary_files": len(out["summary_files"]),
        "no_edge_note": NO_EDGE_NOTE,
    })
    return 0


def cmd_signals(args: argparse.Namespace) -> int:
    from .signals import QuantSignalEngine

    try:
        asof = utc_ts(args.asof).to_pydatetime()
    except (BarDataError, ValueError) as exc:
        raise CliError(f"--asof must be a timezone-aware ISO timestamp (e.g. 2026-11-04T15:30:00Z): {exc}") from exc
    data = _data(args, "1Min", asof=asof)
    model = None
    if args.model_run:
        from .ml import load_trained_model

        model = load_trained_model(args.model_run)
    engine = QuantSignalEngine(CostModel(), min_net_edge_bps=Decimal(args.min_net_edge_bps),
                               horizon_minutes=args.horizon_minutes, model=model)
    causal_id = new_causal_id()
    report = engine.generate_report(data.bars, asof, data.symbols, causal_id)
    run_id = new_run_id("sig")
    payload = {
        "command": "signals",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "asof": asof,
        "causal_id": causal_id,
        "data": data.note,
        "dataset_version": data.dataset_version,
        "timeframe": data.timeframe,
        "outcome": report.outcome,
        "no_trade_reason": report.no_trade_reason,
        "candidate": report.candidate.symbol if report.candidate else None,
        "data_adjustments": data.adjustments,
        "signals": [s.model_dump(mode="json") for s in report.signals],
        "skipped": report.skipped,
        "model": engine.model_version,
        "no_edge_note": NO_EDGE_NOTE,
    }
    path = write_json(_experiments_dir(args) / "signals" / run_id / "signals.json", payload)
    payload["written_to"] = relpath(path)
    _emit(payload)
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from .ml import WalkForwardConfig, run_walk_forward

    data = _data(args, "1Day")
    horizon = args.horizon_bars or (5 if not is_intraday(data.timeframe) else 30)
    cfg = WalkForwardConfig(horizon_bars=horizon, calibration=args.calibration, n_folds=args.folds)
    out = run_walk_forward(data.bars, CostModel(), _experiments_dir(args) / "ml", config=cfg,
                           allow_final_test_reuse=args.allow_final_test_reuse, data_adjustments=data.adjustments)
    p = out.payload
    _emit({
        "command": "walkforward",
        "run_dir": p["run_dir"],
        "data": data.note,
        "dataset_version": p["dataset_version"],
        "is_simulated": p["is_simulated"],
        "timeframe": p["timeframe"],
        "horizon_bars": p["horizon_bars"],
        "panel_rows": p["panel_rows"],
        "data_adjustments": _adj_summary(data.adjustments),
        "variants": {
            k: {"mean_fold_ic": v.get("mean_fold_ic"), "pooled_rank_ic": v.get("pooled_rank_ic"),
                "p_value": v.get("p_value"), "bh_q_value": v.get("bh", {}).get("q_value"),
                "bh_reject": v.get("bh", {}).get("reject"), "pooled_brier_calibrated": v.get("pooled_brier_calibrated")}
            for k, v in p["development"]["summary"].items()
        },
        "selection": p["selection"],
        "final_test": {k: v for k, v in p["final_test"].items() if k != "metrics"}
        | {"metrics": {k: {m: v.get(m) for m in ("rank_ic", "ic_p_value_one_sided", "brier_calibrated",
                                                  "brier_climatology", "mean_net_bps_when_selected", "n")}
                       for k, v in p["final_test"].get("metrics", {}).items()}},
        "probability_calibrated_for_signals": p["probability_calibrated_for_signals"],
        "no_edge_note": NO_EDGE_NOTE,
    })
    return 0


def cmd_qlib(args: argparse.Namespace) -> int:
    from .qlib_bridge import run_qlib_workflow

    data = _data(args, "1Day")
    man = run_qlib_workflow(data.bars, _experiments_dir(args) / "qlib", horizon=args.horizon,
                            data_adjustments=data.adjustments)
    _emit({"command": "qlib", "data": data.note} | {k: v for k, v in man.items() if k not in ("conversion", "files_sha256")})
    return 0 if man.get("status") in {"OK", "SKIPPED"} else 3


def _quote_spreads(data: LoadedData, args: argparse.Namespace) -> Any:
    """PIT quote spreads (bps) of a stored dataset for the model suite's cost labels; None when unavailable."""
    from .models.targets import QuoteSpreads

    if data.source != "store" or getattr(args, "no_quotes", False) or data.manifest is None:
        return None
    store = _store(Path(args.datasets_dir) if args.datasets_dir else None)
    getter = getattr(store, "quote_columns", None)
    if store is None or getter is None:
        return None
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from packages.contracts.enums import DataFeed

    feed = DataFeed(str(data.manifest.get("feed")))
    frames: list[pd.DataFrame] = []
    for sym in data.symbols:
        try:
            qc = getter(sym, feed, dataset_version=data.dataset_version)
        except Exception:  # noqa: S112 - a symbol without readable quotes falls back to the bucket estimate
            continue
        if len(qc) == 0:
            continue
        bid = pc.cast(qc.table.column("bid"), pa.float64()).to_numpy()
        ask = pc.cast(qc.table.column("ask"), pa.float64()).to_numpy()
        mid = (bid + ask) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = np.where((bid > 0) & (ask >= bid), (ask - bid) / mid * 10_000.0, np.nan)
        frames.append(pd.DataFrame({"symbol": sym, "available_at_ns": np.asarray(qc.available_at_us, dtype=np.int64) * 1000,
                                    "spread_bps": spread}))
    if not frames:
        return None
    return QuoteSpreads.from_frame(pd.concat(frames, ignore_index=True), source=f"stored quotes ({feed.value})")


def _filter_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    ts = pd.to_datetime(out["start_ts"], utc=True)
    if start:
        out = out.loc[(ts >= utc_ts(start)).to_numpy()]
        ts = pd.to_datetime(out["start_ts"], utc=True)
    if end:
        out = out.loc[(ts < utc_ts(end)).to_numpy()]
    if out.empty:
        raise CliError(f"no bars inside --start {start} / --end {end}")
    return out


MODEL_SUITE_EXCLUDED_KINDS = frozenset({"lean_sample_import"})


def pick_models_manifest(store: Any) -> dict[str, Any] | None:
    """Default dataset for ``train-models`` without ``--dataset``: the newest non-defective stored dataset with
    1-minute bars that is a contiguous history (LEAN's bundled sample is a set of isolated vendor sample days across
    2008-2023 - train on it only when named explicitly, with a ``--start/--end`` window)."""
    manifests = [dict(m) for m in store.list_manifests(include_defective=False)]
    ok = [m for m in manifests if "1Min" in (m.get("timeframes") or [])
          and str(m.get("kind") or "") not in MODEL_SUITE_EXCLUDED_KINDS and not m.get("defects_injected")]
    return ok[-1] if ok else None


def _resolve_models_dataset(args: argparse.Namespace) -> None:
    if args.dataset or args.source == "synthetic":
        return
    store = _store(Path(args.datasets_dir) if args.datasets_dir else None)
    if store is None:
        return
    m = pick_models_manifest(store)
    if m is not None:
        args.dataset = str(m["dataset_version"])
    elif list(store.list_manifests(include_defective=False)):
        known = [x.get("dataset_version") for x in store.list_manifests(include_defective=False)]
        raise CliError(f"no contiguous 1Min dataset for the model suite among {known}; pass --dataset (and a "
                       "--start/--end window for the LEAN sample)")


def cmd_train_models(args: argparse.Namespace) -> int:
    from .models.config import SuiteConfig
    from .models.registry import registry_root, save_registry
    from .models.walkforward import train_suite

    _resolve_models_dataset(args)
    data = _data(args, "1Min")
    if not is_intraday(data.timeframe):
        raise CliError(f"train-models needs intraday bars (got {data.timeframe}); pass --timeframe 1Min")
    bars = _filter_window(data.bars, args.start, args.end)
    quotes = _quote_spreads(data, args)
    base = registry_root(_experiments_dir(args))
    res = train_suite(bars, CostModel(), base, config=SuiteConfig(n_folds=args.folds), quotes=quotes,
                      allow_final_test_reuse=args.allow_final_test_reuse, data_adjustments=_adj_summary(data.adjustments),
                      data_note=data.note + (f"; window {args.start or '-'} .. {args.end or '-'}" if args.start or args.end
                                             else ""))
    reg = save_registry(res, base)
    rep = reg.report()
    _emit({
        "command": "train-models",
        "registry_version": reg.registry_version,
        "registry_dir": relpath(reg.path),
        "data": data.note,
        "dataset_version": rep["dataset_version"],
        "is_simulated": rep["is_simulated"],
        "timeframe": rep["timeframe"],
        "rows": res.meta["rows"],
        "symbols": res.meta["symbols"],
        "spread_source": res.meta["targets_definition"]["spread_source"],
        "selected": res.meta["selected"],
        "candidates": {t["target"]: [{k: c.get(k) for k in ("model_id", "net_sharpe", "net_return", "n_test", "ic",
                                                            "brier", "selected", "rejected_reason")}
                                     for c in t["candidates"]] for t in rep["targets"]},
        "calibration": next((t["calibration"] for t in rep["targets"] if t["target"] == "success"), None)
        and {k: v for k, v in next(t["calibration"] for t in rep["targets"] if t["target"] == "success").items()
             if k != "reliability"},
        "walk_forward": rep["walk_forward"],
        "multiple_testing": rep["multiple_testing"],
        "probability_calibrated": res.meta["probability_calibrated"],
        "notes": rep["notes"],
    })
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .models.predict import PredictionService
    from .models.registry import load_registry
    from .signals import QuantSignalEngine

    try:
        asof = utc_ts(args.asof).to_pydatetime()
    except (BarDataError, ValueError) as exc:
        raise CliError(f"--asof must be a timezone-aware ISO timestamp (e.g. 2026-11-04T15:30:00Z): {exc}") from exc
    reg = load_registry(_experiments_dir(args), args.registry)
    if reg is None:
        raise CliError("no valid model registry (missing, corrupt or tampered); run train-models first - fail closed")
    data = _data(args, reg.timeframe, asof=asof)
    svc = PredictionService(reg, CostModel(), signal_horizon_minutes=args.signal_horizon_minutes)
    spreads = _latest_spreads(data, args, asof)
    preds = svc.predict(data.bars, asof, data.symbols, quote_spreads_bps=spreads)
    engine = QuantSignalEngine(CostModel(), horizon_minutes=args.signal_horizon_minutes, model=svc)
    report = engine.generate_report(data.bars, asof, data.symbols, new_causal_id(), quote_spreads_bps=spreads)
    got = {p.symbol for p in preds}
    _emit({
        "command": "predict",
        "registry_version": reg.registry_version,
        "registry_dataset_version": reg.dataset_version,
        "asof": asof,
        "data": data.note,
        "dataset_version": data.dataset_version,
        "is_simulated": bool(reg.is_simulated or any(p.is_simulated for p in preds)),
        "selected": reg.meta["selected"],
        "predictions": [p.model_dump(mode="json") for p in preds],
        "no_prediction": sorted(set(data.symbols) - got),
        "signal_engine": {"outcome": report.outcome, "candidate": report.candidate.symbol if report.candidate else None,
                          "no_trade_reason": report.no_trade_reason, "model_version": engine.model_version},
        "no_edge_note": NO_EDGE_NOTE,
    })
    return 0


def _latest_spreads(data: LoadedData, args: argparse.Namespace, asof: datetime) -> dict[str, Decimal] | None:
    """Last stored quote spread per symbol published by ``asof`` (at most 5 minutes old); None without a store."""
    if data.source != "store" or data.manifest is None:
        return None
    store = _store(Path(args.datasets_dir) if args.datasets_dir else None)
    if store is None or not hasattr(store, "latest_quote"):
        return None
    from datetime import timedelta

    from packages.contracts.enums import DataFeed

    from .costs import spread_bps_from_quote

    out: dict[str, Decimal] = {}
    for sym in data.symbols:
        q = store.latest_quote(sym, asof, feed=DataFeed(str(data.manifest.get("feed"))))
        if q is None or asof - q.available_at_ts > timedelta(minutes=5):
            continue
        try:
            out[sym] = spread_bps_from_quote(q.bid, q.ask)
        except ValueError:
            continue
    return out or None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m services.research.cli", description="No-LLM quantitative research")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", default=None, help="stored dataset_version (default: latest suitable)")
        p.add_argument("--source", default="auto", choices=["auto", "store", "synthetic"])
        p.add_argument("--timeframe", default=None, help="e.g. 1Day, 1Min (default per command)")
        p.add_argument("--symbols", default=None, help="comma-separated subset")
        p.add_argument("--out", default=None, help="experiments base dir (default: <project>/experiments)")
        p.add_argument("--datasets-dir", default=None, help="MarketDataStore root (default: settings.datasets_dir)")

    p = sub.add_parser("datasets", help="list stored datasets")
    p.add_argument("--datasets-dir", default=None)
    p = sub.add_parser("baselines", help="run no-LLM baselines")
    common(p)
    p = sub.add_parser("signals", help="score candidates at --asof")
    common(p)
    p.add_argument("--asof", required=True)
    p.add_argument("--min-net-edge-bps", default="5")
    p.add_argument("--horizon-minutes", type=int, default=60)
    p.add_argument("--model-run", default=None, help="experiments/ml/<run_id> to score with its selected model")
    p = sub.add_parser("walkforward", help="LightGBM/CatBoost walk-forward")
    common(p)
    p.add_argument("--horizon-bars", type=int, default=None)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--calibration", default="isotonic", choices=["isotonic", "sigmoid"])
    p.add_argument("--allow-final-test-reuse", action="store_true",
                   help="re-evaluate an already-evaluated frozen test (result labelled CONTAMINATED)")
    p = sub.add_parser("qlib", help="Qlib v0.9.8 LightGBM workflow in the isolated env")
    common(p)
    p.add_argument("--horizon", type=int, default=5)
    p = sub.add_parser("train-models", help="train + walk-forward-select the local quant model suite")
    common(p)
    p.add_argument("--start", default=None, help="only bars starting at/after this ISO time")
    p.add_argument("--end", default=None, help="only bars starting before this ISO time")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--no-quotes", action="store_true", help="ignore stored quotes (liquidity-bucket spread estimates)")
    p.add_argument("--allow-final-test-reuse", action="store_true",
                   help="re-evaluate an already-evaluated frozen test (result labelled CONTAMINATED)")
    p = sub.add_parser("predict", help="point-in-time PredictionSets from the latest valid model registry")
    common(p)
    p.add_argument("--asof", required=True)
    p.add_argument("--registry", default=None, help="registry_version (default: newest complete registry)")
    p.add_argument("--signal-horizon-minutes", type=int, default=60)
    return ap


COMMANDS = {
    "datasets": cmd_datasets,
    "baselines": cmd_baselines,
    "signals": cmd_signals,
    "walkforward": cmd_walkforward,
    "qlib": cmd_qlib,
    "train-models": cmd_train_models,
    "predict": cmd_predict,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return COMMANDS[args.command](args)
    except (CliError, BarDataError, InsufficientDataError, ValueError) as exc:  # data problems: fail closed, exit 2
        sys.stderr.write(json.dumps({"error": type(exc).__name__, "detail": str(exc)}) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
