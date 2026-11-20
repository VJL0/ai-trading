"""Market-data CLI: ``uv run --frozen python -m services.market_data.cli <command>``.

Commands (all print JSON to stdout; errors go to stderr with a non-zero exit code):

``fixture``             create/reuse the deterministic SYNTHETIC fixture (``--inject-defects`` for validator tests)
``import-lean-sample``  import the REAL sample data bundled in the pinned LEAN clone (feed LEAN_SAMPLE)
``export-lean``         export a stored dataset to LEAN's on-disk format (``--dataset VER --out DIR``)
``validate``            integrity check (manifest hash, row counts, Parquet sha256) + re-run the validators
``ingest-alpaca``       backfill real Alpaca IEX (or explicitly configured SIP) bars; needs PAPER keys
``universe``            point-in-time eligibility snapshot from the risk policy defaults
``status``              datasets, active series, row counts, credential *presence* and feed limitations

Exit codes: 0 ok, 1 validation/integrity failure, 2 configuration problem (e.g. missing credentials/dataset).
Secrets are never printed: only their presence (true/false) is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.config import PROJECT_ROOT, load_settings

from .common import FEED_MODE_LIMITATIONS, iso


class CliError(Exception):
    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False, default=str))


def _datasets_dir(args: argparse.Namespace) -> Path:
    return Path(args.datasets_dir) if args.datasets_dir else load_settings().datasets_dir


def _db(args: argparse.Namespace) -> Any:
    """Operational SQLite (data_quality_events, md_* tables) unless --no-db; failures are reported, not fatal."""
    if getattr(args, "no_db", False):
        return None
    try:
        from packages.core.db import Database

        return Database(Path(args.db) if getattr(args, "db", None) else load_settings().db_path)
    except Exception as exc:  # the analytical import must still be auditable via its manifest
        print(f"warning: operational database unavailable ({type(exc).__name__}: {exc}); "
              "data-quality events are kept only in the dataset manifest", file=sys.stderr)
        return None


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("dataset_version", "kind", "series_mode", "feed", "is_synthetic", "source_id", "symbols", "timeframes",
            "start", "end", "row_counts", "lean_commit", "vintage", "manifest_path", "created_at")
    out = {k: m.get(k) for k in keys}
    out["defects_injected"] = len(m.get("defects_injected") or [])
    out["limitations"] = m.get("limitations", [])
    return out


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise CliError("timestamps must include a timezone (e.g. 2026-11-04T20:00:00Z)")
    return dt.astimezone(UTC)


# ------------------------------------------------------------------------------------------ commands
def cmd_fixture(args: argparse.Namespace) -> int:
    from .fixtures import DEFAULT_SEED, DEFAULT_SESSIONS, DEFAULT_SYMBOLS, ensure_synthetic_fixture

    manifest = ensure_synthetic_fixture(
        _datasets_dir(args), symbols=tuple(args.symbols or DEFAULT_SYMBOLS), sessions=args.sessions or DEFAULT_SESSIONS,
        seed=DEFAULT_SEED if args.seed is None else args.seed, end_date=_parse_date(args.end_date),
        inject_defects=args.inject_defects)
    out = _summary(manifest)
    out["label"] = "SYNTHETIC — not market data"
    _emit(out)
    return 0


def cmd_import_lean_sample(args: argparse.Namespace) -> int:
    from .lean_sample import DEFAULT_LEAN_ROOT, import_lean_sample

    root = Path(args.lean_root) if args.lean_root else DEFAULT_LEAN_ROOT
    if not (root / "Data" / "equity" / "usa").is_dir():
        raise CliError(f"LEAN sample data not found under {root}/Data/equity/usa; clone/build LEAN first (make lean-build)")
    db = _db(args)
    try:
        manifest = import_lean_sample(_datasets_dir(args), lean_root=root, symbols=args.symbols or None,
                                      resolutions=tuple(args.resolutions), include_quotes=not args.no_quotes, db=db)
    finally:
        if db is not None:
            db.close()
    out = _summary(manifest)
    out["validation"] = manifest.get("extra", {}).get("validation")
    out["minute_gap_count"] = manifest.get("extra", {}).get("minute_gap_count")
    out["daily_gap_count"] = manifest.get("extra", {}).get("daily_gap_count")
    _emit(out)
    return 0


def _resolve_dataset(store: Any, dataset: str | None, *, default_feed: DataFeed | None = DataFeed.SYNTHETIC) -> dict[str, Any]:
    if dataset:
        m = store.get_manifest(dataset)
        if m is None:
            raise CliError(f"unknown dataset {dataset!r}; see `status` for the available dataset_versions")
        return m
    m = store.latest_manifest(default_feed)
    if m is None:
        raise CliError(f"no {default_feed.value if default_feed else ''} dataset stored; run `fixture` first")
    return m


def cmd_export_lean(args: argparse.Namespace) -> int:
    from .lean_export import export_bars_to_lean
    from .lean_sample import DEFAULT_LEAN_ROOT
    from .store import MarketDataStore

    store = MarketDataStore(_datasets_dir(args))
    manifest = _resolve_dataset(store, args.dataset)
    version = str(manifest["dataset_version"])
    symbols = [s.upper() for s in (args.symbols or manifest.get("symbols") or [])]
    if not symbols:
        raise CliError(f"dataset {version} has no symbols")
    bars = []
    for tf in manifest.get("timeframes") or []:
        if tf in ("1Min", "1Day"):
            bars += store.load_bars(symbols, None, None, tf, dataset_version=version)
    if not bars:
        raise CliError(f"dataset {version} has no 1Min/1Day bars to export")
    quotes = store.load_quotes(symbols, None, None, dataset_version=version) if args.quotes else []
    meta = None if args.no_metadata else (DEFAULT_LEAN_ROOT / "Data")
    out_dir = Path(args.out)
    summary = export_bars_to_lean(bars, out_dir, quotes=quotes,
                                  copy_metadata_from=meta if meta is not None and meta.is_dir() else None)
    provenance = {"exported_at": iso(datetime.now(UTC)), "dataset_version": version, "feed": manifest.get("feed"),
                  "is_synthetic": manifest.get("is_synthetic"), "source_id": manifest.get("source_id"),
                  "limitations": manifest.get("limitations", []), "symbols": summary["symbols"], "rows": summary["rows"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lat-export-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _emit({**provenance, "lean_data_dir": str(out_dir), "files": len(summary["files"]),
           "metadata_copied": summary["metadata_copied"]})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .calendar import get_calendar
    from .quality import (
        QualityPolicy,
        gaps_within_observed_sessions,
        record_quality_events,
        validate_bars,
        validate_quotes,
        with_gaps,
    )
    from .store import MarketDataStore

    store = MarketDataStore(_datasets_dir(args))
    manifest = _resolve_dataset(store, args.dataset, default_feed=None)
    version = str(manifest["dataset_version"])
    integrity = store.verify_dataset(version)
    symbols = list(manifest.get("symbols") or [])
    cal = get_calendar()
    policy = QualityPolicy()
    if manifest.get("kind") == "lean_sample_import":
        policy = policy.with_actions(stale_quote="IGNORE")  # historical bundle: receipt time is the vintage
    reports: dict[str, Any] = {}
    db = _db(args) if args.record else None
    try:
        bars = []
        for tf in manifest.get("timeframes") or []:
            bars += store.load_bars(symbols, None, None, tf, dataset_version=version) if symbols else []
        if bars:
            from dataclasses import replace

            rep = validate_bars(bars, cal, replace(policy, report_gaps=False))
            rep = with_gaps(rep, gaps_within_observed_sessions(rep.accepted, cal))
            reports["bars"] = rep.summary()
            if db is not None:
                record_quality_events(db, rep, DataFeed(manifest["feed"]), context={"dataset_version": version,
                                                                                    "kind": "cli-validate"})
        quotes = store.load_quotes(symbols, None, None, dataset_version=version) if symbols and args.quotes else []
        if quotes:
            qrep = validate_quotes(quotes, cal, policy)
            reports["quotes"] = qrep.summary()
            if db is not None:
                record_quality_events(db, qrep, DataFeed(manifest["feed"]), context={"dataset_version": version,
                                                                                     "kind": "cli-validate"})
    finally:
        if db is not None:
            db.close()
    rejected = sum(int(r.get("rejected", 0)) for r in reports.values())
    ok = bool(integrity["ok"]) and rejected == 0
    _emit({"dataset_version": version, "feed": manifest.get("feed"), "is_synthetic": manifest.get("is_synthetic"),
           "ok": ok, "integrity": integrity, "revalidation": reports,
           "note": "stored rows were already accepted at write time; rejects here indicate corruption or policy drift"})
    return 0 if ok else 1


def cmd_ingest_alpaca(args: argparse.Namespace) -> int:
    from packages.core.clock import SystemClock
    from packages.core.secrets import SecretStore

    from .alpaca_provider import AlpacaMarketDataProvider, FeedEntitlementError, MarketDataUnavailable, MissingCredentials
    from .ingestion import IngestionService
    from .quality import FeedQualityMonitor
    from .store import MarketDataStore

    mode = FeedMode(args.feed_mode)
    if mode == FeedMode.SIMULATED:
        raise CliError("ingest-alpaca serves IEX_FREE or SIP_PAID; SIMULATED data comes from `fixture`")
    settings = load_settings()
    secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
    clock = SystemClock()
    try:
        provider = AlpacaMarketDataProvider(secrets, mode, clock)
    except MissingCredentials as exc:
        raise CliError(str(exc)) from None
    store = MarketDataStore(_datasets_dir(args))
    monitor = FeedQualityMonitor(mode, provider.source_id, clock, symbols=tuple(s.upper() for s in args.symbols))
    db = _db(args)
    try:
        svc = IngestionService(provider, store, db, clock, monitor, symbols=args.symbols, timeframe=args.timeframe,
                               backfill_lookback=timedelta(days=args.days))
        try:
            result = asyncio.run(svc.backfill(reason="cli ingest-alpaca"))
        except (FeedEntitlementError, MarketDataUnavailable) as exc:
            raise CliError(str(exc)) from None
    finally:
        if db is not None:
            db.close()
    _emit({"feed_mode": mode.value, "feed": provider.feed.value, "limitations": list(FEED_MODE_LIMITATIONS[mode]),
           "real_alpaca_data": True, **result.as_dict()})
    return 0 if result.ok else 1


def cmd_universe(args: argparse.Namespace) -> int:
    from packages.contracts.policy import RiskPolicy

    from .store import MarketDataStore
    from .universe import DEFAULT_MIN_AVG_DOLLAR_VOLUME, build_universe_snapshot, persist_universe_snapshot

    store = MarketDataStore(_datasets_dir(args))
    feed = DataFeed(args.feed) if args.feed else _effective_feed()
    if args.asof:
        asof = _parse_ts(args.asof)
    else:
        m = store.latest_manifest(feed)
        if m is None or not m.get("end"):
            raise CliError(f"no {feed.value} dataset stored; pass --asof/--feed or run `fixture`")
        asof = _parse_ts(str(m["end"])) + timedelta(hours=1)
    policy = RiskPolicy()
    snap = build_universe_snapshot(
        store, policy, asof, candidates=args.candidates or None, feed=feed,
        min_avg_dollar_volume=Decimal(args.min_adv) if args.min_adv else DEFAULT_MIN_AVG_DOLLAR_VOLUME)
    db = _db(args)
    try:
        persist_universe_snapshot(snap, store=store, db=db)
    finally:
        if db is not None:
            db.close()
    _emit(snap.as_dict())
    return 0


def _effective_feed_mode() -> FeedMode:
    """Same rule as the API runtime: explicit LAT_FEED_MODE, else IEX_FREE with paper keys, else SIMULATED."""
    from packages.core.secrets import SecretStore

    settings = load_settings()
    if settings.feed_mode is not None:
        return settings.feed_mode
    secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
    return FeedMode.IEX_FREE if secrets.alpaca_paper_available() else FeedMode.SIMULATED


def _effective_feed() -> DataFeed:
    from .common import FEED_MODE_TO_FEED

    return FEED_MODE_TO_FEED[_effective_feed_mode()]


def cmd_status(args: argparse.Namespace) -> int:
    from packages.core.secrets import SecretName, SecretStore

    from .store import MarketDataStore

    settings = load_settings()
    store = MarketDataStore(_datasets_dir(args))
    secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
    paper = secrets.alpaca_paper_available()
    effective = settings.feed_mode or (FeedMode.IEX_FREE if paper else FeedMode.SIMULATED)
    _emit({
        "datasets_dir": str(store.datasets_dir),
        "datasets": [_summary(m) for m in store.list_manifests()],
        "active_series": store.active_series(),
        "tables": store.summary(),
        "credentials_present": {"alpaca_paper": paper, "anthropic": secrets.has(SecretName.ANTHROPIC_API_KEY),
                                "openai": secrets.has(SecretName.OPENAI_API_KEY)},
        "feed_mode_effective": effective.value,
        "feed_mode_limitations": {m.value: list(v) for m, v in FEED_MODE_LIMITATIONS.items()},
        "sip_note": "SIP_PAID requires the paid Alpaca Algo Trader Plus plan; it is never selected or purchased automatically",
    })
    return 0


# ------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m services.market_data.cli", description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--datasets-dir", help=f"default: {PROJECT_ROOT / 'datasets'}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fixture", help="create/reuse the deterministic SYNTHETIC fixture")
    f.add_argument("--symbols", nargs="+")
    f.add_argument("--sessions", type=int)
    f.add_argument("--seed", type=int)
    f.add_argument("--end-date", help="last session (YYYY-MM-DD); default fixed 2026-11-04 for reproducibility")
    f.add_argument("--inject-defects", action="store_true", help="validator-test dataset (excluded from default loads)")
    f.set_defaults(func=cmd_fixture)

    li = sub.add_parser("import-lean-sample", help="import LEAN's bundled sample equity data")
    li.add_argument("--lean-root", help="LEAN clone (default trading/lean/Lean)")
    li.add_argument("--symbols", nargs="+")
    li.add_argument("--resolutions", nargs="+", default=["daily", "minute"], choices=["daily", "minute"])
    li.add_argument("--no-quotes", action="store_true")
    li.add_argument("--no-db", action="store_true", help="do not write data_quality_events to the SQLite ledger")
    li.add_argument("--db", help="SQLite path (default: var/<namespace>/ledger.sqlite)")
    li.set_defaults(func=cmd_import_lean_sample)

    ex = sub.add_parser("export-lean", help="export a dataset to LEAN format")
    ex.add_argument("--dataset", help="dataset_version (default: latest SYNTHETIC dataset)")
    ex.add_argument("--out", required=True, help="LEAN data folder to write (e.g. trading/lean/data-local)")
    ex.add_argument("--symbols", nargs="+")
    ex.add_argument("--quotes", action=argparse.BooleanOptionalAction, default=True, help="also write minute quote bars")
    ex.add_argument("--no-metadata", action="store_true", help="do not copy market-hours/symbol-properties from LEAN")
    ex.set_defaults(func=cmd_export_lean)

    v = sub.add_parser("validate", help="integrity + re-validation of a stored dataset")
    v.add_argument("--dataset", help="dataset_version (default: latest non-defective dataset)")
    v.add_argument("--quotes", action=argparse.BooleanOptionalAction, default=True)
    v.add_argument("--record", action="store_true", help="write the findings to data_quality_events")
    v.add_argument("--db")
    v.set_defaults(func=cmd_validate)

    ia = sub.add_parser("ingest-alpaca", help="backfill real Alpaca bars (PAPER keys required)")
    ia.add_argument("--symbols", nargs="+", required=True)
    ia.add_argument("--feed-mode", default=FeedMode.IEX_FREE.value, choices=[FeedMode.IEX_FREE.value, FeedMode.SIP_PAID.value],
                    help="IEX_FREE (free, IEX only) or SIP_PAID (paid plan; never auto-selected)")
    ia.add_argument("--days", type=float, default=3.0, help="lookback when nothing is stored yet")
    ia.add_argument("--timeframe", default="1Min")
    ia.add_argument("--no-db", action="store_true")
    ia.add_argument("--db")
    ia.set_defaults(func=cmd_ingest_alpaca)

    u = sub.add_parser("universe", help="point-in-time eligibility snapshot")
    u.add_argument("--asof", help="ISO timestamp with timezone (default: end of the latest dataset + 1h)")
    u.add_argument("--candidates", nargs="+", help="default: risk policy allowlist")
    u.add_argument("--feed", choices=[f.value for f in DataFeed],
                   help="default: the effective feed (SYNTHETIC without Alpaca keys, IEX with them)")
    u.add_argument("--min-adv", help="minimum average daily dollar volume (decimal string)")
    u.add_argument("--no-db", action="store_true")
    u.add_argument("--db")
    u.set_defaults(func=cmd_universe)

    st = sub.add_parser("status", help="datasets, active series and feed configuration")
    st.set_defaults(func=cmd_status)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
