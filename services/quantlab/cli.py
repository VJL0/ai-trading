"""quantlab command line. Data commands call free Alpaca market-data endpoints (GET only); nothing trades.

  python -m services.quantlab.cli fetch-equities          # daily SIP raw+adjusted, active + delisted symbols
  python -m services.quantlab.cli fetch-crypto            # hourly + daily bars, tradable USD pairs
  python -m services.quantlab.cli build-panel             # point-in-time equity panel (npz) + universe masks
  python -m services.quantlab.cli research [--families A,C,D]   # development walk-forward + trial ledger
  python -m services.quantlab.cli holdout --candidate ID  # one-use protected holdout evaluation
  python -m services.quantlab.cli scan                    # point-in-time opportunity scan (latest session)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.core.config import Settings

from .paths import data_root


def _http(settings: Settings, per_minute: int = 150):  # type: ignore[no-untyped-def]
    """GET-only client; 150/min leaves headroom for the running paper API on the same 200/min account limit.
    ``LAT_QUANTLAB_PER_MINUTE`` can only LOWER the rate (the in-API daily job sets it)."""
    from services.capabilities.http import ReadOnlyAlpacaHttp

    per_minute = max(1, min(per_minute, settings.quantlab_per_minute))
    return ReadOnlyAlpacaHttp.from_secrets(settings.project_root, use_keychain=settings.use_keychain, per_minute=per_minute)


def _files_digest(paths: list[Path]) -> str:
    import hashlib

    h = hashlib.sha256()
    for f in sorted(paths):
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()[:12]


def cmd_fetch_equities(args: argparse.Namespace, settings: Settings) -> int:
    from services.capabilities.probe import load_assets

    from .data import discover_delisted, fetch_equity_daily, last_completed_session, research_equity_symbols

    root = data_root(settings.project_root)
    assets = load_assets(settings.datasets_dir)
    end = date.fromisoformat(args.end) if args.end else last_completed_session()
    with _http(settings) as http:
        dl_path = root / "reference" / "delisted.parquet"
        if dl_path.is_file() and not args.refresh_delisted:
            import pandas as pd

            delisted = pd.read_parquet(dl_path)
        else:
            delisted = discover_delisted(http, date(2016, 1, 1), end)
            dl_path.parent.mkdir(parents=True, exist_ok=True)
            delisted.to_parquet(dl_path, index=False)
        symbols = research_equity_symbols(assets, delisted)
        print(json.dumps({"symbols": len(symbols), "delisted_events": len(delisted), "end": end.isoformat()}),
              flush=True)
        for adj in args.adjustments.split(","):
            out = root / "equity_daily" / f"sip_{adj}_{end.strftime('%Y%m%d')}"
            syms = symbols
            if adj != "raw":
                # adjusted bars only for symbols that ever pass a loose PIT liquidity screen in the raw data
                syms = liquid_ever(root / "equity_daily" / f"sip_raw_{end.strftime('%Y%m%d')}")
                print(json.dumps({"adjustment": adj, "liquid_ever_symbols": len(syms)}), flush=True)
            man = fetch_equity_daily(http, syms, out, adjustment=adj, end=end, workers=args.workers,
                                     progress=lambda d, n, a=adj: print(f"{a} {d}/{n}", flush=True) if d % 20 == 0 else None)
            rows = sum(b["rows"] for b in man.data["batches"].values())
            print(json.dumps({"adjustment": adj, "dir": str(out), "rows": rows, "requests": http.requests}),
                  flush=True)
    return 0


def liquid_ever(raw_dir: Path, *, min_price: float = 3.0, min_adv: float = 5e6) -> list[str]:
    """Symbols whose 20-session mean raw dollar volume reached ``min_adv`` on some day with close >= ``min_price``."""
    import numpy as np

    from .data import load_bars

    raw = load_bars(raw_dir)
    raw = raw.sort_values(["symbol", "t"])
    raw["dv"] = raw["c"] * raw["v"]
    adv = np.asarray(raw.groupby("symbol")["dv"].transform(lambda x: x.rolling(20, min_periods=10).mean()),
                     dtype=float)
    ok = (adv >= min_adv) & (raw["c"].to_numpy(dtype=float) >= min_price)
    return sorted(set(raw["symbol"].to_numpy()[ok].tolist()))


def cmd_fetch_exlisted(args: argparse.Namespace, settings: Settings) -> int:
    """Survivorship repair: SIP daily history for formerly listed companies now carried as OTC assets."""
    import pandas as pd

    from services.capabilities.probe import load_assets

    from .data import exlisted_candidates, fetch_equity_daily, last_completed_session, load_bars

    root = data_root(settings.project_root)
    raw_base = sorted((root / "equity_daily").glob("sip_raw_2*"))[-1]
    stamp = raw_base.name.rsplit("_", 1)[1]
    end = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    known = set(load_bars(raw_base)["symbol"].unique())
    dl_path = root / "reference" / "delisted.parquet"
    delisted = pd.read_parquet(dl_path) if dl_path.is_file() else None
    cands = exlisted_candidates(load_assets(settings.datasets_dir), known, delisted)
    print(json.dumps({"candidates": len(cands), "end": end.isoformat()}), flush=True)
    out_raw = root / "equity_daily" / f"exlisted_raw_{end:%Y%m%d}"
    with _http(settings) as http:
        fetch_equity_daily(http, cands, out_raw, adjustment="raw", end=end, workers=args.workers,
                           progress=lambda d, n: print(f"exlisted raw {d}/{n}", flush=True) if d % 20 == 0 else None)
        liquid = liquid_ever(out_raw) if any(out_raw.glob("*.parquet")) else []
        n_hist = len(set(load_bars(out_raw)["symbol"].tolist())) if any(out_raw.glob("*.parquet")) else 0
        print(json.dumps({"with_sip_history": n_hist, "liquid_ever": len(liquid)}), flush=True)
        if liquid:
            fetch_equity_daily(http, liquid, root / "equity_daily" / f"exlisted_all_{end:%Y%m%d}", adjustment="all",
                               end=end, workers=args.workers)
    _ = last_completed_session
    return 0


def cmd_fetch_delta(args: argparse.Namespace, settings: Settings) -> int:
    """Recent daily bars (raw + adjusted) for the panel's symbols, overlapping the last 10 sessions."""
    from services.market_data.calendar import get_calendar

    from .data import fetch_equity_daily, last_completed_session
    from .pipeline import Lab

    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("equity_")
    cal = get_calendar()
    last = date.fromisoformat(str(panel.dates[-1])[:10])
    start = cal.sessions_ending(last, 10)[0]
    end = last_completed_session()
    if end <= last:
        print(json.dumps({"up_to_date": str(last)}))
        return 0
    syms = sorted(str(s) for s in panel.symbols)
    out = data_root(settings.project_root) / "equity_daily" / f"delta_{end:%Y%m%d}"
    with _http(settings) as http:
        for adj in ("raw", "all"):
            fetch_equity_daily(http, syms, out / adj, adjustment=adj, start=start, end=end, workers=args.workers)
    print(json.dumps({"delta": str(out), "start": str(start), "end": str(end), "requests": http.requests}))
    return 0


def cmd_fetch_crypto(args: argparse.Namespace, settings: Settings) -> int:
    from services.capabilities.probe import load_assets

    from .data import crypto_usd_pairs, fetch_crypto_bars

    root = data_root(settings.project_root)
    pairs = crypto_usd_pairs(load_assets(settings.datasets_dir))
    with _http(settings) as http:
        for tf in args.timeframes.split(","):
            end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            out = root / "crypto" / f"{tf}_{end.strftime('%Y%m%d')}"
            man = fetch_crypto_bars(http, pairs, out, timeframe=tf, end=end)
            rows = sum(b["rows"] for b in man.data["batches"].values())
            print(json.dumps({"timeframe": tf, "pairs": len(pairs), "rows": rows, "dir": str(out)}), flush=True)
    return 0


def cmd_build_panel(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import Lab

    lab = Lab(settings.project_root)
    out: dict[str, str] = {}
    if args.what in ("equity", "all"):
        out["equity"] = str(lab.build_equity_panel())
    if args.what in ("crypto", "all"):
        out.update({f"crypto_{k}": str(v) for k, v in lab.build_crypto_panels().items()})
    if args.what == "session" or (args.what == "all" and any(lab.data.glob("intraday/session5m_all_*"))):
        out["session"] = str(lab.build_session_panel())
    print(json.dumps(out, indent=1))
    return 0


def cmd_crypto_spreads(args: argparse.Namespace, settings: Settings) -> int:
    from services.capabilities.http import DATA_BASE
    from services.capabilities.probe import load_assets

    from .data import crypto_usd_pairs

    pairs = crypto_usd_pairs(load_assets(settings.datasets_dir))
    spreads: dict[str, float] = {}
    with _http(settings) as http:
        for i in range(0, len(pairs), 20):
            data = http.get(f"{DATA_BASE}/v1beta3/crypto/us/latest/orderbooks", {"symbols": ",".join(pairs[i:i + 20])})
            for sym, ob in (data.get("orderbooks") or {}).items():
                if ob.get("b") and ob.get("a"):
                    bid, ask = float(ob["b"][0]["p"]), float(ob["a"][0]["p"])
                    if bid > 0 and ask >= bid:
                        spreads[sym] = (ask - bid) / ((ask + bid) / 2) * 1e4
    path = data_root(settings.project_root) / "reference" / "crypto_spreads.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"asof": datetime.now(UTC).isoformat(), "spread_bps": spreads,
                                "note": "single top-of-book snapshot; research doubles it (min 5 bps)"}, indent=1))
    print(json.dumps({"pairs": len(spreads), "median_bps": sorted(spreads.values())[len(spreads) // 2]
                      if spreads else None}))
    return 0


def cmd_research(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import Lab

    want = set(args.families.split(",")) if args.families else None
    if args.asset == "events":
        from .research import clean_json
        from .strategies.events_family import FAMILY, run_from_project

        if want is None or FAMILY in want:
            try:
                summary = run_from_project(settings.project_root, force=args.force)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(json.dumps(clean_json(summary), allow_nan=False))
        return 0
    lab = Lab(settings.project_root)
    summaries = []
    if args.asset in ("equity", "all"):
        panel, fp = lab.load_panel("equity_")
        dev = lab.dev_slice(panel)
        fams = [f for f in lab.equity_families(dev) if want is None or f.family in want]
        if want is None or "G_ml_rank" in want:
            fams.append(lab.ml_family(dev))
        if fams:
            summaries.append(lab.run_research(fams, dev, fp, force=args.force))
    if args.asset in ("xsrank", "all") and (want is None or "K_xs_rank" in want):
        panel, fp = lab.load_panel("equity_")
        dev = lab.dev_slice(panel)
        xs = lab.xs_rank_lab(dev)
        summary = lab.run_research([lab.xs_rank_family(dev, xs)], dev, fp, force=args.force)
        summaries.append(summary)
        if not args.no_diagnostics:
            _write_xs_rank_diagnostics(lab, xs, summary.get("run_id"))
    if args.asset in ("orb", "all"):
        panel, fp = lab.load_panel("equity_")
        fam = lab.orb_family(panel)
        if want is None or fam.family in want:
            orb_inputs = [p for d in sorted(lab.data.glob("intraday/orb_*"))
                          for p in (d / "manifest.json", d / "news_counts.parquet")]
            summaries.append(lab.run_research([fam], lab.dev_slice(panel), f"{fp}:orb:{_files_digest(orb_inputs)}",
                                              force=args.force))
    if args.asset in ("session", "all"):
        ip, sfp = None, ""
        try:
            ip, sfp = lab.load_session_panel()
        except FileNotFoundError:
            if args.asset == "session":
                raise
        if ip is not None:
            panel, fp = lab.load_panel("equity_")
            from .strategies.session import FAMILIES as SESSION_FAMILIES

            chosen = tuple(f for f in SESSION_FAMILIES if want is None or f in want)
            if chosen:
                summaries.append(lab.run_research(lab.session_families(ip, lab.dev_slice(panel), families=chosen),
                                                  lab.dev_slice(panel), f"{sfp}|daily:{fp}", force=args.force))
    if args.asset in ("crypto", "all"):
        panel, fp = lab.load_panel("crypto_1D_")
        dev = lab.dev_slice(panel)
        fams = [f for f in lab.crypto_families(dev) if want is None or f.family in want]
        if fams:
            spreads = _files_digest([lab.data / "reference" / "crypto_spreads.json"])
            summaries.append(lab.run_research(fams, dev, f"{fp}:spreads:{spreads}", force=args.force))
    if args.asset in ("crypto", "all"):
        try:
            eq, _ = lab.load_panel("equity_")
            cmp_ = lab.crypto_vs_etf(eq, lab.load_panel("crypto_1D_")[0])
            (lab.out / "crypto_vs_etf.json").write_text(json.dumps(cmp_, indent=1))
            print(json.dumps({"crypto_vs_etf": {k: (v.get("strategy", {}).get("sharpe") if isinstance(v, dict) else v)
                                                for k, v in cmp_.items()}}, default=str))
        except FileNotFoundError:
            pass
    for s in summaries:
        for f in s["families"]:
            print(json.dumps({k: f[k] for k in ("family", "passed_dev_gate", "wf_oos", "dsr", "spa_p_vs_benchmark",
                                                "alpha_t", "pbo", "fail_reasons")}, default=str))
    return 0


def _write_xs_rank_diagnostics(lab: Any, xs: Any, run_id: str | None) -> dict[str, Any]:
    """Family K's descriptive evidence (rank IC, deciles, calibration/coverage, construction and benchmark
    comparisons) from the development panel; records no trials."""
    from .research import clean_json
    from .strategies import xs_rank as K

    diag = clean_json(K.diagnostics(xs))
    diag["run_id"] = run_id
    text = json.dumps(diag, indent=1, allow_nan=False)
    (lab.out / "xs_rank_diagnostics.json").write_text(text)
    if run_id and (lab.out / "runs" / run_id).is_dir():
        (lab.out / "runs" / run_id / f"{K.FAMILY}_diagnostics.json").write_text(text)
    for key, row in diag.get("per_model_horizon", {}).items():
        print(json.dumps({"k": key, "ic_mean": row.get("rank_ic", {}).get("mean"), "ic_t": row.get("rank_ic", {}).get(
            "t"), "top_minus_cost_bps": row.get("top_decile_minus_cost_bps")}, default=str))
    return diag


def cmd_xsrank_diagnostics(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import Lab

    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("equity_")
    latest = lab.out / "latest_research.json"
    run_id = None
    if latest.is_file():
        run_id = json.loads(latest.read_text()).get("families", {}).get("K_xs_rank", {}).get("run_id")
    _write_xs_rank_diagnostics(lab, lab.xs_rank_lab(lab.dev_slice(panel)), run_id)
    return 0


def cmd_scan(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import Lab, crypto_opportunities, equity_opportunities, option_opportunities, run_selection
    from .research import clean_json
    from .scanner import scan

    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("equity_")
    result = clean_json(scan(panel))
    (lab.out / "latest_scan.json").write_text(json.dumps(result, indent=1, default=str))
    opps = equity_opportunities(panel, lab)
    with contextlib.suppress(FileNotFoundError):
        opps += crypto_opportunities(lab.load_panel("crypto_1D_")[0], lab)
    opps += option_opportunities(lab)
    sel = run_selection(opps, lab.out)
    print(json.dumps({"asof": result["asof"], "regime": result["regime"], "universe": result["universe_size"],
                      "selection": sel["action"], "reason": sel["reason"], "candidates": sel["counts"]}, default=str))
    return 0


def cmd_experiment(args: argparse.Namespace, settings: Settings) -> int:
    """Agent factor spec -> development-only family evaluation (trial recorded under F_agent)."""
    from .factors import FAMILY, EventDataUnavailable, agent_family, prepare_agent_events, validate_spec
    from .ledger import duplicate_on_same_evidence, methodology_hash
    from .pipeline import Lab
    from .research import clean_json, evaluate_family

    spec = validate_spec(json.loads(args.spec))
    lab = Lab(settings.project_root)
    panel, fp = lab.load_panel("equity_")
    dev = lab.dev_slice(panel)
    try:
        dev, dev_fp = prepare_agent_events(dev, settings.project_root, spec["expression"])
    except EventDataUnavailable as exc:
        print(json.dumps({"refused": True, "reason": "DATA_UNAVAILABLE", "detail": str(exc)}))
        return 3
    meth = methodology_hash()
    prior = lab.ledger.prior_outcomes(FAMILY, spec, canonicalize=validate_spec)
    dup = duplicate_on_same_evidence(prior, fingerprint=fp, dev_fingerprint=dev_fp, methodology=meth)
    if dup is not None:  # hypothesis memory: never re-test the same spec on the same evidence
        print(json.dumps({"refused": True, "reason": "DUPLICATE_HYPOTHESIS",
                          "detail": "this exact spec was already evaluated on the same development data and "
                                    "methodology; a re-test needs new evidence (corrected or extended development data "
                                    "or a methodology fix)",
                          "prior": {k: dup.get(k) for k in ("at", "status", "note")}}))
        return 3
    from .novelty import novelty_report, prior_agent_expressions
    from .strategies.momentum import universe_mask

    novelty = novelty_report(spec["expression"], dev, universe_mask(dev, spec["universe"]),
                             prior_agent_expressions(lab.ledger.rows(FAMILY, 500), spec["expression"], validate_spec))
    if novelty["redundant"]:  # originality screen: an existing signal in disguise is not a new hypothesis
        print(json.dumps(clean_json({"refused": True, "reason": "REDUNDANT_HYPOTHESIS",
                                     "detail": f"mean rank correlation {novelty['nearest_corr']} with "
                                               f"{novelty['nearest']} reaches |{novelty['threshold']}|; submit a "
                                               "factor that adds information beyond existing signals",
                                     "prior": {"nearest": novelty["nearest"], "max_abs_corr": novelty["max_abs_corr"]},
                                     "novelty": novelty})))
        return 3
    fam = agent_family(dev, spec, args.actor)
    rep = evaluate_family(fam, dev, lab.ledger, fingerprint=f"{fp}|dev:{dev_fp}|m:{meth}",
                          out_dir=lab.out / "agent_experiments" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    wf = rep.get("walk_forward", {}).get("metrics", {})
    print(json.dumps(clean_json({"family": FAMILY, "passed_dev_gate": rep.get("passed"),
                                 "fail_reasons": rep.get("fail_reasons"), "wf_oos_sharpe": wf.get("sharpe"),
                                 "wf_oos_cagr": wf.get("cagr"), "wf_max_drawdown": wf.get("max_drawdown"),
                                 "dsr": rep.get("dsr", {}).get("dsr"), "n_trials": rep.get("dsr", {}).get("n_trials"),
                                 "alpha_t": rep.get("alpha_vs_benchmark", {}).get("alpha_t"),
                                 "prior_outcomes": [{k: r.get(k) for k in ("at", "status", "note")} for r in prior[:3]],
                                 "novelty": {k: novelty[k] for k in ("max_abs_corr", "nearest", "nearest_corr",
                                                                     "complexity_nodes", "compared")},
                                 "note": "development-only; the protected holdout was not used"})))
    return 0


OPTION_UNDERLYINGS = ("SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "META")


def cmd_options_scan(args: argparse.Namespace, settings: Settings) -> int:
    """Live (indicative, delayed) option-chain scan: IV vs HAR-RV forecast, term structure, skew, spreads."""
    import numpy as np

    from services.capabilities.http import DATA_BASE

    from .data import load_bars
    from .options import scan_underlying
    from .pipeline import Lab
    from .research import clean_json

    lab = Lab(settings.project_root)
    syms = args.underlyings.split(",")
    returns: dict[str, np.ndarray] = {}
    last_close: dict[str, float] = {}
    try:
        panel, _ = lab.load_panel("equity_")
        for u in syms:
            if u in set(panel.symbols):
                returns[u] = panel.f["ret_cc"][-1260:, panel.col(u)]
                last_close[u] = float(panel.f["close"][-1, panel.col(u)])
        rv_source = "adjusted panel"
    except FileNotFoundError:  # before the first panel build: raw closes (no split in the 2-year window checked)
        raw = load_bars(lab._latest_dir("equity_daily/sip_raw_2*"), symbols=set(syms))
        for u in syms:
            c = raw.loc[raw["symbol"] == u, "c"].to_numpy(dtype=float)[-505:]
            if len(c) > 30:
                returns[u] = c[1:] / c[:-1] - 1
                last_close[u] = float(c[-1])
        rv_source = "raw daily closes"
    syms = [u for u in syms if u in returns]
    out = []
    with _http(settings) as http:
        latest = http.get(f"{DATA_BASE}/v2/stocks/trades/latest", {"symbols": ",".join(syms), "feed": "iex"})
        for u in syms:
            spot = float((latest.get("trades") or {}).get(u, {}).get("p") or last_close[u])
            rets = returns[u]
            res = scan_underlying(http, u, spot, np.asarray(rets))
            res["spot_source"] = "IEX latest trade" if (latest.get("trades") or {}).get(u) else "last daily close"
            out.append(res)
    payload = clean_json({"asof": datetime.now(UTC).isoformat(), "feed": "indicative (delayed/derived)",
                          "realized_vol_source": rv_source,
                          "evidence": "RESEARCH_ONLY", "underlyings": out,
                          "note": "Options are disabled by the risk policy; indicative quotes are not executable."})
    (lab.out / "latest_options_scan.json").write_text(json.dumps(payload, indent=1, default=str))
    from .options import journal_candidates, settle_journal

    journal = lab.out / "options_journal.jsonl"
    added = journal_candidates(journal, payload)
    closes: dict[tuple[str, str], float] = {}
    try:
        panel, _ = lab.load_panel("equity_")
        dates = [str(d)[:10] for d in panel.dates]
        for u in syms:
            if u in set(panel.symbols):
                col = panel.f["close"][:, panel.col(u)]
                closes.update({(u, d): float(c) for d, c in zip(dates, col, strict=True) if np.isfinite(c)})
    except FileNotFoundError:
        pass
    print(json.dumps({"journal_added": added, "journal": settle_journal(journal, closes)}))
    for u in out:
        sfc = u.get("surface", {})
        print(json.dumps({"underlying": u["underlying"], "quotes": u.get("n_quotes"),
                          "atm_iv_30d": sfc.get("atm_iv_30d"), "rv_forecast": sfc.get("rv_forecast"),
                          "iv_rv": sfc.get("iv_rv_ratio"), "skew": sfc.get("skew_25d_30d"),
                          "atm_spread_pct": sfc.get("atm_spread_pct_30d")}, default=str))
    return 0


def cmd_crypto_scan(args: argparse.Namespace, settings: Settings) -> int:
    import numpy as np

    from .panel import compound, trailing_mean, trailing_std
    from .pipeline import Lab
    from .research import clean_json

    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("crypto_1D_")
    r = panel.f["ret_cc"]
    today = datetime.now(UTC).date().isoformat()
    complete = [i for i, d in enumerate(panel.dates) if str(d)[:10] < today]  # never use a still-forming day
    t = complete[-1]
    m30 = compound(r, 30, 24)[t]
    m90 = compound(r, 90, 72)[t]
    vol = trailing_std(r, 30, 24)[t] * np.sqrt(365)
    sma50 = trailing_mean(panel.f["close"], 50, 45)[t]
    adv = panel.f["adv20"][t]
    spreads = lab.crypto_spreads()
    rows = []
    for j, sym in enumerate(panel.symbols):
        if not np.isfinite(panel.f["close"][t, j]):
            continue
        rows.append({"pair": str(sym), "close": float(panel.f["close"][t, j]), "mom_30d": float(m30[j]),
                     "mom_90d": float(m90[j]), "vol_30d": float(vol[j]),
                     "quality_30d": float(m30[j] / (vol[j] * np.sqrt(30 / 365))) if vol[j] > 0 else None,
                     "above_sma50": bool(panel.f["close"][t, j] > sma50[j]) if np.isfinite(sma50[j]) else None,
                     "adv30_usd": float(np.nan_to_num(adv[j])), "spread_bps_used": spreads.get(str(sym))})
    rows.sort(key=lambda x: -(x["quality_30d"] or -99))
    b = next((x for x in rows if x["pair"] == "BTC/USD"), None)
    payload = clean_json({"asof": str(panel.dates[t])[:10], "regime": {
        "btc_above_sma50": b["above_sma50"] if b else None, "btc_mom_90d": b["mom_90d"] if b else None},
        "pairs": rows, "note": "conditions only; crypto execution is not implemented (research/shadow only)"})
    (lab.out / "latest_crypto_scan.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"asof": payload["asof"], "regime": payload["regime"],
                      "top": [x["pair"] for x in rows[:5]]}))
    return 0


def cmd_orb_fetch(args: argparse.Namespace, settings: Settings) -> int:
    """Point-in-time stocks-in-play plan from the daily panel, then 5-minute SIP bars for those symbol-days."""
    import numpy as np

    from .data import fetch_intraday_symbol_days
    from .pipeline import Lab
    from .strategies.orb import build_plan

    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("equity_")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    plan = build_plan(panel, start, end, plan_top=args.top)
    out = data_root(settings.project_root) / "intraday" / f"orb_{start:%Y%m%d}_{end:%Y%m%d}_top{args.top}"
    with _http(settings) as http:
        man = fetch_intraday_symbol_days(http, plan, out, timeframe="5Min", tag=f"orb_top{args.top}",
                                         progress=lambda d, n: print(f"orb {d}/{n}", flush=True) if d % 50 == 0
                                         else None)
        from .data import fetch_news_counts

        news = fetch_news_counts(http, plan, out / "news_counts.parquet")
    rows = sum(b["rows"] for b in man.data["batches"].values())
    print(json.dumps({"days": len(plan), "rows": rows, "dir": str(out), "requests": http.requests,
                      "news_symbol_days_with_news": int(np.count_nonzero(news["news"].to_numpy(dtype=float) > 0))}))
    return 0


def cmd_session_fetch(args: argparse.Namespace, settings: Settings) -> int:
    """Continuous 5-minute SIP bars (split/dividend adjusted) for the session-timing universe, development period
    only: the end date is clamped to the day before the protected holdout starts."""
    from . import gates
    from .data import fetch_intraday_range
    from .strategies.session import UNIVERSES

    start = date.fromisoformat(args.start)
    end = min(date.fromisoformat(args.end), gates.HOLDOUT_START - timedelta(days=1))
    symbols = sorted(set(args.symbols.split(",")) if args.symbols else set(UNIVERSES["allow24"]))
    out = data_root(settings.project_root) / "intraday" / f"session5m_all_{start:%Y%m%d}_{end:%Y%m%d}"
    with _http(settings) as http:
        man = fetch_intraday_range(http, symbols, out, start=start, end=end, timeframe="5Min", adjustment="all",
                                   progress=lambda d, n: print(f"session-fetch {d}/{n}", flush=True))
    rows = sum(b["rows"] for b in man.data["batches"].values())
    print(json.dumps({"symbols": len(symbols), "rows": rows, "dir": str(out), "complete": man.data.get("complete"),
                      "requests": http.requests}))
    return 0


def cmd_session_diagnostics(args: argparse.Namespace, settings: Settings) -> int:
    """Trade-level diagnostics of each session family's walk-forward-selected variant (already in the trial ledger;
    re-simulating it adds no trial and selects nothing)."""

    from .costs import EquityCosts
    from .intraday import DailyContext, SimConfig, leg_diagnostics, simulate
    from .pipeline import Lab
    from .research import clean_json
    from .strategies import session as S

    lab = Lab(settings.project_root)
    latest = json.loads((lab.out / "latest_research.json").read_text())
    ip, _ = lab.load_session_panel()
    panel, _ = lab.load_panel("equity_")
    ctx = DailyContext.align(ip, lab.dev_slice(panel))
    fe = S.SessionFeatures(ip, ctx)
    pos = S.month_positions(ip)
    out: dict[str, Any] = {"note": "development period; selected variants only; costs = runtime model (x1)"}
    for fam in S.FAMILIES:
        info = latest["families"].get(fam)
        if not info or not info.get("selected_variant"):
            continue
        prm = next(v for v in S.VARIANTS[fam]() if ",".join(f"{k}={x}" for k, x in sorted(v.items()))
                   == info["selected_variant"])
        n = S.max_names(prm)
        _res, outs = simulate(ip, ctx, S.legs_for(fam, fe, prm, pos),
                              SimConfig(weight=1.0 / n, max_concurrent=n, costs=EquityCosts()))
        out[fam] = {"variant": prm, **leg_diagnostics(outs)}
    (lab.out / "session_diagnostics.json").write_text(json.dumps(clean_json(out), indent=1))
    print(json.dumps(clean_json(out), default=str)[:4000])
    return 0


def cmd_spread_sample(args: argparse.Namespace, settings: Settings) -> int:
    """Observed SIP NBBO spreads for the session universe on evenly spaced development sessions vs the cost model."""

    import numpy as np
    import pandas as pd

    from services.market_data.calendar import get_calendar

    from . import gates
    from .costs import EquityCosts
    from .data import sample_quoted_spreads
    from .pipeline import Lab
    from .research import clean_json
    from .strategies.session import UNIVERSES

    cal = get_calendar()
    sessions = cal.sessions_in_range(date.fromisoformat(args.start), gates.HOLDOUT_START - timedelta(days=1))
    pick = [sessions[int(i)] for i in np.linspace(0, len(sessions) - 1, args.days)]
    syms = list(UNIVERSES["allow24"])
    with _http(settings) as http:
        df = sample_quoted_spreads(http, syms, pick)
    lab = Lab(settings.project_root)
    panel, _ = lab.load_panel("equity_")
    dev = lab.dev_slice(panel)
    model = EquityCosts()
    out: dict[str, Any] = {"sessions": [d.isoformat() for d in pick], "clocks": sorted(set(df["clock"])),
                           "note": "SIP NBBO, 2-second windows; model = runtime liquidity-bucket full spread at the "
                                   "sampled session's PIT ADV (the runtime itself uses the live quote when it has one)",
                           "symbols": {}}
    rows = pd.Index(dev.dates)
    for sym in syms:
        sub = df[df["symbol"] == sym]
        if sub.empty or sym not in set(dev.symbols.tolist()):
            continue
        j = dev.col(sym)
        idx = rows.get_indexer(pd.DatetimeIndex(pd.to_datetime(sub["day"])))
        adv = dev.f["adv20"][idx[idx >= 0], j]
        model_bps = float(np.median(model.full_spread_bps(adv))) if len(adv) else None
        vals = np.asarray(sub["median_spread_bps"], dtype=float)
        clocks = np.asarray(sub["clock"], dtype=str)
        by_clock = {str(c): float(np.median(vals[clocks == c])) for c in sorted(set(clocks.tolist()))}
        out["symbols"][sym] = {"observed_full_spread_bps": float(np.median(vals)),
                               "by_clock": by_clock, "model_full_spread_bps": model_bps,
                               "model_one_way_slippage_fixed_bps": model.slippage_fixed_bps}
    (lab.out / "spread_sample.json").write_text(json.dumps(clean_json(out), indent=1))
    print(json.dumps({s: (v["observed_full_spread_bps"], v["model_full_spread_bps"])
                      for s, v in out["symbols"].items()}, default=str))
    return 0


def cmd_session_profile(args: argparse.Namespace, settings: Settings) -> int:
    """Descriptive time-of-day return profile (development period; nothing is selected or traded)."""
    from .intraday import session_profile
    from .pipeline import Lab
    from .research import clean_json
    from .strategies.session import UNIVERSES

    lab = Lab(settings.project_root)
    ip, _fp = lab.load_session_panel()
    prof = clean_json(session_profile(ip, UNIVERSES["allow24"]))
    (lab.out / "session_profile.json").write_text(json.dumps(prof, indent=1))
    print(json.dumps({s: {k: v.get("mean_bps") for k, v in w.items()} for s, w in prof.items()
                      if s in ("SPY", "QQQ", "IWM")}, default=str)[:3000])
    return 0


def cmd_holdout(args: argparse.Namespace, settings: Settings) -> int:
    """ONE look at the protected holdout for a family's development-selected spec (refused unless DEV_PASSED)."""
    from .ledger import HoldoutLookRefused
    from .pipeline import Lab

    lab = Lab(settings.project_root)
    latest = json.loads((lab.out / "latest_research.json").read_text())
    info = latest["families"].get(args.family)
    if not info:
        print(f"unknown family {args.family}", file=sys.stderr)
        return 2
    dev_report = json.loads((lab.out / "runs" / info["run_id"] / f"{args.family}.json").read_text())
    from .strategies.session import FAMILIES as SESSION_FAMILIES

    if args.family in SESSION_FAMILIES:
        return _session_holdout(lab, args.family, dev_report, settings)
    if args.family.startswith("D"):
        full, _ = lab.load_panel("crypto_1D_")
        spec = next(f for f in lab.crypto_families(full) if f.family == args.family)
    elif args.family.startswith("B_"):
        full, _ = lab.load_panel("equity_")
        spec = lab.orb_family(full, dev_only=False)
    else:
        full, _ = lab.load_panel("equity_")
        spec = lab.equity_family_by_name(args.family, full)  # A..C3, J, G (ML) and K (learning-to-rank)
    params = dev_report["selected_params"]
    try:
        res = lab.evaluate_holdout(spec, full, params, dev_report)
    except HoldoutLookRefused as exc:
        print(json.dumps({"refused": str(exc)}))
        return 3
    print(json.dumps(res, default=str))
    return 0


def _session_holdout(lab: Any, family: str, dev_report: dict[str, Any], settings: Settings) -> int:
    """One holdout look for a DEV_PASSED session family.

    Everything that needs no holdout data is checked first. The look is then committed in the vault BEFORE any
    holdout-period bar is downloaded (an unresolved look of this same spec is resumed, never a new one). The download
    starts ``overlap_days`` calendar days before the holdout so the development bars can be re-based to the new
    download's split/dividend basis on overlapping bars (adjusted levels depend on the fetch date); a symbol without
    overlap refuses the evaluation and leaves the look open. Trailing features warm up on development sessions; only
    holdout sessions are scored."""
    import pandas as pd

    from . import gates
    from .data import fetch_intraday_range, load_bars, rebase_onto
    from .intraday import IntradayPanel
    from .ledger import HoldoutLookRefused
    from .strategies.session import UNIVERSES

    overlap_days = 14
    params = dev_report["selected_params"]
    dev_dirs = sorted(lab.data.glob("intraday/session5m_all_*"))
    if not dev_dirs:
        print(json.dumps({"refused": "no development session dataset"}))
        return 2
    full, _ = lab.load_panel("equity_")
    if pd.Timestamp(full.dates[-1]) < pd.Timestamp(gates.HOLDOUT_END):
        print(json.dumps({"refused": "daily panel does not cover the holdout; run fetch-delta/build-panel first"}))
        return 2
    symbols = list(UNIVERSES["allow24"])
    try:
        look = lab.claim_holdout_look(family, params, resume=True)
    except HoldoutLookRefused as exc:
        print(json.dumps({"refused": str(exc)}))
        return 3
    start = gates.HOLDOUT_START - timedelta(days=overlap_days)
    out = lab.data / "intraday" / f"session5m_holdout_{start:%Y%m%d}_{gates.HOLDOUT_END:%Y%m%d}"
    with _http(settings) as http:
        fetch_intraday_range(http, symbols, out, start=start, end=gates.HOLDOUT_END, timeframe="5Min",
                             adjustment="all")
    bars, missing = rebase_onto(load_bars(dev_dirs[-1]), load_bars(out), symbols)
    if missing:
        print(json.dumps({"refused": f"no overlapping bars to re-base {missing}; look {look} stays open (resumable)"}))
        return 4
    full_ip = IntradayPanel.build(bars, symbols=sorted(set(bars["symbol"].astype(str)) & set(symbols)))
    spec = lab.session_families(full_ip, full, families=(family,), holdout_look=look)[0]
    res = lab.evaluate_holdout(spec, full, params, dev_report, look=look)
    print(json.dumps(res, default=str))
    return 0


def cmd_shadow(args: argparse.Namespace, settings: Settings) -> int:
    from . import shadow
    from .pipeline import Lab
    from .research import clean_json

    lab = Lab(settings.project_root)
    res: dict[str, object] = {}
    panel, _ = lab.load_panel("equity_")
    res["equity"] = shadow.update(lab, panel)
    try:
        cpanel, _ = lab.load_panel("crypto_1D_")
        res["crypto"] = shadow.update(lab, cpanel, asset_class="crypto")
    except FileNotFoundError:
        pass
    (lab.out / "latest_shadow.json").write_text(json.dumps(clean_json(res), indent=1, default=str))
    print(json.dumps(clean_json(res), default=str)[:2000])
    return 0


def cmd_timing(args: argparse.Namespace, settings: Settings) -> int:
    """Diagnostic (not a selection trial): each daily family's selected variant executed at the next OPEN vs the
    next CLOSE (MOC-style), development period only."""
    from dataclasses import replace

    from .backtest import run_weights
    from .pipeline import Lab
    from .research import clean_json

    lab = Lab(settings.project_root)
    latest = json.loads((lab.out / "latest_research.json").read_text())
    panel, fp = lab.load_panel("equity_")
    dev = lab.dev_slice(panel)
    specs = {f.family: f for f in lab.equity_families(dev)}
    specs["G_ml_rank"] = lab.ml_family(dev)
    out: dict[str, object] = {}
    for fam, spec in specs.items():
        info = latest["families"].get(fam)
        if not info:
            continue
        params = json.loads((lab.out / "runs" / info["run_id"] / f"{fam}.json").read_text())["selected_params"]
        tg = spec.make_targets(params)
        row: dict[str, object] = {}
        for ex in ("open", "close"):
            m = run_weights(dev.f, dev.dates, tg, replace(spec.config, execution=ex)).metrics()
            row[ex] = {k: m.get(k) for k in ("sharpe", "cagr", "max_drawdown", "cost_drag_annual")}
            lab.ledger.record(family=fam, variant=f"TIMING:{ex}", spec=params, data_fingerprint=fp, stage="DIAG",
                              metrics=m, status="OK", note="execution-timing diagnostic, not a selection trial")
        out[fam] = row
        print(fam, json.dumps(clean_json(row)))
    (lab.out / "timing.json").write_text(json.dumps(clean_json(out), indent=1))
    return 0


def cmd_orb_prospective(args: argparse.Namespace, settings: Settings) -> int:
    """Forward-only evaluation of the pre-registered ORB specs (with and without pre-open news confirmation)."""
    import numpy as np
    import pandas as pd

    from services.market_data.calendar import get_calendar

    from . import gates
    from .data import fetch_intraday_symbol_days, fetch_news_counts, load_bars
    from .pipeline import Lab
    from .prospective import DEFAULT_ORB_SPECS, ProspectiveJournal
    from .research import clean_json
    from .strategies.orb import FAMILY, OrbSimulator, build_plan

    lab = Lab(settings.project_root)
    journal = ProspectiveJournal(lab.out / "prospective.sqlite3")
    journal.register(FAMILY, DEFAULT_ORB_SPECS)
    specs = [s for s in journal.specs() if s["family"] == FAMILY]
    panel, _ = lab.load_panel("equity_")
    cal = get_calendar()
    todo: list[date] = []
    for d in panel.dates:
        day = date.fromisoformat(str(d)[:10])
        if day <= gates.HOLDOUT_END:
            continue
        if any(journal.eligible(s, cal.session_open(day)) and day.isoformat() not in journal.recorded(s["spec_hash"])
               for s in specs):
            todo.append(day)
    if not todo:
        print(json.dumps({"status": "nothing to record", "summary": clean_json(journal.summary())}, default=str))
        return 0
    plan = {k: v for k, v in build_plan(panel, todo[0], todo[-1]).items() if k in set(todo)}
    pdir = data_root(settings.project_root) / "intraday" / "prospective_orb"
    with _http(settings) as http:
        fetch_intraday_symbol_days(http, plan, pdir / "bars", timeframe="5Min", tag="orb_prospective")
        news = fetch_news_counts(http, plan, pdir / "news_counts.parquet")
    cut = pd.Timestamp(gates.HOLDOUT_START)
    dev_frames = [load_bars(d) for d in sorted(lab.data.glob("intraday/orb_*")) if any(d.glob("*.parquet"))]
    dev = pd.concat(dev_frames, ignore_index=True) if dev_frames else pd.DataFrame()
    if len(dev):  # the protected holdout window is never simulated: only pre-holdout days feed the normalizer
        dev = dev[pd.DatetimeIndex(pd.to_datetime(dev["t"], utc=True)).tz_convert("America/New_York").tz_localize(None)
                  < cut]
    bars = pd.DataFrame(pd.concat([dev, load_bars(pdir / "bars")], ignore_index=True))
    sim = OrbSimulator(bars, panel, news=news)
    wanted = {ts for ts in sim.days if date.fromisoformat(str(ts)[:10]) in set(todo)}
    recorded = []
    for s in specs:
        res = sim.run(s["spec"], trade_days=wanted)
        for i, d in enumerate(res.dates):
            day = date.fromisoformat(str(d)[:10])
            if journal.eligible(s, cal.session_open(day)) and journal.record(
                    s["spec_hash"], day, float(res.net[i]), float(res.gross[i]), int(np.round(res.n_positions[i]) * 2)):
                recorded.append({"spec": s["spec"].get("catalyst"), "session": day.isoformat(),
                                 "net": float(res.net[i])})
    summary = clean_json(journal.summary())
    (lab.out / "latest_prospective.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"recorded": recorded, "summary": summary}, default=str)[:3000])
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    from .report import render, render_session

    text = render_session(settings.project_root) if args.session else render(settings.project_root)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m services.quantlab.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-equities")
    p.add_argument("--adjustments", default="raw,all")
    p.add_argument("--end", default=None)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--refresh-delisted", action="store_true")
    p = sub.add_parser("fetch-exlisted")
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("fetch-delta")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("fetch-crypto")
    p.add_argument("--timeframes", default="1Hour,1Day")
    p = sub.add_parser("build-panel")
    p.add_argument("--what", choices=("equity", "crypto", "session", "all"), default="all")
    sub.add_parser("crypto-spreads")
    p = sub.add_parser("orb-fetch")
    p.add_argument("--start", default="2024-11-20")
    p.add_argument("--end", default="2026-11-20")
    p.add_argument("--top", type=int, default=40)
    p = sub.add_parser("research")
    p.add_argument("--asset", choices=("equity", "crypto", "orb", "session", "xsrank", "events", "all"), default="all")
    p.add_argument("--families", default=None)
    p.add_argument("--force", action="store_true",
                   help="re-evaluate families already evaluated on the same data and code (adds trials)")
    p.add_argument("--no-diagnostics", action="store_true", help="xsrank: skip the descriptive diagnostics file")
    sub.add_parser("xsrank-diagnostics", help="family K rank IC / calibration / construction diagnostics (no trials)")
    sub.add_parser("scan")
    sub.add_parser("shadow")
    sub.add_parser("timing")
    sub.add_parser("orb-prospective")
    p = sub.add_parser("report")
    p.add_argument("--out", default=None)
    p.add_argument("--session", action="store_true", help="session-timing families only")
    p = sub.add_parser("session-fetch")
    p.add_argument("--start", default="2016-01-04")
    p.add_argument("--end", default="2025-11-20")
    p.add_argument("--symbols", default=None)
    sub.add_parser("session-profile")
    sub.add_parser("session-diagnostics")
    p = sub.add_parser("spread-sample")
    p.add_argument("--start", default="2019-01-02")
    p.add_argument("--days", type=int, default=16)
    p = sub.add_parser("holdout")
    p.add_argument("--family", required=True)
    p = sub.add_parser("options-scan")
    p.add_argument("--underlyings", default=",".join(OPTION_UNDERLYINGS))
    sub.add_parser("crypto-scan")
    p = sub.add_parser("experiment")
    p.add_argument("--spec", required=True)
    p.add_argument("--actor", default="agent")
    args = ap.parse_args(argv)
    settings = Settings()
    handlers = {"fetch-equities": cmd_fetch_equities, "fetch-crypto": cmd_fetch_crypto,
                "build-panel": cmd_build_panel, "crypto-spreads": cmd_crypto_spreads, "research": cmd_research,
                "scan": cmd_scan, "experiment": cmd_experiment, "options-scan": cmd_options_scan,
                "orb-fetch": cmd_orb_fetch, "holdout": cmd_holdout, "fetch-delta": cmd_fetch_delta,
                "fetch-exlisted": cmd_fetch_exlisted, "shadow": cmd_shadow,
                "report": cmd_report, "timing": cmd_timing, "orb-prospective": cmd_orb_prospective,
                "crypto-scan": cmd_crypto_scan, "session-fetch": cmd_session_fetch,
                "session-profile": cmd_session_profile, "spread-sample": cmd_spread_sample,
                "session-diagnostics": cmd_session_diagnostics, "xsrank-diagnostics": cmd_xsrank_diagnostics}
    try:
        return handlers[args.cmd](args, settings)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


_ = Path  # re-exported for type checkers of optional commands
