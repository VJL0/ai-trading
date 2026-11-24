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
from datetime import UTC, date, datetime
from pathlib import Path

from packages.core.config import Settings

from .paths import data_root


def _http(settings: Settings, per_minute: int = 150):  # type: ignore[no-untyped-def]
    """GET-only client; 150/min leaves headroom for the running paper API on the same 200/min account limit."""
    from services.capabilities.http import ReadOnlyAlpacaHttp

    return ReadOnlyAlpacaHttp.from_secrets(settings.project_root, per_minute=per_minute)


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

    lab = Lab(settings.project_root)
    want = set(args.families.split(",")) if args.families else None
    summaries = []
    if args.asset in ("equity", "all"):
        panel, fp = lab.load_panel("equity_")
        dev = lab.dev_slice(panel)
        fams = [f for f in lab.equity_families(dev) if want is None or f.family in want]
        if want is None or "G_ml_rank" in want:
            fams.append(lab.ml_family(dev))
        if fams:
            summaries.append(lab.run_research(fams, dev, fp))
    if args.asset in ("orb", "all"):
        panel, fp = lab.load_panel("equity_")
        fam = lab.orb_family(panel)
        if want is None or fam.family in want:
            summaries.append(lab.run_research([fam], lab.dev_slice(panel), fp + ":orb"))
    if args.asset in ("crypto", "all"):
        panel, fp = lab.load_panel("crypto_1D_")
        dev = lab.dev_slice(panel)
        fams = [f for f in lab.crypto_families(dev) if want is None or f.family in want]
        if fams:
            summaries.append(lab.run_research(fams, dev, fp))
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
    import numpy as np

    from .backtest import BacktestConfig, apply_rebalance_schedule, hold_schedule
    from .factors import FAMILY, evaluate_expression, validate_spec
    from .pipeline import Lab
    from .research import FamilySpec, clean_json, evaluate_family
    from .strategies import momentum as A

    spec = validate_spec(json.loads(args.spec))
    lab = Lab(settings.project_root)
    panel, fp = lab.load_panel("equity_")
    dev = lab.dev_slice(panel)
    cache = A.ScoreCache(dev)

    def targets(p: dict) -> np.ndarray:  # type: ignore[type-arg]
        uni = cache.universe(p["universe"])
        sc = evaluate_expression(p["expression"], dev, uni)
        w = A.top_n_weights(sc, uni, int(p["n"])) * cache.regime(p["regime"])[:, None]
        return apply_rebalance_schedule(w, hold_schedule(len(dev.dates), int(p["rebalance"])))

    fam = FamilySpec(FAMILY, "us_equity", f"agent factor ({args.actor}): {spec['expression']}", [spec], targets,
                     "SPY buy-and-hold", A.buy_and_hold(dev, "SPY"), BacktestConfig(),
                     execution_notes=["agent-proposed; same gates as every family"])
    rep = evaluate_family(fam, dev, lab.ledger, fingerprint=fp,
                          out_dir=lab.out / "agent_experiments" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    wf = rep.get("walk_forward", {}).get("metrics", {})
    print(json.dumps(clean_json({"family": FAMILY, "passed_dev_gate": rep.get("passed"),
                                 "fail_reasons": rep.get("fail_reasons"), "wf_oos_sharpe": wf.get("sharpe"),
                                 "wf_oos_cagr": wf.get("cagr"), "wf_max_drawdown": wf.get("max_drawdown"),
                                 "dsr": rep.get("dsr", {}).get("dsr"), "n_trials": rep.get("dsr", {}).get("n_trials"),
                                 "alpha_t": rep.get("alpha_vs_benchmark", {}).get("alpha_t"),
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
    if args.family.startswith("D"):
        full, _ = lab.load_panel("crypto_1D_")
        spec = next(f for f in lab.crypto_families(full) if f.family == args.family)
    elif args.family.startswith("B_"):
        full, _ = lab.load_panel("equity_")
        spec = lab.orb_family(full, dev_only=False)
    else:
        full, _ = lab.load_panel("equity_")
        spec = next(f for f in lab.equity_families(full) if f.family == args.family)
    params = dev_report["selected_params"]
    try:
        res = lab.evaluate_holdout(spec, full, params, dev_report)
    except HoldoutLookRefused as exc:
        print(json.dumps({"refused": str(exc)}))
        return 3
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
    from .report import render

    text = render(settings.project_root)
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
    p.add_argument("--what", choices=("equity", "crypto", "all"), default="all")
    sub.add_parser("crypto-spreads")
    p = sub.add_parser("orb-fetch")
    p.add_argument("--start", default="2024-11-20")
    p.add_argument("--end", default="2026-11-20")
    p.add_argument("--top", type=int, default=40)
    p = sub.add_parser("research")
    p.add_argument("--asset", choices=("equity", "crypto", "orb", "all"), default="all")
    p.add_argument("--families", default=None)
    sub.add_parser("scan")
    sub.add_parser("shadow")
    sub.add_parser("timing")
    sub.add_parser("orb-prospective")
    p = sub.add_parser("report")
    p.add_argument("--out", default=None)
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
                "crypto-scan": cmd_crypto_scan}
    try:
        return handlers[args.cmd](args, settings)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


_ = Path  # re-exported for type checkers of optional commands
