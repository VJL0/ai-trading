"""End-to-end quantlab orchestration: panels -> development research -> FDR -> registry -> (one-use) holdout ->
scan -> cross-asset opportunity selection. Every result file is written under ``experiments/quantlab``.

The development panel is sliced to end strictly before ``gates.HOLDOUT_START``; only :func:`evaluate_holdout`
receives later rows, and only after the vault committed the look.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.contracts.policy import RiskPolicy

from . import gates
from .backtest import BacktestConfig, BacktestResult, performance_metrics, run_weights
from .costs import CryptoCosts, EquityCosts
from .data import load_bars
from .ledger import HoldoutLookRefused, HoldoutVault, TrialLedger, spec_hash
from .opportunity import Evidence, EvidenceStatus, Opportunity, select
from .panel import EquityPanel
from .paths import data_root, results_root
from .registry import State, StrategyRegistry, TransitionRefused
from .research import FamilySpec, clean_json, evaluate_family, family_fdr
from .strategies import crypto as D
from .strategies import momentum as A
from .strategies import reversion as C
from .targets import latest_decision

log = logging.getLogger("lat.quantlab")


def _append_new_symbols(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    have = set(base["symbol"].tolist())
    keep = np.array([s not in have for s in extra["symbol"].tolist()], dtype=bool)
    return pd.DataFrame(pd.concat([base, extra.loc[keep]], ignore_index=True))


class Lab:
    def __init__(self, project_root: Path | None = None) -> None:
        from packages.core.config import PROJECT_ROOT

        self.root = project_root or PROJECT_ROOT
        self.data = data_root(self.root)
        self.out = results_root(self.root)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = TrialLedger(self.out / "trials.sqlite3")
        self.vault = HoldoutVault(self.out / "holdout.sqlite3")
        self.registry = StrategyRegistry(self.out / "registry.sqlite3")
        self.holdout = self.vault.register(gates.HOLDOUT_NAME, gates.HOLDOUT_START, gates.HOLDOUT_END)

    # ------------------------------------------------------------------ panels
    def _latest_dir(self, pattern: str) -> Path:
        dirs = sorted(self.data.glob(pattern))
        if not dirs:
            raise FileNotFoundError(f"no dataset matching {pattern} under {self.data}")
        return dirs[-1]

    def build_equity_panel(self) -> Path:
        from services.capabilities.probe import load_assets

        from .data import merge_with_deltas

        raw_dir = self._latest_dir("equity_daily/sip_raw_2*")
        adj_dir = self._latest_dir("equity_daily/sip_all_2*")
        adj = load_bars(adj_dir)
        extra_adj = sorted(self.data.glob("equity_daily/exlisted_all_*"))
        if extra_adj:  # survivorship repair: formerly listed names now carried as OTC assets (SIVB, FRC, ...)
            adj = _append_new_symbols(adj, load_bars(extra_adj[-1]))
        syms = set(adj["symbol"].tolist())
        raw = load_bars(raw_dir, symbols=syms)
        extra_raw = sorted(self.data.glob("equity_daily/exlisted_raw_*"))
        if extra_raw:
            raw = _append_new_symbols(raw, load_bars(extra_raw[-1], symbols=syms))
        deltas = sorted((self.data / "equity_daily").glob("delta_*"))
        if deltas:
            raw = merge_with_deltas(raw, [load_bars(d / "raw", symbols=syms) for d in deltas], adjusted=False)
            adj = merge_with_deltas(adj, [load_bars(d / "all", symbols=syms) for d in deltas], adjusted=True)
        assets = load_assets(self.root / "datasets")
        if extra_adj:  # recovered names carry their CURRENT (often OTC) asset record: classify by name, not venue
            from services.capabilities.instruments import classify

            recovered = set(load_bars(extra_adj[-1])["symbol"].tolist())
            assets = assets.copy()
            mask = assets["symbol"].isin(sorted(recovered)).to_numpy() & (assets["asset_class"] == "us_equity").to_numpy()
            assets.loc[mask, "kind"] = [str(classify(sym, name, None)) for sym, name in
                                        zip(assets.loc[mask, "symbol"], assets.loc[mask, "name"], strict=True)]
        panel = EquityPanel.build(raw, adj, assets)
        panel.meta.update({"raw_dir": raw_dir.name, "adj_dir": adj_dir.name, "deltas": [d.name for d in deltas],
                           "exlisted": [d.name for d in extra_adj],
                           "raw_manifest": json.loads((raw_dir / "manifest.json").read_text())["params"],
                           "adj_manifest": json.loads((adj_dir / "manifest.json").read_text())["params"]})
        last = str(panel.dates[-1])[:10].replace("-", "")
        path = self.data / "panels" / f"equity_{adj_dir.name}_to{last}.npz"
        panel.save(path)
        return path

    def build_crypto_panels(self) -> dict[str, Path]:
        out = {}
        for tf, freq in (("1Hour", "1h"), ("1Day", "1D")):
            try:
                d = self._latest_dir(f"crypto/{tf}_*")
            except FileNotFoundError:
                continue
            bars = load_bars(d)
            panel = D.build_crypto_panel(bars, freq="1D" if freq == "1D" else "1h")
            panel.meta.update({"source_dir": d.name})
            path = self.data / "panels" / f"crypto_{freq}_{d.name}.npz"
            panel.save(path)
            out[freq] = path
        return out

    def load_panel(self, prefix: str) -> tuple[EquityPanel, str]:
        files = sorted((self.data / "panels").glob(f"{prefix}*.npz"))
        if not files:
            raise FileNotFoundError(f"no panel {prefix}*; run build-panel")
        p = EquityPanel.load(files[-1])
        return p, str(p.meta.get("sha256", files[-1].name))[:16]

    def dev_slice(self, panel: EquityPanel) -> EquityPanel:
        end = pd.Timestamp(self.holdout[0] - timedelta(days=1))
        return panel.slice_dates(None, end)  # pyright: ignore[reportArgumentType]

    # ------------------------------------------------------------------ family specs
    def equity_families(self, dev: EquityPanel) -> list[FamilySpec]:
        cache = A.ScoreCache(dev)
        bench = A.buy_and_hold(dev, "SPY")
        cfg = BacktestConfig(capital=100_000.0, execution="open")
        band_cfg = replace(cfg, rebalance_band=0.02)
        liquid = cache.universe("liquid1000")
        large = cache.universe("large300")
        notes_multi = ["multi-day holding requires active-policy and strategy authorization",
                       "current risk policy: 1 position, <= 25% of equity, allowlist only"]
        return [
            FamilySpec(A.FAMILY, "us_equity", "cross-sectional momentum (5 scores x regimes x breadth x rebalance)",
                       list(A.variants()), lambda p: A.targets(cache, p), "SPY buy-and-hold", bench, cfg,
                       execution_notes=notes_multi),
            FamilySpec(A.ROTATION_FAMILY, "etf", "allowlisted asset-class/sector ETF rotation (1-3 holdings)",
                       list(A.rotation_variants()), lambda p: A.rotation_targets(dev, p), "SPY buy-and-hold", bench,
                       cfg, execution_notes=["multi-day holding requires active-policy and strategy authorization"]),
            FamilySpec(C.FAMILY_REVERSAL, "us_equity", "weekly-loser reversal among large caps (regime-conditioned)",
                       list(C.reversal_variants()), lambda p: C.reversal_targets(dev, large, p), "SPY buy-and-hold",
                       bench, band_cfg, execution_notes=notes_multi),
            FamilySpec(C.FAMILY_PULLBACK, "us_equity", "RSI(2) pullback within an uptrend (liquid stocks)",
                       list(C.pullback_variants()), lambda p: C.pullback_targets(dev, liquid, p), "SPY buy-and-hold",
                       bench, band_cfg, execution_notes=notes_multi),
            FamilySpec(C.FAMILY_ETF_PULLBACK, "etf", "RSI(2) pullback on SPY/QQQ/IWM/DIA, one position",
                       list(C.etf_pullback_variants()), lambda p: C.etf_pullback_targets(dev, p), "SPY buy-and-hold",
                       bench, cfg, execution_notes=["multi-day holding requires active-policy and strategy authorization"]),
        ]

    def ml_family(self, dev: EquityPanel) -> FamilySpec:
        from .strategies import ml_rank as G

        cache = A.ScoreCache(dev)
        wf = G.WalkForwardScores(dev, cache)
        return FamilySpec(G.FAMILY, "us_equity", "walk-forward LightGBM vs ridge cross-sectional ranking (same "
                          "features, same folds, same portfolio rule)", list(G.variants()),
                          lambda p: G.targets(wf, cache, p), "SPY buy-and-hold", A.buy_and_hold(dev, "SPY"),
                          BacktestConfig(capital=100_000.0, execution="open"), first_test_year=2020,
                          execution_notes=["multi-day holding requires active-policy and strategy authorization"])

    def crypto_families(self, dev: EquityPanel) -> list[FamilySpec]:
        costs = CryptoCosts(spread_bps=self.crypto_spreads())
        flat = tuple(costs.one_way_bps(str(s)) for s in dev.symbols)
        cfg = BacktestConfig(capital=100_000.0, execution="open", flat_cost_bps=flat, periods_per_year=365.0,
                             delist_after=D.MAX_GAP_DAYS)
        bench = D.buy_and_hold(dev, ("BTC/USD",))
        notes = ["crypto execution is NOT_IMPLEMENTED in the risk engine/ledger (research and shadow only)"]
        return [
            FamilySpec(D.FAMILY_TSMOM, "crypto", "time-series momentum, volatility targeted (BTC or BTC/ETH/SOL)",
                       list(D.tsmom_variants()), lambda p: D.tsmom_targets(dev, p), "BTC buy-and-hold", bench, cfg,
                       first_test_year=2023, periods_per_year=365.0, execution_notes=notes),
            FamilySpec(D.FAMILY_XSMOM, "crypto", "cross-sectional momentum across liquid USD pairs",
                       list(D.xsmom_variants()), lambda p: D.xsmom_targets(dev, p), "BTC buy-and-hold", bench, cfg,
                       first_test_year=2023, periods_per_year=365.0, execution_notes=notes),
        ]

    def orb_family(self, panel: EquityPanel, *, dev_only: bool = True) -> FamilySpec:
        from .strategies import orb as B

        dirs = sorted(self.data.glob("intraday/orb_*"))
        if not dirs:
            raise FileNotFoundError("no ORB intraday data; run orb-fetch")
        bars = pd.concat([load_bars(d) for d in dirs if any(d.glob("*.parquet"))], ignore_index=True)
        news_parts = [pd.read_parquet(d / "news_counts.parquet") for d in dirs if (d / "news_counts.parquet").is_file()]
        news = pd.concat(news_parts, ignore_index=True) if news_parts else None
        sim = B.OrbSimulator(bars, panel, news=news)
        if dev_only:
            cut = pd.Timestamp(self.holdout[0])
            sim.days = [d for d in sim.days if d < cut]
        bench = B.spy_intraday_benchmark(sim)
        variants = list(B.variants())
        notes: list[str] = []
        dev_news = 0
        if news is not None and len(news):
            in_dev = news["session"].astype(str).to_numpy() < str(self.holdout[0])
            counts = news["news"].to_numpy(dtype=float)
            dev_news = int(np.count_nonzero(counts[in_dev] > 0))
        if dev_news == 0:
            variants = [v for v in variants if v.get("catalyst") != "news"]
            notes.append("news-catalyst variants excluded: Alpaca news history on this plan starts 2026-04 (inside the "
                         "holdout), so catalyst confirmation is prospective-only")
        return FamilySpec(B.FAMILY, "us_equity", "stocks-in-play 5-minute opening-range breakout (long-only)",
                          variants, lambda p: np.zeros(panel.shape), "SPY open-to-close", np.zeros(0),
                          first_test_year=int(sim.days[0].year) + 1 if sim.days else 2025,
                          wf_period="half", wf_min_train_days=250,
                          runner=lambda p, cm: sim.run(p, cm), benchmark_result=bench,
                          execution_notes=[*notes, "intraday, same-session exit: compatible with the current lifecycle",
                                           "current policy: 1 position, $2,000 max order, allowlist of 24 symbols",
                                           "stocks in play are mostly NOT on the allowlist"])

    def crypto_vs_etf(self, equity: EquityPanel, crypto: EquityPanel, *, lookback: int = 50) -> dict[str, Any]:
        """Same BTC trend rule (price > SMA(lookback), long or flat) on BTC/USD vs IBIT over the common
        development window; each instrument with its own cost model (and IBIT's 0.25% expense ratio)."""
        from .panel import trailing_mean

        end = pd.Timestamp(self.holdout[0] - timedelta(days=1))
        out: dict[str, Any] = {"rule": f"long when close > SMA({lookback}), else flat; daily decisions"}
        try:
            j_etf = equity.col("IBIT")
            j_btc = crypto.col("BTC/USD")
        except KeyError as exc:
            return {"status": "MISSING", "reason": str(exc)}
        start = equity.dates[np.where(np.isfinite(equity.f["close"][:, j_etf]))[0][0]] + pd.Timedelta(days=lookback * 2)
        for name, panel, j, cfg in (
            ("IBIT", equity, j_etf, BacktestConfig(capital=100_000.0, execution="open")),
            ("BTC/USD", crypto, j_btc, BacktestConfig(capital=100_000.0, execution="open", periods_per_year=365.0,
                                                      flat_cost_bps=tuple(CryptoCosts(spread_bps=self.crypto_spreads())
                                                                          .one_way_bps(str(s)) for s in crypto.symbols),
                                                      delist_after=D.MAX_GAP_DAYS)),
        ):
            sub = panel.slice_dates(None, end)  # pyright: ignore[reportArgumentType]
            px = sub.f["adj_close"][:, j]
            sma = trailing_mean(sub.f["adj_close"][:, [j]], lookback, int(lookback * 0.9))[:, 0]
            w = np.zeros(sub.shape)
            w[:, j] = np.where(np.isfinite(sma) & (px > sma), 1.0, 0.0)
            hold = np.r_[True, np.diff(w[:, j]) != 0]
            tgt = np.where(hold[:, None], w, np.nan)
            res = run_weights(sub.f, sub.dates, tgt, cfg)
            ser = res.series()
            idx = pd.DatetimeIndex(ser.index)
            keep = idx >= start
            r = np.asarray(ser.to_numpy(), dtype=float)[keep]
            idx = idx[keep]
            if name == "IBIT":  # expense ratio accrues while invested
                inv = w[1:, j][-len(r):] if len(r) else np.zeros(0)
                r = r - inv * 0.0025 / 252
            bh = np.nan_to_num(sub.f["ret_cc"][1:, j][-len(r):]) if len(r) else np.zeros(0)
            out[name] = {"strategy": performance_metrics(r, idx, periods_per_year=cfg.periods_per_year),
                         "buy_and_hold": performance_metrics(bh, periods_per_year=cfg.periods_per_year),
                         "trades": int(res.trades)}
        out["note"] = ("IBIT trades only in U.S. regular hours (weekend gaps realized at Monday's open); BTC/USD trades "
                       "24/7 with 25 bps taker fees. Both are long-only; neither is currently executable here (IBIT is"
                       " not on the allowlist; crypto has no execution path).")
        return clean_json(out)

    def crypto_spreads(self) -> dict[str, float]:
        path = self.data / "reference" / "crypto_spreads.json"
        if path.is_file():
            return {k: max(5.0, 2.0 * float(v)) for k, v in json.loads(path.read_text())["spread_bps"].items()}
        return {}

    # ------------------------------------------------------------------ execution tier
    def policy_tier(self, spec: FamilySpec, panel: EquityPanel, params: dict[str, Any],
                    policy: RiskPolicy | None = None, equity: float = 5000.0) -> dict[str, Any]:
        """Selected variant under TODAY's risk policy: allowlisted symbols only, at most ``max_positions`` names,
        each capped by concentration / position / order notional, whole shares, on the actual account size."""
        policy = policy or RiskPolicy()
        if spec.runner is not None or spec.asset_class == "crypto":
            return {"status": "NOT_APPLICABLE", "reason": "intraday/crypto families use their own execution model"}
        w = spec.make_targets(params)
        hold = np.isnan(w).all(axis=1)
        w = np.nan_to_num(w)
        allow = np.isin(panel.symbols, policy.symbol_allowlist)
        w = np.where(allow[None, :], w, 0.0)
        cap = min(float(policy.max_concentration_pct), float(policy.max_position_notional) / equity,
                  float(policy.max_order_notional) / equity)
        out = np.zeros_like(w)
        k = max(1, int(policy.max_positions))
        for t in range(len(w)):
            row = w[t]
            if row.max() <= 0:
                continue
            top = np.argsort(-row)[:k]
            top = top[row[top] > 0]
            out[t, top] = np.minimum(row[top] / row[top].sum() * min(1.0, cap * len(top)), cap)
        out[hold] = np.nan
        cfg = replace(spec.config, capital=equity, whole_shares=True)
        res = run_weights(panel.f, panel.dates, out, cfg)
        m = res.metrics(spec.periods_per_year)
        return {"status": "OK", "equity": equity, "position_cap": cap, "max_positions": k,
                "allowlist_size": len(policy.symbol_allowlist),
                "metrics": {key: m.get(key) for key in ("sharpe", "cagr", "max_drawdown", "avg_exposure", "trades",
                                                        "cost_drag_annual", "start", "end")},
                "note": "multi-day holding is still blocked by the intraday-only position lifecycle"}

    # ------------------------------------------------------------------ research
    def run_research(self, families: list[FamilySpec], panel: EquityPanel, fingerprint: str,
                     run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or datetime.now(UTC).strftime("ql_%Y%m%dT%H%M%SZ")
        out_dir = self.out / "runs" / run_id
        reports = []
        for spec in families:
            log.info("family %s: %d variants", spec.family, len(spec.variants))
            rep = evaluate_family(spec, panel, self.ledger, fingerprint=fingerprint, out_dir=out_dir)
            if rep.get("selected_params") is not None:
                try:
                    rep["policy_tier"] = self.policy_tier(spec, panel, rep["selected_params"])
                except Exception as exc:  # recorded, never hidden
                    rep["policy_tier"] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"[:300]}
                (out_dir / f"{spec.family}.json").write_text(json.dumps(clean_json(
                    {k: v for k, v in rep.items() if not k.startswith("_")}), indent=1, default=str))
            reports.append(rep)
            log.info("family %s passed=%s %s", spec.family, rep.get("passed"), rep.get("fail_reasons"))
        return self._summarize(run_id, out_dir, reports)

    def _summarize(self, run_id: str, out_dir: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
        fdr = family_fdr(reports)
        fams = []
        for r in reports:
            fam = r["family"]
            bh = fdr.get(fam, {})
            passed = bool(r.get("passed")) and bool(bh.get("reject", False))
            fails = list(r.get("fail_reasons", []))
            if r.get("passed") and not bh.get("reject", False):
                fails.append(f"does not survive Benjamini-Hochberg across families (q={bh.get('q_value')})")
            wf = r.get("walk_forward", {}).get("metrics", {})
            fams.append({
                "family": fam, "asset_class": r.get("asset_class"), "description": r.get("description"),
                "n_variants": r.get("n_variants"), "passed_dev_gate": passed, "fail_reasons": fails,
                "selected_variant": r.get("selected_variant"), "selected_spec_hash": r.get("selected_spec_hash"),
                "wf_oos": {k: wf.get(k) for k in ("sharpe", "cagr", "max_drawdown", "ann_vol", "sortino", "calmar",
                                                  "cvar95_daily", "start", "end", "pct_positive_years")},
                "benchmark": r.get("benchmark", {}).get("name"),
                "benchmark_wf_period": {k: r.get("walk_forward", {}).get("benchmark_same_period", {}).get(k)
                                        for k in ("sharpe", "cagr", "max_drawdown")},
                "dsr": r.get("dsr", {}).get("dsr"), "n_trials": r.get("dsr", {}).get("n_trials"),
                "spa_p_vs_benchmark": r.get("spa_vs_benchmark", {}).get("p_value"),
                "alpha_t": r.get("alpha_vs_benchmark", {}).get("alpha_t"),
                "alpha_annual": r.get("alpha_vs_benchmark", {}).get("alpha_annual"),
                "beta": r.get("alpha_vs_benchmark", {}).get("beta"),
                "pbo": r.get("pbo", {}).get("pbo"), "stress_sharpe_2x_costs": r.get("stress", {}).get("sharpe"),
                "turnover_annual": r.get("selected_full_dev", {}).get("turnover_annual"),
                "cost_drag_annual": r.get("selected_full_dev", {}).get("cost_drag_annual"),
                "trades": r.get("selected_trades"), "execution_notes": r.get("execution_notes"),
                "bh_q_value": bh.get("q_value"),
                "policy_tier": r.get("policy_tier"),
            })
            if passed and r.get("selected_params") is not None:
                sid = f"{fam}:{r['selected_spec_hash']}"
                self.registry.register(sid, family=fam, spec_hash=r["selected_spec_hash"],
                                       spec=r["selected_params"], asset_class=str(r.get("asset_class")),
                                       holding="intraday" if fam.startswith("B_") else "multi_day")
                if self.registry.state(sid) == State.RESEARCH:
                    self.registry.transition(sid, State.DEV_PASSED, reason=f"development gate passed in {run_id}",
                                             evidence={"dev_passed": True, "spec_hash": r["selected_spec_hash"],
                                                       "run_id": run_id})
        summary = clean_json({"run_id": run_id, "generated_at": datetime.now(UTC).isoformat(),
                              "holdout": [str(self.holdout[0]), str(self.holdout[1])], "gate": gates.DEV_GATE,
                              "families": fams, "no_edge_note": reports[0].get("no_edge_note") if reports else None})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        latest = self.out / "latest_research.json"
        merged: dict[str, Any] = json.loads(latest.read_text()) if latest.is_file() else {"families": {}}
        for f in fams:
            merged["families"][f["family"]] = {**f, "run_id": run_id}
        merged.update({"updated_at": datetime.now(UTC).isoformat(), "holdout": summary["holdout"],
                       "gate": gates.DEV_GATE, "no_edge_note": summary["no_edge_note"]})
        latest.write_text(json.dumps(clean_json(merged), indent=1))
        return summary

    # ------------------------------------------------------------------ holdout (one look)
    def evaluate_holdout(self, spec: FamilySpec, full: EquityPanel, params: dict[str, Any],
                         dev_report: dict[str, Any]) -> dict[str, Any]:
        sid = f"{spec.family}:{spec_hash(params)}"
        if self.registry.state(sid) != State.DEV_PASSED:
            raise HoldoutLookRefused(f"{sid} is not DEV_PASSED; holdout looks are reserved for gate-passing specs")
        look = self.vault.claim(gates.HOLDOUT_NAME, spec.family, params, sid)
        res = spec.run(full, params)
        s = res.series()
        h0, h1 = pd.Timestamp(self.holdout[0]), pd.Timestamp(self.holdout[1])
        ix = pd.DatetimeIndex(s.index)
        mask = (ix >= h0) & (ix <= h1)
        m = performance_metrics(np.asarray(s.to_numpy())[mask], ix[mask], periods_per_year=spec.periods_per_year)
        dev_mdd = float(dev_report.get("walk_forward", {}).get("metrics", {}).get("max_drawdown", -0.2))
        ok, fails = gates.holdout_verdict(m, dev_mdd)
        result = clean_json({"strategy_id": sid, "metrics": m, "passed": ok, "fail_reasons": fails,
                             "look_id": look, "holdout": [str(self.holdout[0]), str(self.holdout[1])]})
        self.vault.record_result(look, result)
        self.ledger.record(family=spec.family, variant="HOLDOUT", spec=params, data_fingerprint="holdout",
                           stage="HOLDOUT", metrics=m, status="PASS" if ok else "FAIL", note="; ".join(fails))
        if ok:
            self.registry.transition(sid, State.VALIDATED, reason="holdout passed",
                                     evidence={"holdout_passed": True, "spec_hash": spec_hash(params),
                                               "look_id": look, "metrics": m})
            self.registry.transition(sid, State.SHADOW, reason="validated: prospective shadow evaluation", evidence={})
        else:
            self.registry.transition(sid, State.REJECTED, reason="holdout failed: " + "; ".join(fails)[:400],
                                     evidence={"look_id": look, "metrics": m})
        (self.out / "holdout_results").mkdir(exist_ok=True)
        (self.out / "holdout_results" / f"{sid.replace(':', '_')}.json").write_text(json.dumps(result, indent=1))
        return result


# ------------------------------------------------------------------------------------------- opportunities
def latest_capabilities(root: Path) -> dict[str, Any]:
    from services.capabilities.probe import load_latest

    return load_latest(root / "datasets") or {}


def equity_opportunities(panel: EquityPanel, lab: Lab, *, policy: RiskPolicy | None = None) -> list[Opportunity]:
    """Current candidates from every equity family's selected spec, evidence-tagged from the registry/run."""
    from services.capabilities.policy import load_active_policy
    from services.market_data.calendar import get_calendar

    from .shadow import lifecycle_supported

    active = policy or load_active_policy(lab.root)
    policy = active or RiskPolicy()
    caps = latest_capabilities(lab.root).get("asset_classes", {})
    latest = json.loads((lab.out / "latest_research.json").read_text()) if (lab.out / "latest_research.json").is_file() \
        else {"families": {}}
    t = len(panel.dates) - 1
    session_day = date.fromisoformat(str(panel.dates[t])[:10])
    asof = get_calendar().session_close(session_day)
    families = {f.family: f for f in lab.equity_families(panel)}
    opps: list[Opportunity] = []
    for fam, info in latest.get("families", {}).items():
        spec = families.get(fam)
        params = None
        if spec is not None and info.get("selected_variant"):
            params = next((p for p in spec.variants if ",".join(f"{k}={v}" for k, v in sorted(p.items()))
                           == info["selected_variant"]), None)
        if spec is None or params is None:
            continue
        _, row = latest_decision(spec.make_targets(params), t)
        sid = f"{fam}:{info.get('selected_spec_hash')}"
        state = lab.registry.state(sid)
        status = {State.VALIDATED: EvidenceStatus.VALIDATED, State.SHADOW: EvidenceStatus.SHADOW,
                  State.PAPER: EvidenceStatus.VALIDATED, State.DEV_PASSED: EvidenceStatus.DEV_PASSED,
                  State.REJECTED: EvidenceStatus.REJECTED}.get(state, EvidenceStatus.RESEARCH_ONLY) \
            if state else (EvidenceStatus.REJECTED if not info.get("passed_dev_gate") else EvidenceStatus.DEV_PASSED)
        wf = info.get("wf_oos", {})
        cagr, vol = float(wf.get("cagr") or 0.0), float(wf.get("ann_vol") or 0.2)
        horizon = float(params.get("rebalance", params.get("max_hold", params.get("hold", 5))))
        lifecycle, lifecycle_detail = lifecycle_supported(lab, {"asset_class": spec.asset_class,
                                                               "holding": "multi_day", "strategy_id": sid})
        for j in np.where(row > 1e-6)[0]:
            sym = str(panel.symbols[j])
            ac = "etf" if panel.kind[j] == "ETF" else "us_equity"
            blockers = []
            if active is None:
                blockers.append("active risk policy unreadable")
            if sym not in policy.symbol_allowlist:
                blockers.append("symbol not on the risk-policy allowlist")
            if not caps.get("us_equity_long", {}).get("execution_eligible", False):
                blockers.append("us_equity_long not execution-eligible in the capability snapshot")
            if state != State.PAPER:
                blockers.append("strategy has not completed independent validation and prospective PAPER promotion")
            if not lifecycle:
                blockers.append(lifecycle_detail)
            if horizon > policy.max_holding_sessions:
                blockers.append("strategy horizon exceeds the active policy holding limit")
            adv = float(np.nan_to_num(panel.f["adv20"][t, j]))
            cost = float(EquityCosts().one_way_bps(np.array([adv]), np.array([1000.0]),
                                                   np.array([panel.f["close"][t, j]]), sell=np.array([False]))[0]) * 2
            opps.append(Opportunity(
                opportunity_id=f"{fam}:{sym}:{session_day}", asof=asof, asset_class=ac, instrument=sym,
                strategy_family=fam, direction="LONG", horizon_days=max(1.0, horizon),
                expected_gross_return=(cagr + cost / 1e4 * 252 / max(1.0, horizon)) * horizon / 252,
                expected_cost=cost / 1e4, forecast_std=max(1e-4, vol * math.sqrt(horizon / 252)),
                downside_cvar95=float(wf.get("cvar95_daily") or -0.02) * math.sqrt(horizon),
                capital_required=float(min(policy.max_order_notional, policy.max_position_notional)),
                buying_power_effect=float(min(policy.max_order_notional, policy.max_position_notional)),
                liquidity_capacity_usd=adv * 0.01, expected_slippage_bps=cost / 2,
                evidence=Evidence(status=status, family=fam, report=info.get("run_id"), dsr=info.get("dsr"),
                                  wf_oos_sharpe=wf.get("sharpe"), trades=info.get("trades"),
                                  notes=list(info.get("fail_reasons") or [])[:4]),
                executable=not blockers, execution_blockers=blockers,
                notes=[f"weight {float(row[j]):.3f} in the family's current target portfolio",
                       "expected return = walk-forward OOS CAGR of the family (not a per-name forecast)"]))
    return opps


def crypto_opportunities(cpanel: EquityPanel, lab: Lab) -> list[Opportunity]:
    """Current targets of the crypto families' selected specs (evidence from the latest research run)."""
    latest = json.loads((lab.out / "latest_research.json").read_text()) if (lab.out / "latest_research.json").is_file() \
        else {"families": {}}
    caps = latest_capabilities(lab.root).get("asset_classes", {}).get("crypto_spot", {})
    today = datetime.now(UTC).date().isoformat()
    rows = [i for i, d in enumerate(cpanel.dates) if str(d)[:10] < today]
    if not rows:
        return []
    t = rows[-1]
    specs = {f.family: f for f in lab.crypto_families(cpanel)}
    costs = CryptoCosts(spread_bps=lab.crypto_spreads())
    out: list[Opportunity] = []
    for fam, info in latest.get("families", {}).items():
        spec = specs.get(fam)
        if spec is None or not info.get("selected_variant"):
            continue
        params = next((p for p in spec.variants if ",".join(f"{k}={v}" for k, v in sorted(p.items()))
                       == info["selected_variant"]), None)
        if params is None:
            continue
        _, row = latest_decision(spec.make_targets(params), t)
        wf = info.get("wf_oos", {})
        horizon = float(params.get("rebalance", 7))
        status = EvidenceStatus.DEV_PASSED if info.get("passed_dev_gate") else EvidenceStatus.REJECTED
        for j in np.where(row > 1e-6)[0]:
            pair = str(cpanel.symbols[j])
            c = 2 * costs.one_way_bps(pair) / 1e4
            blockers = ["crypto execution not implemented: " + str(caps.get("policy_execution", {}).get("detail", ""))]
            out.append(Opportunity(
                opportunity_id=f"{fam}:{pair}:{str(cpanel.dates[t])[:10]}",
                asof=datetime.combine(date.fromisoformat(str(cpanel.dates[t])[:10]), datetime.min.time(), tzinfo=UTC)
                     + timedelta(days=1),
                asset_class="crypto",
                instrument=pair, strategy_family=fam, direction="LONG", horizon_days=horizon,
                expected_gross_return=float(wf.get("cagr") or 0.0) * horizon / 365 + c, expected_cost=c,
                forecast_std=max(1e-4, float(wf.get("ann_vol") or 0.6) * math.sqrt(horizon / 365)),
                downside_cvar95=float(wf.get("cvar95_daily") or -0.05) * math.sqrt(horizon),
                capital_required=0.0, buying_power_effect=0.0,
                evidence=Evidence(status=status, family=fam, report=info.get("run_id"), dsr=info.get("dsr"),
                                  wf_oos_sharpe=wf.get("sharpe"), notes=list(info.get("fail_reasons") or [])[:3]),
                executable=False, execution_blockers=blockers,
                notes=[f"weight {float(row[j]):.3f} in the family's current target", "24/7; Alpaca tier-1 fees"]))
    return out


def option_opportunities(lab: Lab) -> list[Opportunity]:
    """Defined-risk spreads from the latest indicative options scan (research-only, never executable here)."""
    path = lab.out / "latest_options_scan.json"
    if not path.is_file():
        return []
    scan = json.loads(path.read_text())
    out: list[Opportunity] = []
    for u in scan.get("underlyings", []):
        for sp in u.get("spreads", []):
            cap = float(sp.get("capital_per_spread") or 0.0)
            if cap <= 0:
                continue
            fees = float(sp.get("fees_per_share") or 0.0) * 100.0 / cap
            days = max(1.0, float(sp.get("T_years") or 0.0) * 365.0)
            out.append(Opportunity(
                opportunity_id=f"E_options:{u['underlying']}:{sp['name']}:{sp['expiry']}", asof=datetime.now(UTC),
                asset_class="us_option", instrument=f"{u['underlying']} {sp['name']} {sp['expiry']}",
                legs=[{"long": sp["long"], "short": sp["short"]}], strategy_family="E_options",
                direction="CREDIT_SPREAD" if "credit" in sp["name"] else "DEBIT_SPREAD", horizon_days=days,
                expected_gross_return=float(sp["expected_pnl_frac"]) + fees, expected_cost=fees,
                forecast_std=max(1e-4, float(sp["std_pnl_frac"])), downside_cvar95=min(0.0, float(sp["cvar95_frac"])),
                max_loss_fraction=1.0, capital_required=cap, buying_power_effect=cap,
                evidence=Evidence(status=EvidenceStatus.RESEARCH_ONLY, family="E_options",
                                  notes=["indicative quotes; no executable quote history for validation"]),
                executable=False, execution_blockers=["options disabled by the risk policy (LIMIT equity orders only)",
                                                      "OPRA real-time quotes not entitled"],
                notes=[f"prob. profit {float(sp['prob_profit']):.2f} under the HAR-RV lognormal model",
                       "entry at executable touch (buy ask / sell bid) plus per-contract fees"]))
    return out


def run_selection(opps: list[Opportunity], out: Path) -> dict[str, Any]:
    sel = select(opps)
    payload = clean_json(json.loads(sel.model_dump_json()))
    payload["counts"] = {"candidates": len(opps), "selectable": sum(r["selectable"] for r in payload["ranked"])}
    (out / "latest_selection.json").write_text(json.dumps(payload, indent=1))
    return payload


def holdout_family_spec(lab: Lab, family: str, full: EquityPanel) -> FamilySpec:
    fams = {f.family: f for f in lab.equity_families(full)}
    if family in fams:
        return fams[family]
    raise KeyError(family)


_ = (BacktestResult, run_weights, date, TransitionRefused)
