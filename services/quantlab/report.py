"""Render the multi-asset research verification report (Markdown) from recorded artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import results_root
from .views import capabilities, research


def _pct(x: Any, d: int = 1) -> str:
    return "—" if x is None else f"{float(x) * 100:.{d}f}%"


def _num(x: Any, d: int = 2) -> str:
    return "—" if x is None else f"{float(x):.{d}f}"


def render(root: Path) -> str:
    out = results_root(root)
    caps = capabilities(root)
    res = research(root)
    lines: list[str] = []
    lines.append("## Strategy families (development period, walk-forward out-of-sample, after modelled costs)\n")
    lines.append(f"Holdout (protected, untouched by development): {' → '.join(res.get('holdout') or [])}. "
                 f"{res.get('no_edge_note') or ''}\n")
    lines.append("| Family | Asset | Variants | WF OOS period | Sharpe | CAGR | Max DD | Benchmark Sharpe / CAGR / DD |"
                 " DSR (trials) | SPA p | alpha t | PBO | 2x cost Sharpe | Verdict |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
    for f in res.get("families", []):
        wf = f.get("wf_oos") or {}
        b = f.get("benchmark_wf_period") or {}
        verdict = "PASS" if f.get("passed_dev_gate") else "FAIL: " + "; ".join((f.get("fail_reasons") or [])[:2])
        lines.append(
            f"| {f['family']} | {f.get('asset_class')} | {f.get('n_variants')} | {wf.get('start')}…{wf.get('end')} | "
            f"{_num(wf.get('sharpe'))} | {_pct(wf.get('cagr'))} | {_pct(wf.get('max_drawdown'))} | "
            f"{_num(b.get('sharpe'))} / {_pct(b.get('cagr'))} / {_pct(b.get('max_drawdown'))} | "
            f"{_num(f.get('dsr'), 3)} ({int(f.get('n_trials') or 0)}) | {_num(f.get('spa_p_vs_benchmark'))} | "
            f"{_num(f.get('alpha_t'))} | {_num(f.get('pbo'))} | {_num(f.get('stress_sharpe_2x_costs'))} | {verdict} |")
    lines.append("")
    lines.append("### Execution tier: selected variant under today's risk policy ($5,000 account, allowlist, "
                 "1 position ≤ 25% of equity, whole shares)\n")
    lines.append("| Family | Sharpe | CAGR | Max DD | Avg exposure | Trades |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for f in res.get("families", []):
        pt = f.get("policy_tier") or {}
        m = pt.get("metrics") or {}
        if pt.get("status") != "OK":
            lines.append(f"| {f['family']} | {pt.get('reason', pt.get('status', '—'))} | | | | |")
            continue
        lines.append(f"| {f['family']} | {_num(m.get('sharpe'))} | {_pct(m.get('cagr'))} | "
                     f"{_pct(m.get('max_drawdown'))} | {_pct(m.get('avg_exposure'))} | {m.get('trades')} |")
    lines.append("")
    looks = res.get("holdout_results") or []
    lines.append("### Protected holdout looks\n")
    if not looks:
        lines.append("None. No family passed the development gate, so the holdout was never opened.\n")
    for h in looks:
        m = h.get("metrics", {})
        lines.append(f"* `{h.get('strategy_id')}`: {'PASS' if h.get('passed') else 'FAIL'} — return "
                     f"{_pct(m.get('total_return'))}, Sharpe {_num(m.get('sharpe'))}, max DD {_pct(m.get('max_drawdown'))}"
                     f"; {'; '.join(h.get('fail_reasons') or [])}")
    sel = json.loads((out / "latest_selection.json").read_text()) if (out / "latest_selection.json").is_file() else {}
    if sel:
        lines.append(f"\n## Cross-asset selection ({str(sel.get('asof'))[:10]}): **{sel.get('action')}** — "
                     f"{sel.get('reason')}\n")
        for r in (sel.get("ranked") or [])[:6]:
            o = r["opportunity"]
            lines.append(f"* {o['instrument']} ({o['strategy_family']}, {o['evidence']['status']}): "
                         f"{'selectable' if r['selectable'] else '; '.join(r['reasons'][:2])}")
    opt = json.loads((out / "latest_options_scan.json").read_text()) if (out / "latest_options_scan.json").is_file() \
        else {}
    if opt:
        lines.append(f"\n## Options volatility scan ({str(opt.get('asof'))[:16]} UTC, {opt.get('feed')})\n")
        lines.append("| Underlying | 30d ATM IV | HAR-RV forecast | IV/RV | 25Δ skew | ATM spread % of premium |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for u in opt.get("underlyings", []):
            s = u.get("surface") or {}
            lines.append(f"| {u['underlying']} | {_pct(s.get('atm_iv_30d'))} | {_pct(s.get('rv_forecast'))} | "
                         f"{_num(s.get('iv_rv_ratio'))} | {_num((s.get('skew_25d_30d') or 0) * 100, 1)} vol pts | "
                         f"{_pct(s.get('atm_spread_pct_30d'))} |")
    cr = json.loads((out / "latest_crypto_scan.json").read_text()) if (out / "latest_crypto_scan.json").is_file() \
        else {}
    if cr:
        lines.append(f"\n## Crypto scan ({cr.get('asof')}): BTC above 50-day average = "
                     f"{cr.get('regime', {}).get('btc_above_sma50')}, BTC 90-day return "
                     f"{_pct(cr.get('regime', {}).get('btc_mom_90d'))}\n")
    lines.append("\n## Capability snapshot\n")
    for name, c in (caps.get("asset_classes") or {}).items():
        lines.append(f"* {name}: research={'yes' if c['research_eligible'] else 'no'}, execution="
                     f"{'yes' if c['execution_eligible'] else 'no'} (policy {c['policy_execution']['status']})")
    return "\n".join(lines) + "\n"
