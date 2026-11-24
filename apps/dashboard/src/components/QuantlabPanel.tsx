/**
 * Multi-asset research (read-only): verified Alpaca capabilities, after-cost strategy-family validation,
 * point-in-time opportunity scan and the cross-asset TRADE/NO_TRADE selection. Every number shown comes from
 * files produced by `python -m services.quantlab.cli`; backtests are labelled as such, never as realized P&L.
 */
import { useEffect, useState } from "react";
import { ApiClient } from "../api/client";
import { formatDecimal, formatFractionAsPercent } from "../lib/decimal";
import { Badge, Panel } from "./ui";

type Layer = { status: string; detail: string };
type AssetClass = {
  research_eligible: boolean;
  execution_eligible: boolean;
  api_schema: Layer;
  market_data: Layer;
  account_permission: Layer;
  policy_execution: Layer;
};
type Capabilities = { status: string; asof?: string; asset_classes?: Record<string, AssetClass> };
type Family = {
  family: string;
  asset_class: string | null;
  passed_dev_gate: boolean;
  wf_oos: { sharpe: number | null; cagr: number | null; max_drawdown: number | null; start?: string; end?: string };
  benchmark: string | null;
  benchmark_wf_period: { sharpe: number | null; cagr: number | null; max_drawdown: number | null };
  dsr: number | null;
  n_trials: number | null;
  spa_p_vs_benchmark: number | null;
  alpha_t: number | null;
  pbo: number | null;
  stress_sharpe_2x_costs: number | null;
  fail_reasons: string[];
};
type Research = { status: string; updated_at?: string; holdout?: string[]; families: Family[]; no_edge_note?: string };
type Ranked = {
  opportunity: { instrument: string; strategy_family: string; asset_class: string; evidence: { status: string } };
  utility: number;
  selectable: boolean;
  reasons: string[];
};
type ScanItem = { symbol: string; score: number; why: string };
type Opportunities = {
  status: string;
  scan?: { asof?: string; regime?: { label?: string; breadth_above_sma50?: number }; relative_strength?: ScanItem[];
    unusual_volume?: ScanItem[]; gap_movers?: ScanItem[] };
  selection?: { action?: string; reason?: string; ranked?: Ranked[] };
};
type Registry = { status: string; strategies: { strategy_id: string; state: string; shadow_sessions?: number }[] };

const api = new ApiClient();

/** Research statistics arrive as JSON numbers; format through the decimal helpers (no float formatting). */
function pct(x: number | null | undefined, digits = 1): string {
  return x == null ? "—" : formatFractionAsPercent(String(x), digits);
}
function num(x: number | null | undefined, digits = 2): string {
  return x == null ? "—" : formatDecimal(String(x), { fractionDigits: digits, grouping: false });
}

export function QuantlabPanel() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [research, setResearch] = useState<Research | null>(null);
  const [opps, setOpps] = useState<Opportunities | null>(null);
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const [c, r, o, g] = await Promise.all([
          api.get<Capabilities>("/quantlab/capabilities", undefined, controller.signal),
          api.get<Research>("/quantlab/research", undefined, controller.signal),
          api.get<Opportunities>("/quantlab/opportunities", undefined, controller.signal),
          api.get<Registry>("/quantlab/strategies", undefined, controller.signal),
        ]);
        if (!controller.signal.aborted) { setCaps(c); setResearch(r); setOpps(o); setRegistry(g); setError(null); }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 30000);
      }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);
  const sel = opps?.selection;
  return <Panel title="Multi-asset research" id="quantlab" source="/quantlab/*">
    <p className="muted small">
      Backtested development results after modelled costs · holdout {research?.holdout?.join(" → ") ?? "—"} protected ·
      not realized performance
    </p>
    {error && <p role="status">Research data unavailable: {error}</p>}
    {!error && !research && <p>Loading research results…</p>}
    {caps?.asset_classes && <div className="research-table"><table>
      <caption className="small muted">Alpaca capabilities (verified {caps.asof?.slice(0, 16) ?? "—"})</caption>
      <thead><tr><th>Asset class</th><th>Data</th><th>Account</th><th>Policy</th><th>Research</th><th>Execution</th></tr></thead>
      <tbody>{Object.entries(caps.asset_classes).map(([name, c]) => <tr key={name}>
        <td>{name}</td><td>{c.market_data.status}</td><td>{c.account_permission.status}</td>
        <td title={c.policy_execution.detail}>{c.policy_execution.status}</td>
        <td>{c.research_eligible ? "yes" : "no"}</td>
        <td>{c.execution_eligible ? <Badge tone="good">yes</Badge> : "no"}</td>
      </tr>)}</tbody>
    </table></div>}
    {research && research.families.length > 0 && <div className="research-table"><table>
      <caption className="small muted">Strategy families — walk-forward out-of-sample, after costs</caption>
      <thead><tr><th>Family</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Benchmark Sharpe</th><th>DSR</th>
        <th>SPA p</th><th>PBO</th><th>Verdict</th></tr></thead>
      <tbody>{research.families.map(f => <tr key={f.family} data-state={f.passed_dev_gate ? "PASS" : "FAIL"}>
        <td title={f.asset_class ?? ""}>{f.family}</td><td>{num(f.wf_oos.sharpe)}</td><td>{pct(f.wf_oos.cagr)}</td>
        <td>{pct(f.wf_oos.max_drawdown)}</td><td>{num(f.benchmark_wf_period?.sharpe)}</td><td>{num(f.dsr, 3)}</td>
        <td>{num(f.spa_p_vs_benchmark, 2)}</td><td>{num(f.pbo, 2)}</td>
        <td title={f.fail_reasons.join("; ")}>{f.passed_dev_gate ? "PASS" : `FAIL · ${f.fail_reasons[0] ?? ""}`}</td>
      </tr>)}</tbody>
    </table></div>}
    {research && research.status === "MISSING" && <p>No research run recorded yet.</p>}
    {sel && <p data-testid="quantlab-selection">
      Selection {opps?.scan?.asof ?? ""}: <strong>{sel.action ?? "—"}</strong> — {sel.reason ?? ""}
      {opps?.scan?.regime?.label ? ` · regime ${opps.scan.regime.label}` : ""}
    </p>}
    {sel?.ranked && sel.ranked.length > 0 && <div className="research-table"><table>
      <caption className="small muted">Cross-asset ranking (evidence-weighted utility)</caption>
      <thead><tr><th>Instrument</th><th>Family</th><th>Evidence</th><th>Utility/day</th><th>Why not selected</th></tr></thead>
      <tbody>{sel.ranked.slice(0, 8).map((r, i) => <tr key={`${r.opportunity.instrument}-${i}`}>
        <td>{r.opportunity.instrument}</td><td>{r.opportunity.strategy_family}</td><td>{r.opportunity.evidence.status}</td>
        <td>{num(r.utility * 10000, 3)} bp</td><td>{r.selectable ? "selectable" : (r.reasons[0] ?? "")}</td>
      </tr>)}</tbody>
    </table></div>}
    {opps?.scan?.relative_strength && <p className="small">
      Relative strength: {opps.scan.relative_strength.slice(0, 5).map(x => x.symbol).join(", ") || "—"} · Unusual
      volume: {opps.scan.unusual_volume?.slice(0, 5).map(x => x.symbol).join(", ") || "—"}
    </p>}
    {registry && registry.strategies.length > 0 && <p className="small">
      Registry: {registry.strategies.map(s => `${s.strategy_id.split(":")[0]} ${s.state}`).join(" · ")}
    </p>}
    {research?.no_edge_note && <p className="muted small">{research.no_edge_note}</p>}
  </Panel>;
}
