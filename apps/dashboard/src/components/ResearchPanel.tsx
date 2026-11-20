import { useEffect, useState } from "react";
import { ApiClient } from "../api/client";
import { formatDecimal } from "../lib/decimal";
import { Panel } from "./ui";

type Job = {
  id: string;
  state: string;
  spec: { family: string; transform: string };
  error: string | null;
  result: null | {
    is_simulated: boolean;
    metrics: { net_sharpe: number | null; n_trades: number; net_return: number };
  };
};
type Research = {
  counts: Record<string, number>;
  jobs: Job[];
  events: { seq: number; at: string; stage: string; detail: string }[];
  promotion_status: string;
  paper?: { active: number | null; confirmations: { id: number; state: string }[] };
};
const api = new ApiClient();

export function ResearchPanel() {
  const [data, setData] = useState<Research | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const next = await api.get<Research>("/research/autonomous", undefined, controller.signal);
        if (!controller.signal.aborted) { setData(next); setError(null); }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 5000);
      }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);
  return <Panel title="Autonomous research" id="research" source="/research/autonomous">
    <p className="muted small">Recorded development experiments · independent validation required before promotion</p>
    {error && <p role="status">Research feed unavailable: {error}</p>}
    {!data && !error && <p>Loading recorded research…</p>}
    {data && <>
      <p>Promotion: {data.promotion_status}{data.paper?.active ? ` · paper confirmation ${data.paper.active}` : ""}</p>
      <p>{Object.entries(data.counts).map(([state, count]) => `${state}: ${count}`).join(" · ") || "No experiments recorded"}</p>
      <div className="research-table"><table>
        <thead><tr><th>Model / features</th><th>State</th><th>Net Sharpe</th><th>Trades</th><th>Evidence</th></tr></thead>
        <tbody>{data.jobs.slice(-12).reverse().map(job => <tr key={job.id} data-state={job.state}>
          <td>{job.spec.family} / {job.spec.transform}</td><td>{job.state}</td>
          <td>{formatDecimal(job.result?.metrics.net_sharpe == null ? null : String(job.result.metrics.net_sharpe), { fractionDigits: 2 })}</td>
          <td>{job.result?.metrics.n_trades ?? "—"}</td>
          <td>{job.error || (job.result ? (job.result.is_simulated ? "SIMULATED · development only" : "Development only") : "Pending")}</td>
        </tr>)}</tbody>
      </table></div>
      {data.events.slice(0, 5).map(event => <p className="small" key={event.seq}>
        <time>{event.at}</time> · {event.stage} · {event.detail}
      </p>)}
    </>}
  </Panel>;
}
