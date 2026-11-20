/**
 * Compact performance strip (GET /api/v1/performance): portfolio value, daily/total P&L, win rate,
 * drawdown, active positions, open orders, API cost, mode label, and an inline-SVG equity sparkline.
 * All values are the backend's decimal strings formatted exactly; nothing is estimated client-side.
 */
import type { PerformanceResponse } from "../api/types";
import { formatMoney } from "../lib/decimal";
import { MODE_LABEL_TEXT, performanceTiles, sparkline } from "../ops/performance";
import { useApp } from "../state/AppContext";
import { useResource } from "../state/useResource";
import { Badge, EmptyState, FetchStamp, Panel, Stamp } from "./ui";

const W = 220;
const H = 48;

export function Sparkline({ curve }: { curve: PerformanceResponse["equity_curve"] | null | undefined }) {
  const s = sparkline(curve, W, H);
  if (!s) {
    return (
      <div className="sparkline sparkline-empty" data-testid="sparkline-empty">
        equity curve: {Array.isArray(curve) && curve.length === 1 ? "1 point (need 2)" : "no points yet"}
      </div>
    );
  }
  const trend = s.trend > 0 ? "up" : s.trend < 0 ? "down" : "flat";
  return (
    <figure className={`sparkline trend-${trend}`} data-testid="sparkline">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Equity curve, ${s.points} points, from ${formatMoney(s.first)} to ${formatMoney(s.last)}; low ${formatMoney(s.min)}, high ${formatMoney(s.max)}`}>
        <path className="spark-area" d={s.area} />
        <path className="spark-line" d={s.line} />
      </svg>
      <figcaption className="muted small">
        {formatMoney(s.first)} → {formatMoney(s.last)} · {s.points} pts
      </figcaption>
    </figure>
  );
}

export function PerformanceStripView({ data, error, fetchedAt, sim }: { data: PerformanceResponse | null; error: unknown; fetchedAt: number | null; sim: boolean }) {
  const modeLabel = data ? data.mode_label : null;
  return (
    <Panel
      id="performance"
      title="Performance"
      source="/performance"
      className="performance-panel"
      error={error}
      meta={
        <>
          {modeLabel && (
            <Badge tone={modeLabel === "LIVE" ? "bad" : modeLabel === "PAPER" ? "info" : "warn"} testId="performance-mode">
              {MODE_LABEL_TEXT[modeLabel] ?? modeLabel} mode{data?.is_simulated ? " (SIMULATED)" : ""}
            </Badge>
          )}
          {data && <Stamp ts={data.asof} sim={sim} label="as of" />}
          <FetchStamp fetchedAt={fetchedAt} />
        </>
      }
    >
      {!data ? (
        <EmptyState title={error ? "Performance unavailable" : "Loading performance…"} testId="performance-empty">
          {error ? "The backend did not return a performance report; no values are shown rather than guessed." : null}
        </EmptyState>
      ) : (
        <div className="perf-strip">
          <ul className="perf-tiles" data-testid="perf-tiles">
            {performanceTiles(data).map((t) => (
              <li key={t.id} className={`perf-tile tone-${t.tone}`} data-testid={`perf-${t.id}`}>
                <span className="perf-label">{t.label}</span>
                <span className="perf-value">{t.value}</span>
                {t.hint && <span className="perf-hint">{t.hint}</span>}
              </li>
            ))}
          </ul>
          <Sparkline curve={data.equity_curve} />
        </div>
      )}
      {data && <p className="muted small">Equity and account drawdown are balance-based; cash transfers and paper-account resets are not adjusted.</p>}
    </Panel>
  );
}

export function PerformanceStrip() {
  const { api, ops } = useApp();
  const res = useResource("performance", (signal) => api.performance(signal), { intervalMs: 10_000, refreshOn: ["order", "cycle", "reconciliation"] });
  return <PerformanceStripView data={res.data} error={res.error} fetchedAt={res.fetchedAt} sim={ops.clock.sim !== null} />;
}
