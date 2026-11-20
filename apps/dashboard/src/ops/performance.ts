/**
 * Performance strip view model (pure). Every value comes from GET /api/v1/performance as a decimal
 * string and is formatted exactly (lib/decimal); nothing is recomputed or estimated here. The only
 * float use is the sparkline's SVG coordinates (display only, via toChartNumber).
 */
import type { PerformanceResponse } from "../api/types";
import { compareDecimalStrings, formatDecimal, formatFractionAsPercent, formatMoney, toChartNumber } from "../lib/decimal";
import type { Tone } from "./labels";

export interface PerfTile {
  id: string;
  label: string;
  value: string;
  tone: Tone;
  hint?: string;
}

function signTone(value: unknown): Tone {
  const c = compareDecimalStrings(value, "0");
  return c === null ? "neutral" : c > 0 ? "good" : c < 0 ? "bad" : "neutral";
}

/** "$1,234.56" with an explicit sign for P&L values. */
export function formatPnl(value: unknown): string {
  return formatMoney(value, { signDisplay: "always" });
}

/** API cost: cents are too coarse for per-call model costs, keep up to 4 decimals. */
export function formatCost(value: unknown): string {
  return formatDecimal(value, { minFractionDigits: 2, maxFractionDigits: 4, prefix: "$" });
}

export const MODE_LABEL_TEXT: Record<string, string> = {
  BACKTEST: "Backtest",
  PAPER: "Paper",
  LIVE: "Live",
  SIMULATED: "Simulated",
};

export function performanceTiles(p: PerformanceResponse): PerfTile[] {
  const winRate = p.win_rate === null || p.win_rate === undefined ? "—" : formatFractionAsPercent(p.win_rate, 1);
  const tiles: PerfTile[] = [
    { id: "portfolio_value", label: "Portfolio value", value: formatMoney(p.portfolio_value), tone: "neutral", hint: p.cash ? `cash ${formatMoney(p.cash)}` : undefined },
    { id: "daily_pnl", label: "Daily P&L", value: formatPnl(p.daily_pnl), tone: signTone(p.daily_pnl) },
    {
      id: "total_pnl",
      label: "Total P&L",
      value: formatPnl(p.total_pnl),
      tone: signTone(p.total_pnl),
      hint:
        p.realized_pnl !== undefined || p.unrealized_pnl !== undefined
          ? `realized ${formatPnl(p.realized_pnl ?? null)} · unrealized ${formatPnl(p.unrealized_pnl ?? null)}`
          : undefined,
    },
    {
      id: "win_rate",
      label: "Win rate",
      value: winRate,
      tone: "neutral",
      hint: p.win_rate === null || p.win_rate === undefined ? "no closed round trip yet" : `${p.round_trips} round trip${p.round_trips === 1 ? "" : "s"}`,
    },
    {
      id: "drawdown",
      label: "Account drawdown",
      value: formatFractionAsPercent(p.max_drawdown, 2),
      tone: (compareDecimalStrings(p.max_drawdown, "0") ?? 0) > 0 ? "warn" : "neutral",
      hint: p.current_drawdown !== undefined && p.current_drawdown !== null ? `current ${formatFractionAsPercent(p.current_drawdown, 2)}` : undefined,
    },
    { id: "active_positions", label: "Active positions", value: Number.isInteger(p.active_positions) ? String(p.active_positions) : "—", tone: "neutral" },
    { id: "open_orders", label: "Open orders", value: Number.isInteger(p.open_orders) ? String(p.open_orders) : "—", tone: p.open_orders > 0 ? "info" : "neutral" },
    { id: "api_cost", label: "API cost", value: formatCost(p.api_cost_usd), tone: "neutral", hint: "model API spend recorded by the backend" },
  ];
  return tiles;
}

export interface Sparkline {
  /** SVG path of the line */
  line: string;
  /** closed SVG path for the area fill */
  area: string;
  points: number;
  first: string;
  last: string;
  min: string;
  max: string;
  trend: -1 | 0 | 1;
}

/**
 * Sparkline geometry for the equity curve. Points are ordered by time (duplicates/unparseable
 * values dropped). min/max/first/last are the exact decimal strings from the API.
 */
export function sparkline(curve: readonly { time: number; equity: string }[] | null | undefined, width: number, height: number, pad = 2): Sparkline | null {
  if (!Array.isArray(curve)) return null;
  const pts: { t: number; v: number; raw: string }[] = [];
  let lastT = -Infinity;
  for (const p of [...curve].sort((a, b) => a.time - b.time)) {
    const v = toChartNumber(p.equity);
    if (v === null || !Number.isFinite(p.time) || p.time <= lastT) continue;
    lastT = p.time;
    pts.push({ t: p.time, v, raw: p.equity });
  }
  if (pts.length < 2) return null;
  let minRaw = pts[0]!.raw;
  let maxRaw = pts[0]!.raw;
  for (const p of pts) {
    if ((compareDecimalStrings(p.raw, minRaw) ?? 0) < 0) minRaw = p.raw;
    if ((compareDecimalStrings(p.raw, maxRaw) ?? 0) > 0) maxRaw = p.raw;
  }
  const t0 = pts[0]!.t;
  const t1 = pts[pts.length - 1]!.t;
  const vMin = Math.min(...pts.map((p) => p.v));
  const vMax = Math.max(...pts.map((p) => p.v));
  const spanT = t1 - t0 || 1;
  const spanV = vMax - vMin || 1;
  const x = (t: number) => pad + ((t - t0) / spanT) * (width - 2 * pad);
  const y = (v: number) => (vMax === vMin ? height / 2 : pad + (1 - (v - vMin) / spanV) * (height - 2 * pad));
  const r = (n: number) => Math.round(n * 100) / 100;
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${r(x(p.t))},${r(y(p.v))}`).join(" ");
  const area = `${line} L${r(x(t1))},${height} L${r(x(t0))},${height} Z`;
  const first = pts[0]!.raw;
  const last = pts[pts.length - 1]!.raw;
  const c = compareDecimalStrings(last, first) ?? 0;
  return { line, area, points: pts.length, first, last, min: minRaw, max: maxRaw, trend: c > 0 ? 1 : c < 0 ? -1 : 0 };
}
