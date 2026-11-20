/**
 * Response envelopes and SSE event bodies of the local HTTP API (docs/architecture/api-contract.md,
 * v1, incl. "Live operations (read-only dashboard)").
 *
 * Contract objects (ApprovedOrderIntent, OrderEvent, ...) come from the generated
 * `../generated/contracts.ts` (scripts/gen_ts_types.py + json-schema-to-typescript), so they track
 * packages/contracts exactly. The live-operations bodies (stage / activity / decision / session) are
 * plain dicts in services/api/activity.py and are typed here by hand against the contract.
 *
 * All money / quantity / price / bps values are `DecimalString`s: they are formatted with
 * `lib/decimal.ts` and never parsed as floats. All timestamps are ISO-8601 UTC strings.
 * Anything the dashboard does not strictly need is optional so a slightly older/newer backend
 * degrades to an explicit "—" instead of a crash.
 */
import type { ApprovedOrderIntent, DecimalString, OrderEvent, UtcTimestamp } from "../generated/contracts";

export type { ApprovedOrderIntent, DecimalString, OrderEvent, UtcTimestamp };

// ------------------------------------------------------------------ /status
export interface HaltRow {
  halt_id: string;
  reason: string;
  source: string;
  created_at: UtcTimestamp;
}

export interface FeedStatus {
  feed_mode: string;
  provider: string;
  is_simulated: boolean;
  last_bar_ts: UtcTimestamp | null;
  last_quote_ts: UtcTimestamp | null;
  staleness_s: DecimalString | null;
  status: string;
  limitations: string[];
  simulated_time?: UtcTimestamp | null;
  dataset_version?: string | null;
}

export interface StatusResponse {
  mode: string;
  /** SIMULATED | PAPER | LIVE. Anything else (or missing) is displayed as UNKNOWN and treated as unsafe. */
  execution_label: string;
  gateway: string;
  live_execution_compiled: boolean;
  entries_locked: boolean;
  lock_reasons: string[];
  halts: HaltRow[];
  startup: { phase: string; started_at: UtcTimestamp | null; detail: string };
  feed: FeedStatus;
  engine: { kind: string; status: string; last_heartbeat: UtcTimestamp | null; detail: string };
  broker: {
    kind: string;
    is_paper: boolean;
    connected: boolean;
    last_reconciled_at: UtcTimestamp | null;
    reconciliation_status: string | null;
  };
  credentials?: Record<string, boolean>;
  ai?: { transport: string; claude_model: string; sol_model: string };
  policy_version: string;
  server_time: UtcTimestamp;
}

// ------------------------------------------------------------------ /health
export type HealthStatus = "OK" | "DEGRADED" | "DOWN" | "DISABLED";

export interface HealthComponent {
  name: string;
  status: string;
  detail: string;
  last_ok: UtcTimestamp | null;
}

/** GET /api/v1/health, live-operations shape. */
export interface HealthReport {
  overall: string;
  components: HealthComponent[];
  asof?: UtcTimestamp | null;
}

/** GET /api/v1/health, original liveness shape (still served by some API builds). */
export interface SimpleHealth {
  status: string;
  mode: string;
  execution_label: string;
  version: string;
  time: UtcTimestamp;
}

export type HealthResponse = HealthReport | SimpleHealth;

export function isHealthReport(h: unknown): h is HealthReport {
  return typeof h === "object" && h !== null && Array.isArray((h as { components?: unknown }).components);
}

// ------------------------------------------------------------------ orders
export interface OrderRow {
  client_order_id: string;
  broker_order_id: string | null;
  intent_id: string | null;
  symbol: string;
  side: string;
  quantity: DecimalString;
  filled_quantity: DecimalString;
  avg_fill_price: DecimalString | null;
  limit_price: DecimalString | null;
  state: string;
  gateway: string;
  execution_label?: string;
  created_at: UtcTimestamp;
  updated_at: UtcTimestamp;
  /** present on the raw rows pushed by the SSE `order` event */
  causal_id?: string | null;
}

export interface OrdersResponse {
  orders: OrderRow[];
}

export type OrderEventRow = OrderEvent & { applied: boolean; apply_note: string | null };

export interface TransitionRow {
  from_state: string | null;
  to_state: string;
  at: UtcTimestamp;
  reason: string | null;
}

export interface FillRow {
  fill_id: string;
  quantity: DecimalString;
  price: DecimalString;
  ts: UtcTimestamp;
  broker_exec_id?: string | null;
}

export interface OrderDetailResponse {
  order: OrderRow;
  intent: ApprovedOrderIntent | null;
  events: OrderEventRow[];
  transitions: TransitionRow[];
  fills: FillRow[];
}

/** Body of the SSE `order_event` event: one raw `order_events` ledger row. */
export interface OrderEventStreamRow {
  seq: number;
  event_id: string;
  client_order_id: string;
  broker_order_id?: string | null;
  event_type: string;
  state: string;
  source: string;
  gateway: string;
  broker_ts: UtcTimestamp | null;
  received_ts: UtcTimestamp;
  /** SQLite integer 0/1 on the stream */
  applied: number | boolean;
  apply_note: string | null;
  payload?: unknown;
}

// ------------------------------------------------------------------ /audit
export interface AuditEventRow {
  seq: number;
  event_id: string;
  ts: UtcTimestamp;
  event_type: string;
  causal_id: string | null;
  source: string;
  mode: string;
  policy_version: string | null;
  payload: unknown;
}

export interface AuditResponse {
  events: AuditEventRow[];
  chain?: { ok: boolean; rows: number; detail: string };
}

// ------------------------------------------------------------------ live operations
export const STAGE_IDS = [
  "MARKET_DATA",
  "FEATURES",
  "MODELS",
  "GPT_REVIEW",
  "RISK_ENGINE",
  "ORDER_DECISION",
  "BROKER",
  "RECONCILIATION",
] as const;
export type StageId = (typeof STAGE_IDS)[number];

export const STAGE_STATUSES = ["idle", "running", "passed", "rejected", "error", "skipped"] as const;
export type StageStatus = (typeof STAGE_STATUSES)[number];

export const ACTIVITY_KINDS = [
  "scan",
  "predict",
  "evaluate",
  "gpt_call",
  "risk",
  "reject",
  "submit",
  "fill",
  "reconcile",
  "session",
  "agent_tool",
  "error",
  "info",
] as const;
export type ActivityKind = (typeof ACTIVITY_KINDS)[number];

/** SSE `stage` body (also an /activity row with event "stage"). */
export interface StageEvent {
  seq: number;
  ts: UtcTimestamp;
  cycle_id: string | null;
  causal_id: string | null;
  session_id: string | null;
  stage: StageId;
  status: StageStatus;
  item: string | null;
  duration_ms: number | null;
  detail: string;
}

/** SSE `activity` body (also an /activity row with event "activity"). */
export interface ActivityEvent {
  seq: number;
  ts: UtcTimestamp;
  kind: ActivityKind;
  /** system | session:<id> | agent:<name> | operator | risk-engine */
  actor: string;
  message: string;
  causal_id: string | null;
  session_id: string | null;
  detail: Record<string, unknown>;
}

export type DecisionStatus = "PROPOSED" | "APPROVED" | "REJECTED" | "SUBMITTED" | "FILLED" | "NO_TRADE" | "ERROR";

/** SSE `decision` body: the current decision card. */
export interface DecisionCard {
  seq?: number;
  ts: UtcTimestamp;
  cycle_id: string | null;
  causal_id: string | null;
  symbol: string | null;
  side: string | null;
  action: string;
  status: string;
  expected_return_bps: DecimalString | null;
  expected_cost_bps: DecimalString | null;
  net_edge_bps: DecimalString | null;
  /** "0..1" decimal string */
  confidence: DecimalString | null;
  confidence_calibrated: boolean;
  volatility_bps: DecimalString | null;
  regime: string | null;
  reason: string;
  risk_failures: string[];
  proposer: string | null;
  is_simulated: boolean;
}

/** GET /api/v1/activity rows: the three live-ops bodies, tagged by `event`. */
export type ActivityRow =
  | (StageEvent & { event: "stage"; kind?: string })
  | (ActivityEvent & { event: "activity" })
  | (DecisionCard & { event: "decision"; seq: number; kind?: string });

export interface ActivityListResponse {
  events: unknown[];
}

export interface CycleStageSnapshot {
  stage: string;
  status: string;
  item: string | null;
  started_at: UtcTimestamp | null;
  duration_ms: number | null;
  detail: string;
}

export interface CurrentTaskSnapshot {
  message: string;
  kind: string;
  since: UtcTimestamp | null;
}

/** GET /api/v1/activity/current */
export interface ActivityCurrentResponse {
  cycle: {
    cycle_id: string | null;
    causal_id: string | null;
    started_at: UtcTimestamp | null;
    session_id: string | null;
    stages: CycleStageSnapshot[];
  } | null;
  current_task: CurrentTaskSnapshot | null;
  decision: DecisionCard | null;
}

/** SSE `cycle` body (decision cycle summary). */
export interface CycleSummary {
  causal_id: string;
  outcome: string;
  ran_at?: UtcTimestamp;
  reason?: string | null;
  candidate_symbol?: string | null;
  client_order_id?: string | null;
  session_id?: string | null;
  execution_label?: string | null;
}

export interface HeartbeatEvent {
  time: UtcTimestamp;
  simulated_time?: UtcTimestamp | null;
}

// ------------------------------------------------------------------ /performance
export interface PerformanceResponse {
  /** BACKTEST | PAPER | LIVE | SIMULATED */
  mode_label: string;
  portfolio_value: DecimalString | null;
  cash: DecimalString | null;
  daily_pnl: DecimalString | null;
  total_pnl: DecimalString | null;
  realized_pnl?: DecimalString | null;
  unrealized_pnl?: DecimalString | null;
  /** "0..1" decimal string, null when no round trip closed yet */
  win_rate: DecimalString | null;
  round_trips: number;
  /** "0..1" decimal strings */
  max_drawdown: DecimalString | null;
  current_drawdown?: DecimalString | null;
  active_positions: number;
  open_orders: number;
  api_cost_usd: DecimalString | null;
  equity_curve: { time: number; equity: DecimalString }[];
  asof: UtcTimestamp | null;
  is_simulated: boolean;
}

// ------------------------------------------------------------------ /sessions
export interface TradingSession {
  session_id: string;
  /** SCHEDULED | RUNNING | STOPPING | COMPLETED | STOPPED | INTERRUPTED | FAILED */
  state: string;
  started_at: UtcTimestamp | null;
  ends_at: UtcTimestamp | null;
  ended_at: UtcTimestamp | null;
  symbols: string[];
  cadence_minutes: number | null;
  use_ai: boolean;
  requested_by: string | null;
  cycles_run: number;
  trades: number;
  stop_reason: string | null;
  schedule_id?: string | null;
}

export interface SessionSchedule {
  schedule_id: string;
  weekdays: string[];
  start_time_et: string;
  duration_minutes: number;
  symbols: string[];
  cadence_minutes: number;
  use_ai: boolean;
  enabled: boolean;
  next_run_at: UtcTimestamp | null;
  created_by: string | null;
}

export interface SessionsResponse {
  active: TradingSession | null;
  sessions: TradingSession[];
  schedules: SessionSchedule[];
  detail?: string | null;
}

// ------------------------------------------------------------------ SSE
/** SSE event names emitted by GET /api/v1/stream. */
export const STREAM_EVENTS = [
  "status",
  "heartbeat",
  "stage",
  "activity",
  "decision",
  "session",
  "cycle",
  "order",
  "order_event",
  "audit",
  "reconciliation",
  "failure",
] as const;
export type StreamEventName = (typeof STREAM_EVENTS)[number];
