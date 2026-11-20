/**
 * Live-operations state: a PURE reducer over the SSE stream (`stage`, `activity`, `decision`,
 * `cycle`, `order`, `order_event`, `audit`, `session`, `heartbeat`, `status`) and the REST snapshots
 * (`/activity/current`, `/activity`, `/orders`, `/orders/{id}`, `/audit`, `/sessions`).
 *
 * Invariants:
 * - Idempotent: every live-ops row carries the backend's monotonically increasing `activity_events.seq`;
 *   a row whose seq was already applied is ignored (counted), so duplicated or replayed deliveries
 *   never double-apply. Order events dedupe by `event_id`, audit rows by `seq`.
 * - Reconnect resync: on every (re)connect the provider dispatches `resyncStart`, fetches
 *   `/activity/current` + `/activity`, then `resyncApply`. Rows that stream in while the fetch is in
 *   flight are buffered and replayed (in seq order) on top of the snapshot, so the result converges
 *   to the backend state whichever of the two is newer. `lastSeq` is reset from the server's rows,
 *   so a fresh backend database (seq restarting at 1) is never ignored as "duplicates".
 * - Nothing is fabricated: malformed bodies are dropped (counted) and unknown values stay unknown.
 * - Money/qty/price/bps stay decimal strings; this module does no arithmetic on them.
 */
import type {
  ActivityCurrentResponse,
  ActivityEvent,
  ActivityKind,
  AuditEventRow,
  CycleStageSnapshot,
  CycleSummary,
  DecisionCard,
  OrderDetailResponse,
  OrderRow,
  SessionSchedule,
  SessionsResponse,
  StageEvent,
  StageId,
  StageStatus,
  StatusResponse,
  StreamEventName,
  TradingSession,
} from "../api/types";
import { ACTIVITY_KINDS, STAGE_IDS, STAGE_STATUSES } from "../api/types";
import { parseUtc } from "../lib/time";

export const MAX_ROWS = 400;
export const MAX_AUDIT = 200;
export const MAX_ORDER_EVENTS = 300;
export const MAX_ORDERS = 20;

// ------------------------------------------------------------------ state shapes
export interface StageState {
  stage: StageId;
  status: StageStatus;
  item: string | null;
  /** server timestamp (activity clock) of the last `running` transition */
  startedAt: string | null;
  durationMs: number | null;
  detail: string;
  /** client epoch ms when this status was observed live on the stream; null when it came from a snapshot */
  observedAt: number | null;
  /** increments on every applied transition (animation key) */
  version: number;
}

export type PipelinePhase = "none" | "running" | "complete";

export interface PipelineState {
  cycleId: string | null;
  causalId: string | null;
  sessionId: string | null;
  startedAt: string | null;
  stages: StageState[];
  phase: PipelinePhase;
  source: "none" | "snapshot" | "stream";
}

export interface CurrentTask {
  message: string;
  kind: string;
  /** server timestamp (activity clock) */
  since: string | null;
  actor: string | null;
  causalId: string | null;
  /** client epoch ms when observed live (null when from a snapshot) */
  observedAt: number | null;
}

export type LiveRow =
  | { event: "stage"; seq: number; data: StageEvent }
  | { event: "activity"; seq: number; data: ActivityEvent }
  | { event: "decision"; seq: number; data: DecisionCard };

export interface OrderEventItem {
  eventId: string;
  seq: number | null;
  clientOrderId: string;
  eventType: string;
  state: string;
  source: string;
  ts: string | null;
  applied: boolean;
  note: string | null;
}

export interface ServerClock {
  /** server wall clock minus client clock, ms (from heartbeat/status `time`/`server_time`) */
  wallOffsetMs: number | null;
  /** latest simulated-time sample (the activity clock in SIMULATED feed mode) */
  sim: { simMs: number; clientMs: number; rate: number | null } | null;
}

export interface SessionsState {
  loaded: boolean;
  active: TradingSession | null;
  list: TradingSession[];
  schedules: SessionSchedule[];
  detail: string | null;
}

export interface LiveOpsState {
  pipeline: PipelineState;
  currentTask: CurrentTask | null;
  decision: DecisionCard | null;
  /** live-ops rows (stage/activity/decision), newest first by seq */
  rows: LiveRow[];
  lastSeq: number;
  resync: { active: boolean; buffer: { row: LiveRow; at: number }[] };
  /** true once a snapshot or list from the backend has been applied */
  synced: boolean;
  lastCycle: CycleSummary | null;
  orders: Record<string, OrderRow>;
  details: Record<string, OrderDetailResponse>;
  /** client_order_id -> causal id (from the signed intent or the raw order row) */
  orderCausal: Record<string, string>;
  orderEvents: OrderEventItem[];
  audit: AuditEventRow[];
  sessions: SessionsState;
  clock: ServerClock;
  counters: { duplicatesIgnored: number; malformed: number };
}

export type LiveOpsAction =
  | { type: "stream"; name: StreamEventName; data: unknown; at: number }
  | { type: "resyncStart" }
  | { type: "resyncApply"; current: ActivityCurrentResponse; recent: readonly unknown[]; at: number }
  | { type: "resyncAbort" }
  | { type: "snapshot"; current: ActivityCurrentResponse; at: number }
  | { type: "activityList"; rows: readonly unknown[] }
  | { type: "auditList"; rows: readonly unknown[] }
  | { type: "ordersList"; orders: readonly unknown[] }
  | { type: "orderDetail"; detail: OrderDetailResponse }
  | { type: "sessions"; data: SessionsResponse }
  | { type: "status"; status: StatusResponse; at: number };

// ------------------------------------------------------------------ constructors
export function idleStages(): StageState[] {
  return STAGE_IDS.map((stage) => ({
    stage,
    status: "idle" as StageStatus,
    item: null,
    startedAt: null,
    durationMs: null,
    detail: "",
    observedAt: null,
    version: 0,
  }));
}

export function emptyPipeline(): PipelineState {
  return { cycleId: null, causalId: null, sessionId: null, startedAt: null, stages: idleStages(), phase: "none", source: "none" };
}

export function initialLiveOpsState(): LiveOpsState {
  return {
    pipeline: emptyPipeline(),
    currentTask: null,
    decision: null,
    rows: [],
    lastSeq: 0,
    resync: { active: false, buffer: [] },
    synced: false,
    lastCycle: null,
    orders: {},
    details: {},
    orderCausal: {},
    orderEvents: [],
    audit: [],
    sessions: { loaded: false, active: null, list: [], schedules: [], detail: null },
    clock: { wallOffsetMs: null, sim: null },
    counters: { duplicatesIgnored: 0, malformed: 0 },
  };
}

// ------------------------------------------------------------------ validation / normalization
function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function str(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}
function seqOf(v: unknown): number | null {
  return typeof v === "number" && Number.isSafeInteger(v) && v > 0 ? v : null;
}
function intOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? Math.round(v) : null;
}
const STAGE_SET = new Set<string>(STAGE_IDS);
const STATUS_SET = new Set<string>(STAGE_STATUSES);
const KIND_SET = new Set<string>(ACTIVITY_KINDS);

export function isStageId(v: unknown): v is StageId {
  return typeof v === "string" && STAGE_SET.has(v);
}

function toStage(raw: Record<string, unknown>): StageEvent | null {
  const seq = seqOf(raw.seq);
  const ts = str(raw.ts);
  if (seq === null || ts === null || !isStageId(raw.stage) || typeof raw.status !== "string" || !STATUS_SET.has(raw.status)) return null;
  return {
    seq,
    ts,
    cycle_id: str(raw.cycle_id),
    causal_id: str(raw.causal_id),
    session_id: str(raw.session_id),
    stage: raw.stage,
    status: raw.status as StageStatus,
    item: str(raw.item),
    duration_ms: intOrNull(raw.duration_ms),
    detail: str(raw.detail) ?? "",
  };
}

function toActivity(raw: Record<string, unknown>): ActivityEvent | null {
  const seq = seqOf(raw.seq);
  const ts = str(raw.ts);
  const message = str(raw.message);
  if (seq === null || ts === null || message === null) return null;
  const kind = typeof raw.kind === "string" && KIND_SET.has(raw.kind) ? (raw.kind as ActivityKind) : "info";
  return {
    seq,
    ts,
    kind,
    actor: str(raw.actor) ?? "system",
    message,
    causal_id: str(raw.causal_id),
    session_id: str(raw.session_id),
    detail: isObj(raw.detail) ? raw.detail : {},
  };
}

function decimalOrNull(v: unknown): string | null {
  return typeof v === "string" && v !== "" ? v : null;
}

export function toDecision(raw: unknown): DecisionCard | null {
  if (!isObj(raw)) return null;
  const action = str(raw.action);
  const status = str(raw.status);
  const ts = str(raw.ts);
  if (action === null || status === null || ts === null) return null;
  const card: DecisionCard = {
    ts,
    cycle_id: str(raw.cycle_id),
    causal_id: str(raw.causal_id),
    symbol: str(raw.symbol),
    side: str(raw.side),
    action,
    status,
    expected_return_bps: decimalOrNull(raw.expected_return_bps),
    expected_cost_bps: decimalOrNull(raw.expected_cost_bps),
    net_edge_bps: decimalOrNull(raw.net_edge_bps),
    confidence: decimalOrNull(raw.confidence),
    confidence_calibrated: raw.confidence_calibrated === true, // anything else is uncalibrated
    volatility_bps: decimalOrNull(raw.volatility_bps),
    regime: str(raw.regime),
    reason: str(raw.reason) ?? "",
    risk_failures: Array.isArray(raw.risk_failures) ? raw.risk_failures.filter((x): x is string => typeof x === "string") : [],
    proposer: str(raw.proposer),
    is_simulated: raw.is_simulated !== false, // fail safe: only an explicit false is "not simulated"
  };
  const seq = seqOf(raw.seq);
  if (seq !== null) card.seq = seq;
  return card;
}

/**
 * Normalize an /activity row or an SSE body. `hint` is the SSE event name; REST rows carry `event`
 * ("stage" | "activity" | "decision") and, for older API builds, only `kind`.
 */
export function normalizeRow(raw: unknown, hint?: "stage" | "activity" | "decision"): LiveRow | null {
  if (!isObj(raw)) return null;
  const tag = hint ?? str(raw.event) ?? (raw.kind === "stage" ? "stage" : raw.kind === "decision" ? "decision" : "activity");
  if (tag === "stage") {
    const data = toStage(raw);
    return data ? { event: "stage", seq: data.seq, data } : null;
  }
  if (tag === "decision") {
    const data = toDecision(raw);
    return data && data.seq !== undefined ? { event: "decision", seq: data.seq, data } : null;
  }
  if (tag === "activity") {
    const data = toActivity(raw);
    return data ? { event: "activity", seq: data.seq, data } : null;
  }
  return null;
}

// ------------------------------------------------------------------ pipeline
const TERMINAL = new Set<StageStatus>(["passed", "rejected", "error", "skipped"]);

export function isTerminalStage(s: StageStatus): boolean {
  return TERMINAL.has(s);
}

function derivePhase(stages: readonly StageState[], taskCleared: boolean, cycleKnown: boolean): PipelinePhase {
  if (!cycleKnown && stages.every((s) => s.status === "idle")) return "none";
  if (stages.some((s) => s.status === "running")) return "running";
  const recon = stages.find((s) => s.stage === "RECONCILIATION");
  if (recon && isTerminalStage(recon.status)) return "complete";
  if (taskCleared && stages.some((s) => s.status !== "idle")) return "complete";
  return stages.every((s) => s.status === "idle") ? "none" : "running";
}

function applyStage(p: PipelineState, ev: StageEvent, at: number): PipelineState {
  const newCycle = ev.cycle_id !== null && ev.cycle_id !== p.cycleId;
  const base: PipelineState = newCycle
    ? {
        cycleId: ev.cycle_id,
        causalId: ev.causal_id,
        sessionId: ev.session_id,
        startedAt: ev.ts,
        stages: idleStages(),
        phase: "running",
        source: "stream",
      }
    : p;
  const stages = base.stages.map((s) =>
    s.stage !== ev.stage
      ? s
      : {
          ...s,
          status: ev.status,
          item: ev.item ?? s.item,
          startedAt: ev.status === "running" ? ev.ts : s.startedAt,
          durationMs: ev.status === "running" ? null : (ev.duration_ms ?? s.durationMs),
          detail: ev.detail,
          observedAt: at,
          version: s.version + 1,
        },
  );
  return {
    ...base,
    causalId: base.causalId ?? ev.causal_id,
    sessionId: base.sessionId ?? ev.session_id,
    stages,
    phase: derivePhase(stages, false, true),
    source: "stream",
  };
}

function pipelineFromSnapshot(current: ActivityCurrentResponse, taskCleared: boolean): PipelineState {
  const cyc = current.cycle;
  if (!cyc) return emptyPipeline();
  const byStage = new Map<string, CycleStageSnapshot>();
  for (const s of Array.isArray(cyc.stages) ? cyc.stages : []) if (isObj(s) && typeof s.stage === "string") byStage.set(s.stage, s);
  const stages = idleStages().map((s) => {
    const snap = byStage.get(s.stage);
    if (!snap) return s;
    const status = typeof snap.status === "string" && STATUS_SET.has(snap.status) ? (snap.status as StageStatus) : "idle";
    return {
      ...s,
      status,
      item: str(snap.item),
      startedAt: str(snap.started_at),
      durationMs: status === "running" ? null : intOrNull(snap.duration_ms),
      detail: str(snap.detail) ?? "",
      version: status === "idle" ? 0 : 1,
    };
  });
  return {
    cycleId: str(cyc.cycle_id),
    causalId: str(cyc.causal_id),
    sessionId: str(cyc.session_id),
    startedAt: str(cyc.started_at),
    stages,
    phase: derivePhase(stages, taskCleared, true),
    source: "snapshot",
  };
}

function applySnapshot(state: LiveOpsState, current: ActivityCurrentResponse): LiveOpsState {
  const taskRaw = isObj(current) ? current.current_task : null;
  const task: CurrentTask | null =
    isObj(taskRaw) && typeof taskRaw.message === "string"
      ? { message: taskRaw.message, kind: str(taskRaw.kind) ?? "info", since: str(taskRaw.since), actor: null, causalId: null, observedAt: null }
      : null;
  return {
    ...state,
    pipeline: pipelineFromSnapshot(isObj(current) ? current : { cycle: null, current_task: null, decision: null }, task === null),
    currentTask: task,
    decision: isObj(current) ? toDecision(current.decision) : null,
    synced: true,
  };
}

// ------------------------------------------------------------------ rows
function insertRows(rows: readonly LiveRow[], add: readonly LiveRow[]): LiveRow[] {
  if (add.length === 0) return rows as LiveRow[];
  const bySeq = new Map<number, LiveRow>();
  for (const r of rows) bySeq.set(r.seq, r);
  for (const r of add) bySeq.set(r.seq, r);
  return [...bySeq.values()].sort((a, b) => b.seq - a.seq).slice(0, MAX_ROWS);
}

function maxSeq(rows: readonly { seq: number }[]): number {
  let m = 0;
  for (const r of rows) if (r.seq > m) m = r.seq;
  return m;
}

/** State effects of one live-ops row (pipeline, current task, decision card). */
function applyEffects(state: LiveOpsState, row: LiveRow, at: number): LiveOpsState {
  switch (row.event) {
    case "stage":
      return { ...state, pipeline: applyStage(state.pipeline, row.data, at) };
    case "activity": {
      const a = row.data;
      return {
        ...state,
        currentTask: { message: a.message, kind: a.kind, since: a.ts, actor: a.actor, causalId: a.causal_id, observedAt: at },
      };
    }
    case "decision":
      return { ...state, decision: row.data };
  }
}

function applyLiveRow(state: LiveOpsState, row: LiveRow, at: number): LiveOpsState {
  if (state.resync.active) {
    if (state.resync.buffer.some((b) => b.row.seq === row.seq)) {
      return { ...state, counters: { ...state.counters, duplicatesIgnored: state.counters.duplicatesIgnored + 1 } };
    }
    const next = applyEffects(state, row, at);
    return {
      ...next,
      rows: insertRows(next.rows, [row]),
      resync: { active: true, buffer: [...state.resync.buffer, { row, at }] },
    };
  }
  if (row.seq <= state.lastSeq) {
    return { ...state, counters: { ...state.counters, duplicatesIgnored: state.counters.duplicatesIgnored + 1 } };
  }
  const next = applyEffects(state, row, at);
  return { ...next, rows: insertRows(next.rows, [row]), lastSeq: row.seq };
}

function normalizeList(raw: readonly unknown[]): { rows: LiveRow[]; malformed: number } {
  const rows: LiveRow[] = [];
  let malformed = 0;
  for (const r of raw) {
    const n = normalizeRow(r);
    if (n) rows.push(n);
    else malformed += 1;
  }
  return { rows, malformed };
}

// ------------------------------------------------------------------ orders
function toOrderRow(raw: unknown): OrderRow | null {
  if (!isObj(raw)) return null;
  const coid = str(raw.client_order_id);
  const state = str(raw.state);
  if (coid === null || state === null) return null;
  const row: OrderRow = {
    client_order_id: coid,
    broker_order_id: str(raw.broker_order_id),
    intent_id: str(raw.intent_id),
    symbol: str(raw.symbol) ?? "?",
    side: str(raw.side) ?? "?",
    quantity: str(raw.quantity) ?? "",
    filled_quantity: str(raw.filled_quantity) ?? "",
    avg_fill_price: str(raw.avg_fill_price),
    limit_price: str(raw.limit_price),
    state,
    gateway: str(raw.gateway) ?? "?",
    created_at: str(raw.created_at) ?? "",
    updated_at: str(raw.updated_at) ?? "",
  };
  const label = str(raw.execution_label);
  if (label !== null) row.execution_label = label;
  const causal = str(raw.causal_id);
  if (causal !== null) row.causal_id = causal;
  return row;
}

/** Keep the incoming row unless it is provably older than what we have (out-of-order delivery). */
function mergeOrder(existing: OrderRow | undefined, incoming: OrderRow): OrderRow {
  if (!existing) return incoming;
  const a = parseUtc(existing.updated_at);
  const b = parseUtc(incoming.updated_at);
  if (a !== null && b !== null && b < a) return existing;
  return {
    ...existing,
    ...incoming,
    execution_label: incoming.execution_label ?? existing.execution_label,
    causal_id: incoming.causal_id ?? existing.causal_id ?? null,
  };
}

function trimOrders(orders: Record<string, OrderRow>): Record<string, OrderRow> {
  const list = Object.values(orders);
  if (list.length <= MAX_ORDERS) return orders;
  list.sort((x, y) => (parseUtc(y.created_at) ?? 0) - (parseUtc(x.created_at) ?? 0));
  return Object.fromEntries(list.slice(0, MAX_ORDERS).map((o) => [o.client_order_id, o]));
}

/** Latest orders first (by created_at, then id for stability). */
export function latestOrders(state: LiveOpsState, n = 3): OrderRow[] {
  return Object.values(state.orders)
    .sort((x, y) => (parseUtc(y.created_at) ?? 0) - (parseUtc(x.created_at) ?? 0) || (x.client_order_id < y.client_order_id ? 1 : -1))
    .slice(0, n);
}

function toOrderEventItem(raw: unknown, seqFromRow: boolean): OrderEventItem | null {
  if (!isObj(raw)) return null;
  const eventId = str(raw.event_id);
  const coid = str(raw.client_order_id);
  if (eventId === null || coid === null) return null;
  return {
    eventId,
    seq: seqFromRow ? seqOf(raw.seq) : null,
    clientOrderId: coid,
    eventType: str(raw.event_type) ?? "?",
    state: str(raw.state) ?? "?",
    source: str(raw.source) ?? "?",
    ts: str(raw.received_ts) ?? str(raw.broker_ts),
    applied: raw.applied === true || raw.applied === 1,
    note: str(raw.apply_note),
  };
}

function orderEventSortKey(e: OrderEventItem): number {
  return parseUtc(e.ts) ?? 0;
}

function upsertOrderEvents(list: readonly OrderEventItem[], add: readonly OrderEventItem[]): OrderEventItem[] {
  const byId = new Map<string, OrderEventItem>();
  for (const e of list) byId.set(e.eventId, e);
  for (const e of add) {
    const prev = byId.get(e.eventId);
    byId.set(e.eventId, prev ? { ...prev, ...e, seq: e.seq ?? prev.seq } : e);
  }
  return [...byId.values()]
    .sort((a, b) => orderEventSortKey(b) - orderEventSortKey(a) || (b.seq ?? 0) - (a.seq ?? 0) || (a.eventId < b.eventId ? 1 : -1))
    .slice(0, MAX_ORDER_EVENTS);
}

// ------------------------------------------------------------------ audit
function toAudit(raw: unknown): AuditEventRow | null {
  if (!isObj(raw)) return null;
  const seq = seqOf(raw.seq);
  const ts = str(raw.ts);
  const type = str(raw.event_type);
  if (seq === null || ts === null || type === null) return null;
  return {
    seq,
    event_id: str(raw.event_id) ?? `seq-${seq}`,
    ts,
    event_type: type,
    causal_id: str(raw.causal_id),
    source: str(raw.source) ?? "?",
    mode: str(raw.mode) ?? "?",
    policy_version: str(raw.policy_version),
    payload: raw.payload ?? null,
  };
}

function upsertAudit(list: readonly AuditEventRow[], add: readonly AuditEventRow[]): AuditEventRow[] {
  const bySeq = new Map<number, AuditEventRow>();
  for (const a of list) bySeq.set(a.seq, a);
  for (const a of add) bySeq.set(a.seq, a);
  return [...bySeq.values()].sort((a, b) => b.seq - a.seq).slice(0, MAX_AUDIT);
}

// ------------------------------------------------------------------ sessions
const ACTIVE_SESSION_STATES = new Set(["RUNNING", "STOPPING"]);

function toSession(raw: unknown): TradingSession | null {
  if (!isObj(raw)) return null;
  const id = str(raw.session_id);
  const state = str(raw.state);
  if (id === null || state === null) return null;
  return {
    session_id: id,
    state,
    started_at: str(raw.started_at),
    ends_at: str(raw.ends_at),
    ended_at: str(raw.ended_at),
    symbols: Array.isArray(raw.symbols) ? raw.symbols.filter((s): s is string => typeof s === "string") : [],
    cadence_minutes: intOrNull(raw.cadence_minutes),
    use_ai: raw.use_ai === true,
    requested_by: str(raw.requested_by),
    cycles_run: intOrNull(raw.cycles_run) ?? 0,
    trades: intOrNull(raw.trades) ?? 0,
    stop_reason: str(raw.stop_reason),
    schedule_id: str(raw.schedule_id),
  };
}

function applySessionEvent(s: SessionsState, raw: unknown): SessionsState {
  const sess = toSession(raw);
  if (!sess) return s;
  const list = [sess, ...s.list.filter((x) => x.session_id !== sess.session_id)];
  let active = s.active;
  if (ACTIVE_SESSION_STATES.has(sess.state)) active = sess;
  else if (active?.session_id === sess.session_id) active = null;
  return { ...s, list, active };
}

// ------------------------------------------------------------------ clock
const MAX_SIM_RATE = 100_000;

export function sampleClock(clock: ServerClock, wallIso: string | null, simIso: string | null | undefined, at: number): ServerClock {
  const wall = parseUtc(wallIso);
  const wallOffsetMs = wall !== null ? wall - at : clock.wallOffsetMs;
  if (simIso === undefined) return { ...clock, wallOffsetMs };
  const simMs = parseUtc(simIso);
  if (simMs === null) return { wallOffsetMs, sim: null };
  const prev = clock.sim;
  let rate = prev?.rate ?? null;
  if (prev && at - prev.clientMs >= 500) {
    const r = (simMs - prev.simMs) / (at - prev.clientMs);
    rate = Number.isFinite(r) ? Math.min(MAX_SIM_RATE, Math.max(0, r)) : rate;
  } else if (prev && at - prev.clientMs < 500) {
    return { wallOffsetMs, sim: { ...prev, simMs: Math.max(prev.simMs, simMs) } };
  }
  return { wallOffsetMs, sim: { simMs, clientMs: at, rate } };
}

/**
 * "Now" on the backend's activity clock (simulated time in SIMULATED feed mode, else the server
 * wall clock corrected for client skew). Used for `since` timers and session time remaining.
 */
export function serverNow(clock: ServerClock, clientNow: number): number {
  if (clock.sim) return clock.sim.simMs + (clock.sim.rate ?? 0) * Math.max(0, clientNow - clock.sim.clientMs);
  return clientNow + (clock.wallOffsetMs ?? 0);
}

// ------------------------------------------------------------------ reducer
function malformed(state: LiveOpsState, n = 1): LiveOpsState {
  return n === 0 ? state : { ...state, counters: { ...state.counters, malformed: state.counters.malformed + n } };
}

function applyStream(state: LiveOpsState, name: StreamEventName, data: unknown, at: number): LiveOpsState {
  switch (name) {
    case "stage":
    case "activity":
    case "decision": {
      const row = normalizeRow(data, name);
      return row ? applyLiveRow(state, row, at) : malformed(state);
    }
    case "cycle": {
      if (!isObj(data) || typeof data.causal_id !== "string" || typeof data.outcome !== "string") return malformed(state);
      const summary = data as unknown as CycleSummary;
      const matches = state.pipeline.causalId === null || state.pipeline.causalId === summary.causal_id;
      // the backend clears its current task when a cycle ends (ActivityLog.end_cycle)
      return {
        ...state,
        lastCycle: summary,
        currentTask: matches ? null : state.currentTask,
        pipeline: matches && state.pipeline.phase !== "none" ? { ...state.pipeline, phase: "complete" } : state.pipeline,
      };
    }
    case "order": {
      const row = toOrderRow(data);
      if (!row) return malformed(state);
      const orders = trimOrders({ ...state.orders, [row.client_order_id]: mergeOrder(state.orders[row.client_order_id], row) });
      const orderCausal = row.causal_id ? { ...state.orderCausal, [row.client_order_id]: row.causal_id } : state.orderCausal;
      return { ...state, orders, orderCausal };
    }
    case "order_event": {
      const item = toOrderEventItem(data, true);
      if (!item) return malformed(state);
      if (state.orderEvents.some((e) => e.eventId === item.eventId && e.seq !== null)) {
        return { ...state, counters: { ...state.counters, duplicatesIgnored: state.counters.duplicatesIgnored + 1 } };
      }
      return { ...state, orderEvents: upsertOrderEvents(state.orderEvents, [item]) };
    }
    case "audit": {
      const row = toAudit(data);
      if (!row) return malformed(state);
      if (state.audit.some((a) => a.seq === row.seq)) {
        return { ...state, counters: { ...state.counters, duplicatesIgnored: state.counters.duplicatesIgnored + 1 } };
      }
      return { ...state, audit: upsertAudit(state.audit, [row]) };
    }
    case "session":
      return { ...state, sessions: applySessionEvent(state.sessions, data) };
    case "heartbeat":
      if (!isObj(data)) return malformed(state);
      return { ...state, clock: sampleClock(state.clock, str(data.time), data.simulated_time === undefined ? undefined : str(data.simulated_time), at) };
    case "status":
      if (!isObj(data)) return malformed(state);
      return {
        ...state,
        clock: sampleClock(
          state.clock,
          str(data.server_time),
          isObj(data.feed) && "simulated_time" in data.feed ? str(data.feed.simulated_time) : undefined,
          at,
        ),
      };
    default:
      return state; // reconciliation / failure: shown through REST refreshes
  }
}

export function liveOpsReducer(state: LiveOpsState, action: LiveOpsAction): LiveOpsState {
  switch (action.type) {
    case "stream":
      return applyStream(state, action.name, action.data, action.at);
    case "resyncStart":
      return { ...state, resync: { active: true, buffer: [] } };
    case "resyncAbort": {
      const buffered = state.resync.buffer.map((b) => b.row);
      return { ...state, resync: { active: false, buffer: [] }, lastSeq: Math.max(state.lastSeq, maxSeq(buffered)) };
    }
    case "resyncApply": {
      const { rows: serverRows, malformed: bad } = normalizeList(Array.isArray(action.recent) ? action.recent : []);
      let next = applySnapshot(state, action.current);
      const buffer = [...state.resync.buffer].sort((a, b) => a.row.seq - b.row.seq);
      for (const b of buffer) next = applyEffects(next, b.row, b.at);
      const bufferedRows = buffer.map((b) => b.row);
      return malformed(
        {
          ...next,
          rows: insertRows(serverRows.sort((a, b) => b.seq - a.seq).slice(0, MAX_ROWS), bufferedRows),
          lastSeq: Math.max(maxSeq(serverRows), maxSeq(bufferedRows)),
          resync: { active: false, buffer: [] },
          synced: true,
        },
        bad,
      );
    }
    case "snapshot":
      return applySnapshot(state, action.current);
    case "activityList": {
      const { rows, malformed: bad } = normalizeList(Array.isArray(action.rows) ? action.rows : []);
      return malformed({ ...state, rows: insertRows(state.rows, rows), lastSeq: Math.max(state.lastSeq, maxSeq(rows)), synced: true }, bad);
    }
    case "auditList": {
      const rows = (Array.isArray(action.rows) ? action.rows : []).map(toAudit);
      const ok = rows.filter((r): r is AuditEventRow => r !== null);
      return malformed({ ...state, audit: upsertAudit(state.audit, ok) }, rows.length - ok.length);
    }
    case "ordersList": {
      let orders = { ...state.orders };
      let bad = 0;
      for (const raw of Array.isArray(action.orders) ? action.orders : []) {
        const row = toOrderRow(raw);
        if (!row) {
          bad += 1;
          continue;
        }
        orders[row.client_order_id] = mergeOrder(orders[row.client_order_id], row);
      }
      orders = trimOrders(orders);
      return malformed({ ...state, orders }, bad);
    }
    case "orderDetail": {
      const d = action.detail;
      const row = toOrderRow(d?.order);
      if (!row) return malformed(state);
      const coid = row.client_order_id;
      const causal = (d.intent && typeof d.intent.causal_id === "string" ? d.intent.causal_id : null) ?? row.causal_id ?? null;
      const events = (Array.isArray(d.events) ? d.events : [])
        .map((e) => toOrderEventItem({ ...e, client_order_id: (e as { client_order_id?: string }).client_order_id ?? coid }, false))
        .filter((e): e is OrderEventItem => e !== null);
      return {
        ...state,
        details: { ...state.details, [coid]: d },
        orders: trimOrders({ ...state.orders, [coid]: mergeOrder(state.orders[coid], row) }),
        orderCausal: causal ? { ...state.orderCausal, [coid]: causal } : state.orderCausal,
        orderEvents: upsertOrderEvents(state.orderEvents, events),
      };
    }
    case "sessions": {
      const d = action.data;
      if (!isObj(d)) return malformed(state);
      const list = (Array.isArray(d.sessions) ? d.sessions : []).map(toSession).filter((s): s is TradingSession => s !== null);
      const schedules = Array.isArray(d.schedules) ? (d.schedules.filter(isObj) as unknown as SessionSchedule[]) : [];
      return {
        ...state,
        sessions: { loaded: true, active: toSession(d.active), list, schedules, detail: str(d.detail) },
      };
    }
    case "status":
      return {
        ...state,
        clock: sampleClock(state.clock, action.status.server_time ?? null, action.status.feed && "simulated_time" in action.status.feed ? (action.status.feed.simulated_time ?? null) : undefined, action.at),
      };
  }
}
