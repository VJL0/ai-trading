/**
 * System health (GET /api/v1/health + client-side stream liveness) and the active trading session
 * (GET /api/v1/sessions + SSE `session`), read-only. Halting is not a dashboard action: it is the AI
 * agent's `request_halt` tool or the operator CLI, shown below as text only.
 */
import { formatAge, parseUtc, useNow } from "../lib/time";
import { healthModel, type HealthLevel } from "../ops/health";
import type { IconName, Tone } from "../ops/labels";
import { tailId } from "../ops/labels";
import { serverNow, type ServerClock, type SessionsState } from "../ops/reducer";
import { useApp } from "../state/AppContext";
import { useResource } from "../state/useResource";
import { Badge, EmptyState, FetchStamp, KeyValues, Panel, Stamp } from "./ui";

export const HALT_HINT = "stop trading: make halt";

const LEVEL_META: Record<HealthLevel, { tone: Tone; icon: IconName }> = {
  OK: { tone: "good", icon: "passed" },
  DEGRADED: { tone: "warn", icon: "warning" },
  DOWN: { tone: "bad", icon: "rejected" },
  DISABLED: { tone: "muted", icon: "skipped" },
  UNKNOWN: { tone: "warn", icon: "pending" },
};

const SESSION_TONE: Record<string, Tone> = {
  RUNNING: "running",
  STOPPING: "warn",
  SCHEDULED: "info",
  COMPLETED: "good",
  STOPPED: "neutral",
  INTERRUPTED: "warn",
  FAILED: "bad",
};

export function SessionInfo({ sessions, clock, clientNow }: { sessions: SessionsState; clock: ServerClock; clientNow: number }) {
  const sim = clock.sim !== null;
  const active = sessions.active;
  const nextSchedule = sessions.schedules
    .filter((s) => s.enabled && s.next_run_at)
    .sort((a, b) => (parseUtc(a.next_run_at) ?? 0) - (parseUtc(b.next_run_at) ?? 0))[0];
  const last = sessions.list.find((s) => s.session_id !== active?.session_id) ?? null;
  if (!sessions.loaded) return <p className="muted">Loading sessions…</p>;
  return (
    <div className="session-info" data-testid="session-info">
      {active ? (
        <>
          <p className="session-head">
            <Badge tone={SESSION_TONE[active.state] ?? "neutral"} testId="session-state">
              {active.state}
            </Badge>{" "}
            session <code title={active.session_id}>{tailId(active.session_id, 10)}</code>
          </p>
          <KeyValues
            rows={[
              [
                "Time remaining",
                (() => {
                  const end = parseUtc(active.ends_at);
                  if (end === null) return "—";
                  const left = (end - serverNow(clock, clientNow)) / 1000;
                  return <span data-testid="session-remaining">{left > 0 ? formatAge(left) : "window ended"}{sim ? " (sim clock)" : ""}</span>;
                })(),
              ],
              ["Cycles run", <span data-testid="session-cycles">{active.cycles_run}</span>],
              ["Trades", active.trades],
              ["Symbols", active.symbols.join(", ") || "—"],
              ["Cadence", active.cadence_minutes !== null ? `every ${active.cadence_minutes} min` : "—"],
              ["AI review", active.use_ai ? "on" : "off (quant only)"],
              ["Requested by", active.requested_by ?? "—"],
              ["Window", <><Stamp ts={active.started_at} sim={sim} /> → <Stamp ts={active.ends_at} sim={sim} /></>],
            ]}
          />
        </>
      ) : (
        <EmptyState title="No active trading session" testId="session-none">
          {sessions.detail ? <p>{sessions.detail}</p> : null}
          {nextSchedule ? (
            <p>
              Next scheduled: <Stamp ts={nextSchedule.next_run_at} full sim={sim} /> ({nextSchedule.weekdays.join(" ")} {nextSchedule.start_time_et} ET, {nextSchedule.duration_minutes} min)
            </p>
          ) : (
            <p>No enabled schedule. Sessions are started by the AI agent (e.g. “Start a 1-hour paper trading session”).</p>
          )}
        </EmptyState>
      )}
      {last && (
        <p className="muted small" data-testid="session-last">
          Last session {tailId(last.session_id)}: {last.state}, {last.cycles_run} cycles, {last.trades} trades
          {last.stop_reason ? ` — ${last.stop_reason}` : ""}
        </p>
      )}
    </div>
  );
}

export function HealthPanel() {
  const { api, status, stream, ops } = useApp();
  const now = useNow(1000);
  const res = useResource("health", (signal) => api.health(signal), { intervalMs: 10_000, refreshOn: ["reconciliation", "failure"] });
  const model = healthModel(res.data, res.error, status, stream, now);
  const overallMeta = LEVEL_META[model.overall];
  return (
    <Panel
      id="health"
      title="System health"
      source="/health · stream liveness · /sessions"
      className="health-panel"
      meta={
        <>
          <Badge tone={overallMeta.tone} icon={overallMeta.icon} testId="health-overall">
            {model.overall}
          </Badge>
          {model.apiOverall && model.apiOverall !== model.overall && <span className="muted small">API reports {model.apiOverall}</span>}
          <FetchStamp fetchedAt={res.fetchedAt} />
        </>
      }
    >
      {model.note && <p className="note small" data-testid="health-note">{model.note}</p>}
      <ul className="health-list" data-testid="health-list">
        {model.rows.map((r) => {
          const m = LEVEL_META[r.level];
          return (
            <li key={r.name} className={`health-row tone-${m.tone}`} data-component={r.name} data-level={r.level}>
              <span className="health-name">{r.label}</span>
              <Badge tone={m.tone} icon={m.icon}>
                {r.level}
              </Badge>
              <span className="health-detail">
                {r.detail}
                {r.source === "status" && <span className="muted"> (from /status)</span>}
              </span>
              {r.lastOk && <Stamp ts={r.lastOk} label="last ok" />}
            </li>
          );
        })}
      </ul>
      <h3 className="subhead">Trading session</h3>
      <SessionInfo sessions={ops.sessions} clock={ops.clock} clientNow={now} />
      <footer className="health-footer" data-testid="halt-hint">
        Read-only view. Halting is done by the AI agent (<code>request_halt</code>) or the operator CLI:{" "}
        <code className="cli-hint">{HALT_HINT}</code>
      </footer>
    </Panel>
  );
}
