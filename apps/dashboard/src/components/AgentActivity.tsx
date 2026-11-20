/**
 * Live agent activity: the exact current task (latest `activity` event or /activity/current
 * `current_task`) with a "since" timer, and the rolling list of recent actions with kind icons and
 * actor (system / session:<id> / agent:<name>).
 */
import type { ActivityEvent } from "../api/types";
import { formatAge, parseUtc, useNow } from "../lib/time";
import { activityKindMeta, describeActor } from "../ops/labels";
import { serverNow, type CurrentTask, type LiveOpsState, type ServerClock } from "../ops/reducer";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";
import { Badge, EmptyState, Panel, Stamp } from "./ui";

export const RECENT_ACTIONS = 15;

/** Elapsed time of the current task: real elapsed when observed live, else from the server clock (wall mode only). */
export function taskElapsedS(task: CurrentTask, clock: ServerClock, clientNow: number): number | null {
  if (task.observedAt !== null) return Math.max(0, (clientNow - task.observedAt) / 1000);
  const since = parseUtc(task.since);
  if (since === null || clock.sim !== null) return null;
  return Math.max(0, (serverNow(clock, clientNow) - since) / 1000);
}

function ActorChip({ actor }: { actor: string | null }) {
  const a = describeActor(actor);
  return (
    <span className={`actor actor-${a.kind}`} title={actor ?? "system"}>
      {a.label}
    </span>
  );
}

export function recentActions(ops: LiveOpsState, n = RECENT_ACTIONS): ActivityEvent[] {
  const out: ActivityEvent[] = [];
  for (const r of ops.rows) {
    if (r.event === "activity") out.push(r.data);
    if (out.length >= n) break;
  }
  return out;
}

export function AgentActivityView({ ops, clientNow }: { ops: LiveOpsState; clientNow: number }) {
  const task = ops.currentTask;
  const sim = ops.clock.sim !== null;
  const actions = recentActions(ops);
  const meta = task ? activityKindMeta(task.kind) : null;
  const elapsed = task ? taskElapsedS(task, ops.clock, clientNow) : null;
  return (
    <Panel id="activity" title="Live agent activity" source="SSE activity · /activity" className="activity-panel">
      <div className={`current-task${task ? " is-active" : ""}`} data-testid="current-task" aria-live="polite">
        {task && meta ? (
          <>
            <span className={`task-icon tone-${meta.tone}`}>
              <Icon name={meta.icon} size={20} />
            </span>
            <div className="task-body">
              <p className="task-label">
                Now: <Badge tone={meta.tone}>{meta.label}</Badge> {task.actor && <ActorChip actor={task.actor} />}
              </p>
              <p className="task-message" data-testid="current-task-message">
                {task.message}
              </p>
              <p className="task-since">
                <Stamp ts={task.since} sim={sim} label="since" />
                {elapsed !== null && <span data-testid="task-elapsed"> · for {formatAge(elapsed)}</span>}
              </p>
            </div>
          </>
        ) : (
          <div className="task-body">
            <p className="task-label">
              <Badge tone="muted" icon="idle">
                idle
              </Badge>
            </p>
            <p className="task-message" data-testid="current-task-message">
              {ops.synced ? "No task running. Waiting for the next cycle (trading session, AI agent or operator CLI)." : "Loading current task…"}
            </p>
          </div>
        )}
      </div>
      <h3 className="subhead">Recent actions</h3>
      {actions.length === 0 ? (
        <EmptyState title={ops.synced ? "No activity recorded yet" : "Loading activity…"} testId="activity-empty">
          Every scan, prediction, GPT call, risk check, submission and reconciliation appears here as it happens.
        </EmptyState>
      ) : (
        <ol className="action-list" aria-label="Recent actions, newest first" data-testid="action-list">
          {actions.map((a) => {
            const m = activityKindMeta(a.kind);
            return (
              <li key={a.seq} className={`action tone-${m.tone}`} data-kind={a.kind}>
                <span className="action-icon">
                  <Icon name={m.icon} size={14} />
                </span>
                <span className="action-kind">{m.label}</span>
                <span className="action-message">{a.message}</span>
                <span className="action-meta">
                  <ActorChip actor={a.actor} />
                  <Stamp ts={a.ts} sim={sim} />
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}

export function AgentActivity() {
  const { ops } = useApp();
  const now = useNow(1000);
  return <AgentActivityView ops={ops} clientNow={now} />;
}
