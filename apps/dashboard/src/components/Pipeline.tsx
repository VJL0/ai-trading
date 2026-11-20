/**
 * Central animated decision pipeline:
 *   Market Data -> Features -> Models -> GPT Review -> Risk Engine -> Order Decision -> Broker -> Reconciliation
 * Driven by `stage` SSE events (initial state from GET /api/v1/activity/current). Running stages pulse,
 * edges carry flow particles while the cycle advances; every status has an icon AND a text label.
 * Under prefers-reduced-motion the CSS animations are off and no particles are rendered.
 */
import type { CycleSummary } from "../api/types";
import { usePrefersReducedMotion } from "../lib/motion";
import { formatAge, formatDurationMs, parseUtc, useNow } from "../lib/time";
import { STAGE_LABEL, STAGE_STATUS_META, tailId } from "../ops/labels";
import { isTerminalStage, serverNow, type PipelineState, type ServerClock, type StageState } from "../ops/reducer";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";
import { Badge, Panel, Stamp } from "./ui";

export type EdgeState = "idle" | "flowing" | "done" | "blocked";

export function edgeState(from: StageState, to: StageState, phase: PipelineState["phase"]): EdgeState {
  const fromDone = isTerminalStage(from.status);
  if (from.status === "rejected" || from.status === "error") return isTerminalStage(to.status) ? "blocked" : to.status === "running" ? "flowing" : "blocked";
  if (fromDone && to.status === "running") return "flowing";
  if (fromDone && isTerminalStage(to.status)) return "done";
  if (fromDone && to.status === "idle" && phase === "running") return "flowing";
  return "idle";
}

function stageTiming(s: StageState, clientNow: number, clock: ServerClock): string {
  if (s.status === "running") {
    if (s.observedAt !== null) return `running ${formatAge(Math.max(0, (clientNow - s.observedAt) / 1000))}`;
    const started = parseUtc(s.startedAt);
    if (started !== null && clock.sim === null) return `running ${formatAge(Math.max(0, (serverNow(clock, clientNow) - started) / 1000))}`;
    return "running";
  }
  if (s.durationMs !== null && isTerminalStage(s.status)) return formatDurationMs(s.durationMs);
  return "";
}

export function StageNode({ s, clientNow, clock }: { s: StageState; clientNow: number; clock: ServerClock }) {
  const meta = STAGE_STATUS_META[s.status];
  const timing = stageTiming(s, clientNow, clock);
  const label = `${STAGE_LABEL[s.stage]}: ${meta.label}${s.item ? `, ${s.item}` : ""}${timing ? `, ${timing}` : ""}`;
  return (
    <div className={`stage-node tone-${meta.tone}`} data-status={s.status} aria-label={label} title={s.detail || undefined}>
      {/* keyed by version so the "settle" animation replays on every transition */}
      <span className="stage-ring" key={s.version} aria-hidden="true" />
      <span className="stage-icon">
        <Icon name={meta.icon} size={18} className={s.status === "running" ? "spin" : undefined} />
      </span>
      <span className="stage-name">{STAGE_LABEL[s.stage]}</span>
      <span className="stage-status" data-testid={`stage-status-${s.stage}`}>
        {meta.label}
      </span>
      <span className="stage-item">{s.item ?? " "}</span>
      <span className="stage-timing">{timing || " "}</span>
      {s.detail && <span className="stage-detail">{s.detail}</span>}
    </div>
  );
}

function Edge({ state, reducedMotion }: { state: EdgeState; reducedMotion: boolean }) {
  return (
    <div className="edge" data-flow={state} aria-hidden="true">
      <span className="edge-line" />
      {state === "flowing" && !reducedMotion && (
        <span className="particles" data-testid="flow-particles">
          <span className="particle" />
          <span className="particle" />
          <span className="particle" />
        </span>
      )}
      <span className="edge-arrow" />
    </div>
  );
}

const PHASE_TEXT: Record<PipelineState["phase"], string> = {
  none: "no decision cycle observed yet",
  running: "cycle in progress",
  complete: "cycle complete",
};

export function PipelineView({
  pipeline,
  clock,
  clientNow,
  reducedMotion,
  lastCycle,
}: {
  pipeline: PipelineState;
  clock: ServerClock;
  clientNow: number;
  reducedMotion: boolean;
  lastCycle: CycleSummary | null;
}) {
  const sim = clock.sim !== null;
  const running = pipeline.stages.find((s) => s.status === "running");
  const outcome = lastCycle && lastCycle.causal_id === pipeline.causalId ? lastCycle.outcome : null;
  const announce = running
    ? `${STAGE_LABEL[running.stage]} running${running.item ? ` for ${running.item}` : ""}`
    : pipeline.phase === "complete"
      ? `Cycle complete${outcome ? `: ${outcome}` : ""}`
      : "";
  return (
    <Panel
      id="pipeline"
      title="Decision pipeline"
      className={`pipeline-panel phase-${pipeline.phase}`}
      source="SSE stage · /activity/current"
      meta={
        <>
          <Badge tone={pipeline.phase === "running" ? "running" : pipeline.phase === "complete" ? "good" : "muted"} testId="pipeline-phase">
            {PHASE_TEXT[pipeline.phase]}
          </Badge>
          {outcome && (
            <Badge tone={outcome === "SUBMITTED" ? "good" : outcome === "NO_TRADE" ? "neutral" : "warn"} testId="pipeline-outcome">
              outcome {outcome}
            </Badge>
          )}
        </>
      }
    >
      <div className="pipeline-meta" data-testid="pipeline-meta">
        {pipeline.cycleId ? (
          <>
            <span>
              cycle <code title={pipeline.cycleId}>{tailId(pipeline.cycleId, 10)}</code>
            </span>
            <Stamp ts={pipeline.startedAt} sim={sim} label="started" />
            <span>{pipeline.sessionId ? <>session <code>{tailId(pipeline.sessionId)}</code></> : "not part of a session"}</span>
            <span className="muted">{pipeline.source === "snapshot" ? "state from /activity/current" : "live from the event stream"}</span>
          </>
        ) : (
          <span className="muted">
            Waiting for the first <code>stage</code> event. Cycles are started by a trading session, the AI agent (MCP tool
            <code>run_quant_cycle</code>) or the operator CLI.
          </span>
        )}
      </div>
      <ol className="pipeline" aria-label="Decision pipeline stages" data-reduced-motion={reducedMotion ? "true" : "false"} data-testid="pipeline-stages">
        {pipeline.stages.map((s, i) => {
          const next = pipeline.stages[i + 1];
          return (
            <li key={s.stage} className="stage" data-stage={s.stage} data-status={s.status} data-testid={`stage-${s.stage}`}>
              <StageNode s={s} clientNow={clientNow} clock={clock} />
              {next && <Edge state={edgeState(s, next, pipeline.phase)} reducedMotion={reducedMotion} />}
            </li>
          );
        })}
      </ol>
      <p className="sr-only" aria-live="polite">
        {announce}
      </p>
    </Panel>
  );
}

export function Pipeline() {
  const { ops } = useApp();
  const clientNow = useNow(1000);
  const reduced = usePrefersReducedMotion();
  return <PipelineView pipeline={ops.pipeline} clock={ops.clock} clientNow={clientNow} reducedMotion={reduced} lastCycle={ops.lastCycle} />;
}
