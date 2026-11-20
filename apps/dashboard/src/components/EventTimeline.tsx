/**
 * Recent event timeline from the backend ledger: /activity (stage, activity, decision rows) +
 * /audit + order events, newest first, grouped by causal id, capped (ops/timeline.ts explains the
 * ordering rule). Groups are native <details> elements: keyboard operable, no buttons.
 */
import { buildTimeline, type TimelineGroup, type TimelineItem } from "../ops/timeline";
import { tailId } from "../ops/labels";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";
import { Badge, EmptyState, Panel, Stamp } from "./ui";

export const OPEN_GROUPS = 2;

function Item({ item, sim }: { item: TimelineItem; sim: boolean }) {
  // audit rows may be wall-clock stamped even in SIMULATED mode: only live-ops/order rows get the sim tag
  const itemSim = sim && item.source !== "audit";
  return (
    <li className={`tl-item tone-${item.tone}`} data-source={item.source}>
      <span className="tl-icon">
        <Icon name={item.icon} size={13} />
      </span>
      <span className="tl-title">{item.title}</span>
      {item.detail && <span className="tl-detail">{item.detail}</span>}
      <span className="tl-meta">
        <span className="tl-source">{item.source.replace("_", " ")}</span>
        {item.actor && <span className="tl-actor">{item.actor}</span>}
        <Stamp ts={item.ts} sim={itemSim} />
      </span>
    </li>
  );
}

function Group({ group, open, sim }: { group: TimelineGroup; open: boolean; sim: boolean }) {
  return (
    <li className={`tl-group kind-${group.kind}`} data-testid="timeline-group" data-causal={group.causalId ?? ""}>
      <details open={open}>
        <summary>
          <span className="tl-group-title">{group.title}</span>
          {group.causalId && (
            <code className="tl-causal" title={group.causalId}>
              {tailId(group.causalId, 10)}
            </code>
          )}
          {group.outcome && <Badge tone={group.outcome.includes("REJECTED") || group.outcome.includes("ERROR") ? "bad" : group.outcome === "NO_TRADE" ? "neutral" : "good"}>{group.outcome}</Badge>}
          <span className="muted small">
            {group.items.length} event{group.items.length === 1 ? "" : "s"}
            {group.ledger.length > 0 ? ` · ${group.ledger.length} ledger record${group.ledger.length === 1 ? "" : "s"}` : ""}
          </span>
          <Stamp ts={group.latestTs} sim={sim} />
        </summary>
        {group.items.length > 0 && (
          <ol className="tl-items" aria-label="Pipeline and activity events, newest first">
            {group.items.map((it) => (
              <Item key={it.key} item={it} sim={sim} />
            ))}
          </ol>
        )}
        {group.ledger.length > 0 && (
          <>
            <p className="tl-sub">Ledger records with this causal id</p>
            <ol className="tl-items tl-ledger" aria-label="Audit and order ledger records">
              {group.ledger.map((it) => (
                <Item key={it.key} item={it} sim={sim} />
              ))}
            </ol>
          </>
        )}
      </details>
    </li>
  );
}

export function EventTimeline() {
  const { ops } = useApp();
  const sim = ops.clock.sim !== null;
  const tl = buildTimeline(ops);
  return (
    <Panel
      id="timeline"
      title="Event timeline"
      source="/activity · /audit · order events"
      className="timeline-panel"
      meta={<span className="muted small">{tl.totalItems} events shown (capped)</span>}
    >
      {sim && (
        <p className="note small" data-testid="timeline-clock-note">
          SIMULATED clock: pipeline, activity and order-ledger events carry simulated market time (tag <span className="sim-tag">sim</span>);
          audit records may carry wall-clock time, so they are grouped by causal id instead of being interleaved by timestamp.
        </p>
      )}
      {tl.groups.length === 0 && tl.unlinkedAudit.length === 0 && tl.unlinkedOrderEvents.length === 0 ? (
        <EmptyState title={ops.synced ? "No events yet" : "Loading events…"} testId="timeline-empty" />
      ) : (
        <div className="timeline-scroll" role="region" aria-label="Event timeline, newest first" tabIndex={0}>
          <ol className="tl-groups" data-testid="timeline-groups">
            {tl.groups.map((g, i) => (
              <Group key={g.key} group={g} open={i < OPEN_GROUPS} sim={sim} />
            ))}
          </ol>
          {tl.unlinkedOrderEvents.length > 0 && (
            <details className="tl-unlinked" data-testid="timeline-unlinked-orders">
              <summary>Order events not linked to a recent cycle ({tl.unlinkedOrderEvents.length})</summary>
              <ol className="tl-items">
                {tl.unlinkedOrderEvents.map((it) => (
                  <Item key={it.key} item={it} sim={sim} />
                ))}
              </ol>
            </details>
          )}
          {tl.unlinkedAudit.length > 0 && (
            <details className="tl-unlinked" data-testid="timeline-unlinked-audit">
              <summary>Audit log not linked to a recent cycle ({tl.unlinkedAudit.length})</summary>
              <ol className="tl-items">
                {tl.unlinkedAudit.map((it) => (
                  <Item key={it.key} item={it} sim={sim} />
                ))}
              </ol>
            </details>
          )}
        </div>
      )}
    </Panel>
  );
}
