/** Small presentational building blocks shared by every panel (no interactive controls). */
import type { ReactNode } from "react";
import { errorMessage } from "../api/client";
import { formatAge, formatUtc, formatUtcTime, useNow } from "../lib/time";
import type { IconName, Tone } from "../ops/labels";
import { Icon } from "./Icon";

export function Badge({ tone = "neutral", children, title, icon, testId }: { tone?: Tone; children: ReactNode; title?: string; icon?: IconName; testId?: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title} data-testid={testId}>
      {icon && <Icon name={icon} size={12} />}
      {children}
    </span>
  );
}

export function EmptyState({ title, children, testId }: { title: string; children?: ReactNode; testId?: string }) {
  return (
    <div className="empty-state" role="status" data-testid={testId}>
      <p className="empty-title">{title}</p>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error, stale }: { error: unknown; stale?: boolean }) {
  if (!error) return null;
  return (
    <div className="error-note" role="alert">
      <Icon name="warning" size={14} />
      <span>
        <strong>{stale ? "STALE — refresh failed: " : "Unavailable: "}</strong>
        {errorMessage(error)}
      </span>
    </div>
  );
}

/** "fetched 14:30:05Z · 3s ago" (client clock). Turns amber/red as the age grows. */
export function FetchStamp({ fetchedAt, staleAfterS = 30 }: { fetchedAt: number | null; staleAfterS?: number }) {
  const now = useNow();
  if (fetchedAt === null) return <span className="stamp stamp-none">not fetched yet</span>;
  const age = Math.max(0, (now - fetchedAt) / 1000);
  const cls = age > staleAfterS * 2 ? "stamp-bad" : age > staleAfterS ? "stamp-warn" : "";
  return (
    <span className={`stamp ${cls}`} title={`fetched ${formatUtc(fetchedAt)} (client clock)`}>
      fetched {formatUtcTime(fetchedAt)} · {formatAge(age)} ago
    </span>
  );
}

/** A backend timestamp, always with its zone; `sim` marks timestamps on the simulated market clock. */
export function Stamp({ ts, sim = false, full = false, label }: { ts: string | null | undefined; sim?: boolean; full?: boolean; label?: string }) {
  if (!ts) return <span className="ts muted">{label ? `${label} ` : ""}—</span>;
  return (
    <time className="ts" dateTime={ts} title={`${ts}${sim ? " (simulated market clock)" : ""}`}>
      {label ? `${label} ` : ""}
      {full ? formatUtc(ts) : formatUtcTime(ts)}
      {sim && <span className="sim-tag">sim</span>}
    </time>
  );
}

export function Panel({
  title,
  source,
  meta,
  error,
  children,
  className,
  id,
  testId,
}: {
  title: ReactNode;
  source?: ReactNode;
  meta?: ReactNode;
  error?: unknown;
  children: ReactNode;
  className?: string;
  id: string;
  testId?: string;
}) {
  return (
    <section className={`panel ${className ?? ""}`} aria-labelledby={`${id}-title`} id={id} data-testid={testId ?? id}>
      <header className="panel-header">
        <h2 id={`${id}-title`}>{title}</h2>
        <div className="panel-meta">
          {meta}
          {source && <span className="source-label">source: {source}</span>}
        </div>
      </header>
      <ErrorNote error={error} />
      {children}
    </section>
  );
}

export function KeyValues({ rows, className }: { rows: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={`kv ${className ?? ""}`}>
      {rows.map(([k, v], i) => (
        <div className="kv-row" key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
