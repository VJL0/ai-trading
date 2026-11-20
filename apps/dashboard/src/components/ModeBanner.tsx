import type { StatusResponse } from "../api/types";
import { bannerModel } from "../lib/mode";
import { formatAge, useNow } from "../lib/time";
import { useApp } from "../state/AppContext";
import { Stamp } from "./ui";

/**
 * Execution label banner: ALWAYS the API's `execution_label` (SIMULATED amber / PAPER blue / LIVE red);
 * no status, an unrecognized label or a stale status shows UNKNOWN (treated as unsafe).
 */
export function ModeBannerView({ status, receivedAt, now }: { status: StatusResponse | null; receivedAt: number | null; now: number }) {
  const model = bannerModel(status, receivedAt, now);
  const age = receivedAt === null ? null : Math.max(0, (now - receivedAt) / 1000);
  const locked = status?.entries_locked === true;
  return (
    <div
      className={`mode-banner mode-${model.tone}${model.unsafe ? " mode-unsafe" : ""}`}
      data-testid="mode-banner"
      data-label={model.label}
      data-tone={model.tone}
      role={model.unsafe ? "alert" : "status"}
      aria-live="polite"
    >
      <span className="mode-label" aria-label={`execution label ${model.label}`}>
        {model.label}
      </span>
      <div className="mode-text">
        <strong className="mode-headline">{model.headline}</strong>
        <span className="mode-detail">{model.detail}</span>
      </div>
      <div className="mode-facts">
        {status && (
          <>
            <span>
              mode <b>{String(status.mode ?? "?")}</b>
            </span>
            <span>
              gateway <b>{String(status.gateway ?? "?")}</b>
            </span>
            <span>
              feed <b>{String(status.feed?.feed_mode ?? "?")}</b>
            </span>
            <span className={locked ? "fact-bad" : "fact-good"} data-testid="entries-state" title={locked ? status.lock_reasons.join("; ") : undefined}>
              new entries <b>{locked ? "LOCKED" : "ALLOWED"}</b>
            </span>
            {status.feed?.simulated_time && <Stamp ts={status.feed.simulated_time} sim label="market clock" />}
          </>
        )}
        <span className="muted">status {age === null ? "never received" : `${formatAge(age)} ago`}</span>
      </div>
      {locked && status && status.lock_reasons.length > 0 && (
        <p className="mode-locks" data-testid="lock-reasons">
          Entries locked: {status.lock_reasons.join(" · ")}
        </p>
      )}
    </div>
  );
}

export function ModeBanner() {
  const { status, statusReceivedAt } = useApp();
  const now = useNow(1000);
  return <ModeBannerView status={status} receivedAt={statusReceivedAt} now={now} />;
}
