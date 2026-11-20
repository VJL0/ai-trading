import type { StreamInfo } from "../api/stream";
import { formatAge, useNow } from "../lib/time";
import { useApp } from "../state/AppContext";

/** Event-stream liveness (client side). Reconnects automatically with backoff; there is no control here. */
export function StreamIndicatorView({ info, now }: { info: StreamInfo; now: number }) {
  const open = info.state === "open";
  const lastAge = info.lastEventAt === null ? null : Math.max(0, (now - info.lastEventAt) / 1000);
  const retryIn = info.nextRetryAt === null ? null : Math.max(0, (info.nextRetryAt - now) / 1000);
  let text: string;
  switch (info.state) {
    case "open":
      text = `live · last event ${lastAge === null ? "—" : `${formatAge(lastAge)} ago`}`;
      break;
    case "connecting":
      text = info.attempt > 0 ? `reconnecting (attempt ${info.attempt + 1})…` : "connecting…";
      break;
    case "disconnected":
      text = `disconnected${retryIn !== null ? ` · retry in ${formatAge(retryIn)}` : ""} · polling`;
      break;
    case "closed":
      text = "stream off · polling only";
      break;
    default:
      text = "stream idle";
  }
  return (
    <div
      className={`stream-indicator ${open ? "stream-open" : "stream-down"}`}
      data-testid="stream-indicator"
      data-state={info.state}
      role="status"
      aria-live="polite"
      title={info.lastError ?? undefined}
    >
      <span className="dot" aria-hidden="true" />
      <span>event stream {text}</span>
    </div>
  );
}

export function StreamIndicator() {
  const { stream } = useApp();
  const now = useNow();
  return <StreamIndicatorView info={stream} now={now} />;
}
