/**
 * Animated order lifecycle for the latest order(s): Proposed -> Validated -> Authorized -> Submitted
 * -> Acknowledged -> Partially filled -> Filled / Rejected / Canceled, from /api/v1/orders,
 * /api/v1/orders/{id} and the `order` / `order_event` SSE events. UNKNOWN is a warning state.
 */
import type { CSSProperties } from "react";
import type { OrderDetailResponse, OrderRow } from "../api/types";
import { formatPrice, formatQty } from "../lib/decimal";
import type { IconName, Tone } from "../ops/labels";
import { lifecycleOf, type LifecycleStep, type StepStatus } from "../ops/lifecycle";
import { latestOrders } from "../ops/reducer";
import { LIFECYCLE_ORDERS } from "../state/useLiveOpsSync";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";
import { Badge, EmptyState, Panel, Stamp } from "./ui";

const STEP_META: Record<StepStatus, { icon: IconName; tone: Tone; text: string }> = {
  done: { icon: "passed", tone: "good", text: "done" },
  current: { icon: "running", tone: "running", text: "in progress" },
  pending: { icon: "pending", tone: "muted", text: "pending" },
  skipped: { icon: "skipped", tone: "neutral", text: "skipped" },
  failed: { icon: "rejected", tone: "bad", text: "failed" },
  canceled: { icon: "canceled", tone: "neutral", text: "canceled" },
  warning: { icon: "warning", tone: "warn", text: "warning" },
};

function Step({ step, sim }: { step: LifecycleStep; sim: boolean }) {
  const meta = STEP_META[step.status];
  return (
    <li className={`lc-step tone-${meta.tone}`} data-step={step.id} data-status={step.status} data-implied={step.implied ? "true" : "false"} title={step.note ?? undefined}>
      <span className="lc-dot" key={step.status}>
        <Icon name={meta.icon} size={14} className={step.status === "current" ? "spin" : undefined} />
      </span>
      <span className="lc-label">{step.label}</span>
      <span className="lc-state">
        {meta.text}
        {step.implied ? " (implied)" : ""}
      </span>
      {step.at && <Stamp ts={step.at} sim={sim} />}
    </li>
  );
}

export function OrderTrack({ order, detail, sim, primary }: { order: OrderRow; detail: OrderDetailResponse | undefined; sim: boolean; primary: boolean }) {
  const view = lifecycleOf(order, detail);
  const progress = `${Math.round((view.activeIndex / (view.steps.length - 1)) * 100)}%`;
  const warn = view.steps.find((s) => s.status === "warning");
  return (
    <article
      className={`order-track${primary ? " primary" : ""}`}
      data-testid={`lifecycle-${order.client_order_id}`}
      data-state={order.state}
      data-outcome={view.outcome}
      aria-label={`Order ${order.side} ${order.quantity} ${order.symbol}: ${order.state}`}
    >
      <header className="order-head">
        <span className="order-title">
          <strong>
            {order.side} {formatQty(order.quantity)} {order.symbol}
          </strong>{" "}
          {order.limit_price && <span className="muted">limit {formatPrice(order.limit_price)}</span>}
        </span>
        <Badge tone={view.outcome === "FILLED" ? "good" : view.outcome === "REJECTED" || view.outcome === "EXPIRED" ? "bad" : view.outcome === "UNKNOWN" || view.outcome === "RECONCILED" ? "warn" : "info"} testId="order-state">
          {order.state}
        </Badge>
        <Badge tone={order.execution_label === "LIVE" ? "bad" : order.execution_label === "PAPER" ? "info" : "warn"}>{order.execution_label ?? order.gateway}</Badge>
      </header>
      <div className="lc-track" style={{ "--lc-progress": progress } as CSSProperties}>
        <span className="lc-rail" aria-hidden="true">
          <span className="lc-fill" />
        </span>
        <ol className="lc-steps" aria-label="Order lifecycle">
          {view.steps.map((s) => (
            <Step key={s.id} step={s} sim={sim} />
          ))}
        </ol>
      </div>
      {warn && (
        <p className="lc-warning" role="alert">
          <Icon name="warning" size={14} /> {warn.note}
        </p>
      )}
      <p className="order-facts">
        <span>
          filled {formatQty(order.filled_quantity)} / {formatQty(order.quantity)}
        </span>
        {order.avg_fill_price && <span>avg fill {formatPrice(order.avg_fill_price)}</span>}
        <span>gateway {order.gateway}</span>
        <span>
          client id <code title={order.client_order_id}>{order.client_order_id}</code>
        </span>
        <Stamp ts={order.updated_at} sim={sim} label="updated" />
        {!detail && <span className="muted">transitions loading…</span>}
      </p>
    </article>
  );
}

export function OrderLifecycle() {
  const { ops } = useApp();
  const orders = latestOrders(ops, LIFECYCLE_ORDERS);
  const sim = ops.clock.sim !== null;
  return (
    <Panel id="lifecycle" title="Order lifecycle" source="/orders · /orders/{id} · SSE order, order_event" className="lifecycle-panel">
      {orders.length === 0 ? (
        <EmptyState title="No orders yet" testId="lifecycle-empty">
          Orders appear here once the risk engine approves a proposal and a signed intent is recorded.
        </EmptyState>
      ) : (
        <div className="order-tracks">
          {orders.map((o, i) => (
            <OrderTrack key={o.client_order_id} order={o} detail={ops.details[o.client_order_id]} sim={sim} primary={i === 0} />
          ))}
        </div>
      )}
    </Panel>
  );
}
