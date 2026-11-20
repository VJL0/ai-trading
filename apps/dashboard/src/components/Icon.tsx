/**
 * Inline SVG icons (stroke-based, 24x24). Always decorative (aria-hidden): every icon is paired with a
 * visible text label, so status is never conveyed by shape or color alone.
 */
import type { IconName } from "../ops/labels";

const PATHS: Record<IconName, string> = {
  idle: "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16",
  running: "M12 3a9 9 0 1 1-9 9",
  passed: "M5 12.5l4.5 4.5L19 7.5",
  rejected: "M6 6l12 12M18 6L6 18",
  error: "M12 3l10 18H2zM12 10v5M12 18v.5",
  skipped: "M5 6l7 6-7 6M13 6l7 6-7 6",
  scan: "M10.5 4a6.5 6.5 0 1 0 0 13a6.5 6.5 0 1 0 0-13M15.5 15.5L21 21",
  predict: "M3 20h18M5 16l4-5 4 3 6-8M15 6h4v4",
  evaluate: "M12 4v16M5 8h14M5 8l-3 6h6zM19 8l-3 6h6z",
  gpt_call: "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9zM18 16l.8 2.2L21 19l-2.2.8L18 22l-.8-2.2L15 19l2.2-.8z",
  risk: "M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z",
  reject: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M5.6 5.6l12.8 12.8",
  submit: "M3 11l18-8-8 18-2-8z",
  fill: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M7.5 12.5l3 3 6-6.5",
  reconcile: "M4 10a8 8 0 0 1 14-4l2 2M20 4v4h-4M20 14a8 8 0 0 1-14 4l-2-2M4 20v-4h4",
  session: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M12 7v5l3.5 2",
  agent_tool: "M9 3h6M12 3v3M6 6h12a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2zM9 11v2M15 11v2M9 16h6",
  info: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M12 11v6M12 7.5v.5",
  audit: "M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7",
  order: "M4 7h16M4 12h16M4 17h10",
  decision: "M4 5h16v10H9l-5 4z",
  warning: "M12 3l10 18H2zM12 10v5M12 18v.5",
  canceled: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M8 12h8",
  pending: "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16M12 8v4",
};

export function Icon({ name, className, size = 16 }: { name: IconName; className?: string; size?: number }) {
  return (
    <svg
      className={`icon icon-${name}${className ? ` ${className}` : ""}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
