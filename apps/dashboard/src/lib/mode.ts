/**
 * Execution-label -> banner mapping. The label is ALWAYS the API's `execution_label`; the dashboard
 * never infers it from the mode, gateway or anything else. Missing, unrecognized or stale status is
 * shown as UNKNOWN and styled as unsafe (fail closed: the operator must not assume "simulated").
 */
import type { StatusResponse } from "../api/types";
import { formatAge } from "./time";

export type BannerTone = "simulated" | "paper" | "live" | "unknown";
export type BannerLabel = "SIMULATED" | "PAPER" | "LIVE" | "UNKNOWN";

export interface BannerModel {
  tone: BannerTone;
  label: BannerLabel;
  headline: string;
  detail: string;
  /** true for LIVE and UNKNOWN: operator attention required */
  unsafe: boolean;
  /** the raw label the API reported (for display when unrecognized/stale) */
  reported: string | null;
}

/** Status pushes arrive every ~2-5 s; after this long without one the label is no longer trusted. */
export const STATUS_STALE_AFTER_S = 20;

const KNOWN: Record<string, Omit<BannerModel, "reported" | "detail"> & { detail: string }> = {
  SIMULATED: {
    tone: "simulated",
    label: "SIMULATED",
    headline: "SIMULATED EXECUTION",
    detail: "Deterministic fake broker / replay data. No order reaches any broker.",
    unsafe: false,
  },
  PAPER: {
    tone: "paper",
    label: "PAPER",
    headline: "PAPER TRADING",
    detail: "Alpaca paper endpoint: broker-side simulated fills, no real money.",
    unsafe: false,
  },
  LIVE: {
    tone: "live",
    label: "LIVE",
    headline: "LIVE — REAL MONEY",
    detail: "Orders can reach a real-money brokerage account.",
    unsafe: true,
  },
};

export function bannerModel(
  status: Pick<StatusResponse, "execution_label" | "mode" | "gateway" | "live_execution_compiled"> | null,
  receivedAtMs: number | null,
  nowMs: number,
): BannerModel {
  if (!status) {
    return {
      tone: "unknown",
      label: "UNKNOWN",
      headline: "EXECUTION MODE UNKNOWN",
      detail: "No status received from the local API. Treat the system as unsafe until the mode is confirmed.",
      unsafe: true,
      reported: null,
    };
  }
  const reported = typeof status.execution_label === "string" ? status.execution_label : null;
  const known = reported !== null ? KNOWN[reported] : undefined;
  if (!known) {
    return {
      tone: "unknown",
      label: "UNKNOWN",
      headline: "EXECUTION MODE UNKNOWN",
      detail: `The API reported an unrecognized execution label (${reported === null ? "missing" : JSON.stringify(reported)}). Treated as unsafe.`,
      unsafe: true,
      reported,
    };
  }
  const age = receivedAtMs === null ? null : (nowMs - receivedAtMs) / 1000;
  if (age === null || age > STATUS_STALE_AFTER_S) {
    return {
      tone: "unknown",
      label: "UNKNOWN",
      headline: "EXECUTION MODE UNKNOWN (STATUS STALE)",
      detail: `Last status ${age === null ? "time unknown" : `${formatAge(age)} ago`} reported ${reported}; no fresh confirmation. Treated as unsafe.`,
      unsafe: true,
      reported,
    };
  }
  const extras: string[] = [];
  if (status.live_execution_compiled) extras.push("WARNING: API reports live execution is compiled in.");
  return {
    ...known,
    detail: [known.detail, ...extras].join(" "),
    unsafe: known.unsafe || status.live_execution_compiled === true,
    reported,
  };
}
