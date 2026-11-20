import { describe, expect, it } from "vitest";
import type { StatusResponse } from "../api/types";
import { bannerModel, STATUS_STALE_AFTER_S } from "./mode";

type S = Pick<StatusResponse, "execution_label" | "mode" | "gateway" | "live_execution_compiled">;
const NOW = 1_800_000_000_000;
const st = (execution_label: unknown, extra: Partial<S> = {}): S =>
  ({ execution_label, mode: "PAPER", gateway: "FAKE", live_execution_compiled: false, ...extra }) as S;

describe("bannerModel", () => {
  it("maps the API's execution_label: SIMULATED amber, PAPER blue, LIVE red", () => {
    expect(bannerModel(st("SIMULATED"), NOW, NOW)).toMatchObject({ label: "SIMULATED", tone: "simulated", unsafe: false });
    expect(bannerModel(st("PAPER"), NOW, NOW)).toMatchObject({ label: "PAPER", tone: "paper", unsafe: false });
    expect(bannerModel(st("LIVE"), NOW, NOW)).toMatchObject({ label: "LIVE", tone: "live", unsafe: true });
  });

  it("never infers the label from mode or gateway", () => {
    // mode says PAPER and gateway FAKE, but the label the API reports wins
    expect(bannerModel(st("LIVE", { mode: "PAPER", gateway: "FAKE" }), NOW, NOW).label).toBe("LIVE");
    // mode says LIVE_APPROVAL, label SIMULATED -> SIMULATED (the API is the authority)
    expect(bannerModel(st("SIMULATED", { mode: "LIVE_APPROVAL" }), NOW, NOW).label).toBe("SIMULATED");
  });

  it("treats a missing, unknown or differently-cased label as UNKNOWN and unsafe", () => {
    for (const label of [undefined, null, "", "paper", "Live", "SIM", "REPLAY", 1]) {
      const m = bannerModel(st(label), NOW, NOW);
      expect(m.label).toBe("UNKNOWN");
      expect(m.tone).toBe("unknown");
      expect(m.unsafe).toBe(true);
    }
  });

  it("is UNKNOWN when no status was ever received", () => {
    expect(bannerModel(null, null, NOW)).toMatchObject({ label: "UNKNOWN", unsafe: true });
  });

  it("falls back to UNKNOWN when the status is stale", () => {
    const fresh = bannerModel(st("SIMULATED"), NOW - (STATUS_STALE_AFTER_S - 1) * 1000, NOW);
    expect(fresh.label).toBe("SIMULATED");
    const stale = bannerModel(st("SIMULATED"), NOW - (STATUS_STALE_AFTER_S + 1) * 1000, NOW);
    expect(stale.label).toBe("UNKNOWN");
    expect(stale.headline).toMatch(/STALE/);
    expect(stale.reported).toBe("SIMULATED");
  });

  it("flags live_execution_compiled=true as unsafe even for SIMULATED", () => {
    const m = bannerModel(st("SIMULATED", { live_execution_compiled: true }), NOW, NOW);
    expect(m.unsafe).toBe(true);
    expect(m.detail).toMatch(/compiled in/);
  });
});
