import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import { QuantlabPanel } from "./QuantlabPanel";

afterEach(() => vi.restoreAllMocks());
const responses: Record<string, unknown> = {
  "/quantlab/capabilities": { status: "OK", asof: "2026-11-23T19:40:00Z", asset_classes: {
    crypto_spot: { research_eligible: true, execution_eligible: false, api_schema: { status: "OK", detail: "" },
      market_data: { status: "OK", detail: "" }, account_permission: { status: "OK", detail: "" },
      policy_execution: { status: "NOT_IMPLEMENTED", detail: "research only" } } } },
  "/quantlab/research": { status: "OK", holdout: ["2025-11-21", "2026-11-20"], no_edge_note: "Backtests only.",
    families: [{ family: "A_momentum", asset_class: "us_equity", passed_dev_gate: false,
      wf_oos: { sharpe: 0.61, cagr: 0.12, max_drawdown: -0.31 }, benchmark: "SPY",
      benchmark_wf_period: { sharpe: 0.7, cagr: 0.13, max_drawdown: -0.34 }, dsr: 0.4, n_trials: 120,
      spa_p_vs_benchmark: 0.6, alpha_t: 0.3, pbo: 0.4, stress_sharpe_2x_costs: 0.4,
      fail_reasons: ["DSR 0.400 < 0.95"] }] },
  "/quantlab/opportunities": { status: "OK", scan: { asof: "2026-11-20", regime: { label: "RISK_ON" },
    relative_strength: [{ symbol: "AAPL", score: 1, why: "rs" }], unusual_volume: [] },
    selection: { action: "NO_TRADE", reason: "no executable candidate", ranked: [] } },
  "/quantlab/strategies": { status: "OK", strategies: [] },
};

describe("QuantlabPanel", () => {
  it("shows verified capabilities, backtest labels, NO_TRADE and no controls", async () => {
    const get = vi.spyOn(ApiClient.prototype, "get").mockImplementation(
      async (path: string) => responses[path] as never);
    const view = render(<QuantlabPanel />);
    expect(await screen.findByText("NOT_IMPLEMENTED")).toBeInTheDocument();
    expect(screen.getByText(/FAIL · DSR 0.400/)).toBeInTheDocument();
    expect(screen.getByTestId("quantlab-selection")).toHaveTextContent("NO_TRADE");
    expect(screen.getByText(/not realized performance/)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(get).toHaveBeenCalledWith("/quantlab/research", undefined, expect.any(AbortSignal));
    view.unmount();
  });
  it("reports failure instead of inventing results", async () => {
    vi.spyOn(ApiClient.prototype, "get").mockRejectedValue(new Error("offline"));
    const view = render(<QuantlabPanel />);
    expect(await screen.findByRole("status")).toHaveTextContent("offline");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    view.unmount();
  });
});
