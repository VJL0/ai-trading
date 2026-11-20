import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import { ResearchPanel } from "./ResearchPanel";

afterEach(() => vi.restoreAllMocks());
describe("ResearchPanel", () => {
  it("shows recorded synthetic evidence and no controls", async () => {
    const get = vi.spyOn(ApiClient.prototype, "get").mockResolvedValue({
      counts: { COMPLETED: 1 }, events: [], promotion_status: "RESEARCH_ONLY",
      jobs: [{ id: "a", state: "COMPLETED", spec: { family: "ridge", transform: "identity" }, error: null,
        result: { is_simulated: true, metrics: { net_sharpe: null, n_trades: 0, net_return: 0 } } }],
    });
    const view = render(<ResearchPanel />);
    expect(await screen.findByText("SIMULATED · development only")).toBeInTheDocument();
    expect(screen.getByText("COMPLETED: 1")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(get).toHaveBeenCalledWith("/research/autonomous", undefined, expect.any(AbortSignal));
    view.unmount();
  });
  it("reports feed failure instead of inventing activity", async () => {
    vi.spyOn(ApiClient.prototype, "get").mockRejectedValue(new Error("offline"));
    const view = render(<ResearchPanel />);
    expect(await screen.findByRole("status")).toHaveTextContent("offline");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    view.unmount();
  });
});
