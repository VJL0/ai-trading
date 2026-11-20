import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { StatusResponse } from "../api/types";
import { clone, FIXTURES } from "../test/helpers";
import { ModeBannerView } from "./ModeBanner";

const NOW = 1_800_000_000_000;

function statusWith(label: unknown, patch: Partial<StatusResponse> = {}): StatusResponse {
  return { ...(clone(FIXTURES.status) as StatusResponse), execution_label: label as string, ...patch };
}

function banner() {
  return screen.getByTestId("mode-banner");
}

describe("ModeBannerView", () => {
  it("SIMULATED renders the amber simulated banner from the API label", () => {
    render(<ModeBannerView status={statusWith("SIMULATED")} receivedAt={NOW} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "SIMULATED");
    expect(banner()).toHaveClass("mode-simulated");
    expect(banner()).not.toHaveClass("mode-unsafe");
    expect(screen.getByText("SIMULATED EXECUTION")).toBeInTheDocument();
    expect(banner()).toHaveAttribute("role", "status");
  });

  it("PAPER renders the blue paper banner", () => {
    render(<ModeBannerView status={statusWith("PAPER")} receivedAt={NOW} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "PAPER");
    expect(banner()).toHaveClass("mode-paper");
  });

  it("LIVE renders the red banner as an alert", () => {
    render(<ModeBannerView status={statusWith("LIVE")} receivedAt={NOW} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "LIVE");
    expect(banner()).toHaveClass("mode-live");
    expect(banner()).toHaveClass("mode-unsafe");
    expect(banner()).toHaveAttribute("role", "alert");
    expect(screen.getByText(/REAL MONEY/)).toBeInTheDocument();
  });

  it("an unrecognized label shows UNKNOWN (unsafe), never a guess", () => {
    render(<ModeBannerView status={statusWith("LIVE_MAYBE")} receivedAt={NOW} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "UNKNOWN");
    expect(banner()).toHaveClass("mode-unknown");
    expect(banner()).toHaveAttribute("role", "alert");
    expect(screen.getByText(/unrecognized execution label/)).toBeInTheDocument();
  });

  it("no status yet is UNKNOWN and says so", () => {
    render(<ModeBannerView status={null} receivedAt={null} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "UNKNOWN");
    expect(screen.getByText(/status never received/)).toBeInTheDocument();
  });

  it("a stale status is UNKNOWN and shows its age", () => {
    render(<ModeBannerView status={statusWith("SIMULATED")} receivedAt={NOW - 90_000} now={NOW} />);
    expect(banner()).toHaveAttribute("data-label", "UNKNOWN");
    expect(screen.getByText(/STATUS STALE/)).toBeInTheDocument();
    expect(screen.getByText("status 1m 30s ago")).toBeInTheDocument();
  });

  it("shows entries LOCKED from the status", () => {
    render(<ModeBannerView status={statusWith("SIMULATED", { entries_locked: true })} receivedAt={NOW} now={NOW} />);
    expect(screen.getByText("LOCKED")).toBeInTheDocument();
  });
});
