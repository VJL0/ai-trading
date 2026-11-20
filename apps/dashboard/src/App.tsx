/**
 * Read-only live operations UI. No trading controls: the trading system is operated by the AI
 * agent (Claude Code / Codex via the local MCP server, or Sign in with ChatGPT) and the operator CLI.
 * This page only issues GET requests and subscribes to /api/v1/stream.
 */
import { MarketMycelium } from "./mycelium/MarketMycelium";
import { ResearchPanel } from "./components/ResearchPanel";
import { AgentActivity } from "./components/AgentActivity";
import { DecisionCard } from "./components/DecisionCard";
import { EventTimeline } from "./components/EventTimeline";
import { HealthPanel } from "./components/HealthPanel";
import { ModeBanner } from "./components/ModeBanner";
import { OrderLifecycle } from "./components/OrderLifecycle";
import { PerformanceStrip } from "./components/PerformanceStrip";
import { Pipeline } from "./components/Pipeline";
import { StreamIndicator } from "./components/StreamIndicator";
import { ErrorNote } from "./components/ui";
import { useApp } from "./state/AppContext";

export function App() {
  const { syncError, statusError, status } = useApp();
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>Trading operations</h1>
          <span className="readonly-tag" data-testid="readonly-tag">
            read-only · localhost
          </span>
        </div>
        <StreamIndicator />
      </header>
      <ModeBanner />
      {statusError && !status ? <ErrorNote error={statusError} /> : null}
      {syncError ? <ErrorNote error={syncError} stale /> : null}
      <MarketMycelium />
      <main className="layout">
        <div className="area-performance">
          <PerformanceStrip />
        </div>
        <div className="area-pipeline">
          <Pipeline />
        </div>
        <div className="area-decision">
          <DecisionCard />
          <OrderLifecycle />
        </div>
        <div className="area-activity">
          <AgentActivity />
        </div>
        <div className="area-research">
          <ResearchPanel />
        </div>
        <div className="area-timeline">
          <EventTimeline />
        </div>
        <div className="area-health">
          <HealthPanel />
        </div>
      </main>
      <footer className="app-footer muted small">
        All values come from the local API (127.0.0.1); timestamps are UTC. Simulated and paper results do not predict live fills.
      </footer>
    </div>
  );
}
