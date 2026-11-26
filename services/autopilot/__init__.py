"""Autopilot: a deterministic supervisor that runs research and PAPER sessions every trading day by itself.

See docs/architecture/autopilot.md. It holds only the agent token, calls only an allowlisted set of agent tools
(``start_paper_session`` is the one mutating tool) and launches only allowlisted research subprocesses.
"""
