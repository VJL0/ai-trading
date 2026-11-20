"""Simulated market clock for SIMULATED feed mode.

With no market-data credentials the system replays the bundled deterministic SYNTHETIC fixture. Using
the wall clock against historical synthetic data would make every quote stale (and the risk gate would
correctly reject everything), so a ManualClock is positioned inside a synthetic regular session and
advanced at ``speed`` simulated seconds per real second. Time only moves forward: at a session close
it jumps to the next session's open (an overnight gap), and after the last session it stays closed.
Every API response in this mode is labeled SIMULATED and exposes ``simulated_time``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from packages.core.clock import ManualClock


class SimulationClock(ManualClock):
    def __init__(self, sessions: Sequence[tuple[datetime, datetime]], *, start_session: int = -1,
                 start_offset_min: int = 45, speed: float = 30.0) -> None:
        if not sessions:
            raise ValueError("SimulationClock needs at least one session")
        self.sessions = [(o.astimezone(UTC), c.astimezone(UTC)) for o, c in sessions]
        self.index = start_session % len(self.sessions)
        super().__init__(self.sessions[self.index][0] + timedelta(minutes=start_offset_min))
        self.speed = speed
        self.running = False
        self.finished = False

    @property
    def session(self) -> tuple[datetime, datetime]:
        return self.sessions[self.index]

    def step(self, seconds: float) -> datetime:
        nxt = self.now() + timedelta(seconds=seconds)
        _open, close = self.session
        if nxt < close:
            self.set(nxt)
        elif self.index + 1 < len(self.sessions):
            self.index += 1
            self.set(self.sessions[self.index][0] + timedelta(seconds=1))
        else:
            self.set(close)
            self.finished = True
        return self.now()

    async def run(self, stop: asyncio.Event, on_tick: Callable[[datetime], None] | None = None,
                  tick_s: float = 1.0) -> None:
        self.running = True
        try:
            while not stop.is_set() and not self.finished:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=tick_s)
                if stop.is_set():
                    break
                self.step(tick_s * self.speed)
                if on_tick is not None:
                    on_tick(self.now())
        finally:
            self.running = False
