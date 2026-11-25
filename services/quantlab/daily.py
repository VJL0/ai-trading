"""After-close quantlab maintenance, scheduled from the API process but executed in an isolated subprocess.

Once per completed XNYS session (``settle`` after the close, early closes included) it runs the same steps as
``make quantlab-daily``: incremental daily SIP bars (``fetch-delta``), the point-in-time equity panel
(``build-panel --what equity``), the market scan + cross-asset selection (``scan``) and the shadow journal
(``shadow``). Research-heavy work never runs inside the latency-sensitive API event loop: each step is a separate
``nice``-d child process with a timeout, so a slow download or a crash cannot block order handling.

State (``experiments/quantlab/daily_state.json``) records the last session completed, so a restart neither repeats
a finished session nor skips one: a missed session is caught up once (the steps are incremental), never replayed
per missed day.

Market-hours safety: the steps run only in the quiet window from the close + ``settle`` until ``quiet_before_open``
before the next session's open. A catch-up that becomes due during pre-market or regular hours waits for the next
close, so research downloads never compete with live paper quotes for the account's 200 requests/minute; the
children are additionally limited to 60 requests/minute or a lower ``LAT_QUANTLAB_PER_MINUTE`` value. Waits are
re-checked against the wall clock at least every ``MAX_WAIT_S`` (a sleeping Mac stops the monotonic clock). A failed session is
retried at most ``MAX_ATTEMPTS`` times.

Nothing here can place orders, change the risk policy or open the protected holdout; the scan and selection
outputs are research artifacts that the risk engine never consumes directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.core.config import Settings
from services.market_data.calendar import get_calendar

log = logging.getLogger("lat.quantlab.daily")

STEPS: tuple[tuple[str, ...], ...] = (("fetch-delta",), ("build-panel", "--what", "equity"), ("scan",), ("shadow",))
STEP_TIMEOUT_S = 2 * 3600
MAX_WAIT_S = 1800.0
MAX_ATTEMPTS = 3
CHILD_PER_MINUTE = 60
QUIET_BEFORE_OPEN = timedelta(minutes=90)


def in_quiet_window(now: datetime, settle: timedelta = timedelta(minutes=30),
                    quiet_before_open: timedelta = QUIET_BEFORE_OPEN) -> bool:
    """True between the latest close + ``settle`` and ``quiet_before_open`` before the next session's open."""
    cal = get_calendar()
    d = cal.ny_date(now)
    if cal.is_session(d) and cal.session_open(d) - quiet_before_open <= now < cal.session_close(d) + settle:
        return False
    nxt = d if cal.is_session(d) and now < cal.session_open(d) else cal.next_session(d)
    return now < cal.session_open(nxt) - quiet_before_open


def due_session(now: datetime, last_done: date | None, settle: timedelta = timedelta(minutes=30)) -> date | None:
    """The most recent session whose close + ``settle`` has passed and that is later than ``last_done``."""
    cal = get_calendar()
    d = cal.ny_date(now)
    if not cal.is_session(d) or cal.session_close(d) + settle > now:
        d = cal.previous_session(d)
    if last_done is not None and d <= last_done:
        return None
    return d


def next_check(now: datetime, settle: timedelta = timedelta(minutes=30)) -> datetime:
    """When the next session becomes due (used to sleep instead of polling)."""
    cal = get_calendar()
    d = cal.ny_date(now)
    if cal.is_session(d) and cal.session_close(d) + settle > now:
        return cal.session_close(d) + settle
    return cal.session_close(cal.next_session(d)) + settle


@dataclass
class StepResult:
    step: str
    returncode: int | None
    seconds: float
    tail: str = ""


Runner = Callable[[Sequence[str], Path, float], "asyncio.Future[StepResult] | Any"]


async def run_step(args: Sequence[str], cwd: Path, timeout_s: float) -> StepResult:
    """One quantlab CLI step as a low-priority child process (killed on timeout)."""
    cmd = [sys.executable, "-m", "services.quantlab.cli", *args]
    if os.name == "posix":
        cmd = ["nice", "-n", "10", *cmd]
    per_minute = min(CHILD_PER_MINUTE, Settings().quantlab_per_minute)
    env = {**os.environ, "LAT_QUANTLAB_PER_MINUTE": str(per_minute)}
    t0 = asyncio.get_running_loop().time()
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, env=env)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return StepResult(" ".join(args), None, asyncio.get_running_loop().time() - t0, "timeout")
    except BaseException:  # cancelled (API shutdown): never leave an orphaned writer behind
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.shield(proc.wait())
        raise
    text = (out or b"").decode(errors="replace")
    return StepResult(" ".join(args), proc.returncode, asyncio.get_running_loop().time() - t0, text[-800:])


@dataclass
class DailyJob:
    root: Path
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    runner: Callable[[Sequence[str], Path, float], Any] = run_step
    on_failure: Callable[[str, str], None] | None = None
    on_done: Callable[[dict[str, Any]], None] | None = None
    steps: tuple[tuple[str, ...], ...] = STEPS
    settle: timedelta = timedelta(minutes=30)

    @property
    def state_path(self) -> Path:
        return self.root / "experiments" / "quantlab" / "daily_state.json"

    def state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def last_done(self) -> date | None:
        v = self.state().get("last_session")
        return date.fromisoformat(v) if v else None

    def _write(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        tmp.replace(self.state_path)

    async def run_once(self) -> dict[str, Any] | None:
        """Run the steps for the due session, if any and only in the quiet window. A failed step stops the sequence;
        the session is NOT marked done and is retried (hourly, at most ``MAX_ATTEMPTS`` times)."""
        now = self.clock()
        session = due_session(now, self.last_done(), self.settle)
        if session is None or not in_quiet_window(now, self.settle):
            return None
        state = self.state()
        attempts = state.get("attempts", {})
        if int(attempts.get(session.isoformat(), 0)) >= MAX_ATTEMPTS:
            return None
        attempts = {session.isoformat(): int(attempts.get(session.isoformat(), 0)) + 1}
        state["attempts"] = attempts
        self._write(state)
        results: list[StepResult] = []
        ok = True
        for args in self.steps:
            if not in_quiet_window(self.clock(), self.settle):  # never run into pre-market / regular hours
                ok = False
                results.append(StepResult(" ".join(args), None, 0.0, "deferred: quiet window ended"))
                break
            res = await self.runner(args, self.root, STEP_TIMEOUT_S)
            results.append(res)
            if res.returncode != 0:
                ok = False
                if self.on_failure:
                    self.on_failure(res.step, f"exit {res.returncode}: {res.tail[-300:]}")
                break
        summary = {"session": session.isoformat(), "ok": ok, "at": self.clock().isoformat(),
                   "steps": [{"step": r.step, "returncode": r.returncode, "seconds": round(r.seconds, 1)}
                             for r in results]}
        state = self.state()
        state["last_attempt"] = summary
        if ok:
            state["last_session"] = session.isoformat()
        self._write(state)
        if ok and self.on_done:
            self.on_done(summary)
        return summary

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                summary = await self.run_once()
            except Exception as exc:  # the loop survives; the failure is reported
                summary = {"ok": False}
                if self.on_failure:
                    self.on_failure("daily", f"{type(exc).__name__}: {exc}")
            now = self.clock()
            wake = next_check(now, self.settle) if summary is None or summary.get("ok") else now + timedelta(hours=1)
            wait_s = min(MAX_WAIT_S, max(60.0, (wake - now).total_seconds()))  # re-check the wall clock
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=wait_s)
