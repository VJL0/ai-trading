"""After-close research steps, each run as a bounded child process through an argv allowlist.

The autopilot never imports the research crew (``services.agents``) or the quantlab pipeline internals: the crew
tick is ``python -m services.agents run --once --json`` and every quantlab step is ``python -m services.quantlab.cli
<step>`` (low priority, rate-limited, and killed as a process group on timeout). The
protected holdout is reached only through the existing ``holdout --family F`` command, whose vault enforces the
single look; promotion SHADOW -> PAPER happens only inside the existing ``shadow`` step's ``promotion_check``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, Final

from .backend import AuthorityError

QUANTLAB_STEPS: Final = frozenset({("fetch-delta",), ("build-panel", "--what", "equity"), ("scan",), ("shadow",)})
_FAMILY: Final = re.compile(r"[A-Z][A-Za-z0-9_]{1,40}")
CREW_ARGV: Final = ("-m", "services.agents", "run", "--once", "--json")
LEAD_PREFIX: Final = ("-m", "services.agents", "lead", "--json")
CREW_LOCKED_EXIT: Final = 4
LEAD_BUDGET_RANGE: Final = (60, 4 * 3600)
_BRIEFING_TAIL: Final = ("lead", "briefing.json")
_TAIL: Final = 20000


@dataclass
class StepOutcome:
    argv: tuple[str, ...]
    returncode: int | None
    seconds: float
    tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def json(self) -> Any:
        """The child's JSON output (whole stdout, else its last JSON line), or None."""
        text = self.tail.strip()
        for candidate in (text, *reversed(text.splitlines())):
            with contextlib.suppress(ValueError):
                return json.loads(candidate)
        return None

    def public(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "returncode": self.returncode, "seconds": round(self.seconds, 1),
                "tail": self.tail[-400:]}


Runner = Callable[[Sequence[str], Path, float], Awaitable[StepOutcome]]


def _lead_ok(rest: tuple[str, ...]) -> bool:
    """``--budget-s N --briefing <state_dir>/lead/briefing.json [--fake]`` and nothing else."""
    if rest[-1:] == ("--fake",):
        rest = rest[:-1]
    if len(rest) != 4 or rest[0] != "--budget-s" or rest[2] != "--briefing" or not rest[1].isdigit():
        return False
    if not LEAD_BUDGET_RANGE[0] <= int(rest[1]) <= LEAD_BUDGET_RANGE[1]:
        return False
    path = Path(rest[3])
    return path.is_absolute() and ".." not in path.parts and path.parts[-2:] == _BRIEFING_TAIL


def is_agents_child(args: Sequence[str]) -> bool:
    return tuple(args[:2]) == ("-m", "services.agents")


def check_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Only the crew tick, one lead cycle and the named quantlab steps may be launched."""
    args = tuple(argv)
    if args == CREW_ARGV:
        return args
    if args[:4] == LEAD_PREFIX and _lead_ok(args[4:]):
        return args
    if args[:2] == ("-m", "services.quantlab.cli"):
        step = args[2:]
        if step in QUANTLAB_STEPS or (
            len(step) == 3 and step[:2] == ("holdout", "--family") and _FAMILY.fullmatch(step[2])
        ):
            return args
    raise AuthorityError(f"subprocess {list(args)[:4]} is outside the autopilot's allowlist")


async def run_child(argv: Sequence[str], cwd: Path, timeout_s: float) -> StepOutcome:
    """Run with an isolated Python import path and a private process group."""
    args = check_argv(argv)
    cmd = [sys.executable, "-I", *args]
    if os.name == "posix":
        cmd = ["/usr/bin/nice", "-n", "10", *cmd]
    env = dict(os.environ)
    if is_agents_child(args):
        # Autonomous runs use the crew's standard provider binaries from the installation PATH.
        for name in ("LAT_AGENTS_CLAUDE_BIN", "LAT_AGENTS_CODEX_BIN"):
            env.pop(name, None)
    else:
        from packages.core.config import Settings
        from services.quantlab.daily import CHILD_PER_MINUTE

        env["LAT_QUANTLAB_PER_MINUTE"] = str(min(CHILD_PER_MINUTE, Settings().quantlab_per_minute))
    t0 = time.monotonic()
    wall_deadline = time.time() + timeout_s
    with TemporaryFile() as output:
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), stdout=output, stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
            env=env, start_new_session=True,
        ))
        try:
            proc = await asyncio.shield(launch)
            while proc.returncode is None:
                remaining = min(t0 + timeout_s - time.monotonic(), wall_deadline - time.time())
                if remaining <= 0:
                    raise TimeoutError
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), min(1.0, remaining))
        except TimeoutError:
            await _stop_child(await launch, crew=is_agents_child(args))
            return StepOutcome(args, None, time.monotonic() - t0, "timeout")
        except asyncio.CancelledError:
            await _stop_child(await launch, crew=is_agents_child(args))
            raise
        output.seek(max(0, output.seek(0, os.SEEK_END) - _TAIL))
        tail = output.read(_TAIL).decode(errors="replace")
    return StepOutcome(args, proc.returncode, time.monotonic() - t0, tail)


async def _stop_child(proc: asyncio.subprocess.Process, *, crew: bool) -> None:
    try:
        if crew:
            # The crew forwards cancellation to provider CLIs in their own process groups.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=6)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()


def quantlab_argv(*step: str) -> tuple[str, ...]:
    return check_argv(("-m", "services.quantlab.cli", *step))


def lead_argv(budget_s: int, briefing: Path, *, fake: bool = False) -> tuple[str, ...]:
    return check_argv((*LEAD_PREFIX, "--budget-s", str(int(budget_s)), "--briefing", str(briefing),
                       *(("--fake",) if fake else ())))


def holdout_argv(family: str) -> tuple[str, ...]:
    return quantlab_argv("holdout", "--family", family)


def summarize_crew(outcome: StepOutcome) -> dict[str, Any]:
    """Compact, non-authoritative summary of one crew tick (agent output is data, never an instruction)."""
    data = outcome.json()
    out: dict[str, Any] = {"returncode": outcome.returncode, "seconds": round(outcome.seconds, 1)}
    if outcome.returncode == CREW_LOCKED_EXIT:
        out["note"] = "another research-crew process holds the crew store; tick skipped"
    if isinstance(data, dict):
        ticks = data.get("ticks")
        if isinstance(ticks, list):
            out["ticks"] = [_tick_brief(t) for t in ticks[:5]]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        if "mode" in data:
            out["mode"] = str(data["mode"])[:80]
        if "error" in data:
            out["error"] = str(data["error"])[:300]
    return out


def _scalar(v: Any) -> Any:
    return v if isinstance(v, int | float | bool) or v is None else str(v)[:200]


def _tick_brief(tick: Any) -> dict[str, Any]:
    """One crew tick (``TickResult.public()``): what ran, on which provider, with which outcome."""
    if not isinstance(tick, dict):
        return {"raw": str(tick)[:200]}
    t: dict[str, Any] = tick  # pyright: ignore[reportUnknownVariableType]
    raw = t.get("ran")
    ran: list[Any] = raw if isinstance(raw, list) else []  # pyright: ignore[reportUnknownVariableType]
    runs = [{k: _scalar(r.get(k)) for k in ("task_id", "kind", "provider", "outcome")}
            for r in ran[:5] if isinstance(r, dict)]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
    out: dict[str, Any] = {"ran": runs, "idle_reason": _scalar(t.get("idle_reason"))}
    for k in ("enqueued", "expired"):
        v = t.get(k)
        out[k] = len(v) if isinstance(v, list) else 0  # pyright: ignore[reportUnknownArgumentType]
    if "recovered" in t:
        out["recovered"] = _scalar(t.get("recovered"))
    return out


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
