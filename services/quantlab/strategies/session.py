"""Session-timing families on allowlisted ETFs and large caps (5-minute SIP bars, long-only).

Each family answers one question from the research brief — *when* in the session (or across sessions) is there an
after-cost edge? — under the runtime's actual constraints (marketable LIMIT entries with non-fills, no entry in the
last 30 minutes, flatten 10 minutes before the close, a protective stop on every position). See
:mod:`services.quantlab.intraday` for the fill model.

* **H1_intraday_momentum** — Gao, Han, Li & Zhou (2018, JFE) "Market intraday momentum": the first half-hour
  return (previous close -> 10:00) predicts the last half-hour return of SPY; Baltussen, Da, Lammers & Martens (2021,
  JFE) attribute it to gamma hedging. The classic exit at 16:00 is NOT executable here (flatten at 15:50, no entry
  after 15:30), so the family tests executable windows: enter at 10:00/12:00/14:00/15:00/15:25 when the
  move-so-far is positive (or above half its trailing 20-day standard deviation), exit 15:50.
* **H2_vwap_reversion** — intraday overextension below the session VWAP (liquidity-driven reversals; e.g.
  Heston, Korajczyk & Sadka 2010 intraday return patterns), optionally only in a daily uptrend; time exits.
* **H3_opening_gap** — opening gaps on ETFs: fade large gap-downs or follow large gap-ups (z-scored by the
  trailing 60-session gap volatility), decided after the first 5 or 30 minutes.
* **I1_overnight** — Cooper, Cliff & Gulen (2008) and Lou, Polk & Skouras (2019): U.S. equity returns accrue
  mostly overnight. Buy at 15:25, sell after the next open (09:35 or 10:00) — the hypothesis itself — or at 15:50
  of the next session (``next_eod``), which is the only exit the runtime lifecycle can perform today: a 2-session
  plan exits at HOLDING_LIMIT (or a stop), there is no strategy-horizon exit for multi-day plans. All forms
  require overnight authorization in the risk policy (disabled today); see :func:`lifecycle_executable`.
* **I2_turn_of_month** — Lakonishok & Smidt (1988), Etula, Rinne, Suominen & Vaittinen (2020, RFS) "Dash for
  cash": returns concentrate around month-ends. Enter at 15:25 on the k-th last session of the month, exit at 15:50
  on the m-th session of the next month (k + m <= 5 sessions, the policy cap). Also requires overnight authorization.

Signals use only bars that closed by the decision time and daily data through the previous close. Thresholds are
scaled by trailing standard deviations computed from EARLIER sessions only.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..intraday import (
    DailyContext,
    IntradayPanel,
    Leg,
    decision_slot,
    flatten_slot,
    latest_entry_slot,
    slot_of,
    trailing_std_rows,
)

H1 = "H1_intraday_momentum"
H2 = "H2_vwap_reversion"
H3 = "H3_opening_gap"
I1 = "I1_overnight"
I2 = "I2_turn_of_month"
FAMILIES = (H1, H2, H3, I1, I2)
INTRADAY_FAMILIES = (H1, H2, H3)
MULTI_SESSION_FAMILIES = (I1, I2)

UNIVERSES: dict[str, tuple[str, ...]] = {
    "SPY": ("SPY",),
    "index4": ("SPY", "QQQ", "IWM", "DIA"),
    "etf12": ("SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD"),
    "allow24": ("SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD",
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V", "JNJ", "PG", "KO", "XOM"),
}
"""Every universe is a subset of the default execution allowlist (``packages.contracts.policy``)."""

DEFAULT_STOP_BPS = 150.0  # RiskPolicy.default_stop_loss_bps


# ------------------------------------------------------------------------------------------------ variants
def h1_variants() -> Iterator[dict[str, Any]]:
    for entry, sig, thr, uni in itertools.product(("10:00", "12:00", "14:00", "15:00", "15:25"),
                                                  ("first30", "to_entry", "open_to_entry"), (0.0, 0.5),
                                                  ("SPY", "index4", "etf12")):
        if entry == "10:00" and sig == "first30":
            continue  # identical to to_entry at 10:00
        yield {"entry": entry, "signal": sig, "threshold_sd": thr, "universe": uni, "exit": "eod"}


def h2_variants() -> Iterator[dict[str, Any]]:
    for entry, k, regime, hold, uni in itertools.product(("10:30", "12:00", "14:00"), (1.0, 2.0),
                                                         ("none", "sma50"), ("60", "eod"),
                                                         ("index4", "etf12", "allow24")):
        yield {"entry": entry, "k_sd": k, "regime": regime, "hold": hold, "universe": uni}


def h3_variants() -> Iterator[dict[str, Any]]:
    for direction, entry, k, hold, uni in itertools.product(("fade_down", "follow_up"), ("09:35", "10:00"),
                                                            (1.0, 2.0), ("120", "eod"),
                                                            ("index4", "etf12", "allow24")):
        yield {"direction": direction, "entry": entry, "k_sd": k, "hold": hold, "universe": uni}


def i1_variants() -> Iterator[dict[str, Any]]:
    for exit_clock, filt, uni in itertools.product(("09:35", "10:00", "next_eod"),
                                                   ("always", "sma200", "intraday_down", "intraday_up"),
                                                   ("SPY", "index4", "etf12")):
        yield {"exit": exit_clock, "filter": filt, "universe": uni}


def i2_variants() -> Iterator[dict[str, Any]]:
    for k, m, uni, regime, stop in itertools.product((1, 2, 3), (1, 2, 3), ("SPY", "index4"), ("none", "sma200"),
                                                     (150.0, 300.0)):
        if k + m <= 5:
            yield {"enter_k_last": k, "exit_m_first": m, "universe": uni, "regime": regime, "stop_bps": stop}


VARIANTS = {H1: h1_variants, H2: h2_variants, H3: h3_variants, I1: i1_variants, I2: i2_variants}


def lifecycle_executable(family: str, p: dict[str, Any]) -> tuple[bool, str]:
    """Whether the position lifecycle could perform this variant's exits (independently of policy authorization)."""
    if family == I1 and p.get("exit") != "next_eod":
        return False, ("exit after the next open needs a strategy-horizon exit for multi-day plans (not implemented: "
                       "a 2-session plan exits at 15:50 of the next session)")
    return True, "exits match the position lifecycle"


def holding_sessions(family: str, p: dict[str, Any]) -> int:
    """Sessions spanned by one position (entry session included) — what the runtime's holding request would be."""
    if family in INTRADAY_FAMILIES:
        return 1
    if family == I1:
        return 2
    return int(p["enter_k_last"]) + int(p["exit_m_first"])


# ------------------------------------------------------------------------------------------------ features
class SessionFeatures:
    """Cached point-in-time intraday features for one panel."""

    def __init__(self, p: IntradayPanel, ctx: DailyContext) -> None:
        self.p, self.ctx = p, ctx
        self.cf = p.ffill_close()
        self.pc = p.prev_close(self.cf)
        self.op = p.first_open()
        self._vwap: np.ndarray | None = None
        self._cache: dict[Any, np.ndarray] = {}

    @property
    def vwap(self) -> np.ndarray:
        if self._vwap is None:
            self._vwap = self.p.session_vwap()
        return self._vwap

    def at(self, s: int) -> np.ndarray:
        return self.cf[:, s, :]

    def trailing_sd(self, key: Any, x: np.ndarray, window: int = 20, min_periods: int = 15) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = trailing_std_rows(x, window, min_periods)
        return self._cache[key]

    def full_day_ok(self, s: int) -> np.ndarray:
        """(D,) sessions on which slot ``s`` is a regular bar and an entry after it is still allowed."""
        return np.array([s + 1 <= latest_entry_slot(ls) for ls in self.p.last_slot])


def _legs_from_mask(fe: SessionFeatures, mask: np.ndarray, score: np.ndarray, dec: int, *, hold: str,
                    stop_bps: float = DEFAULT_STOP_BPS, cols: np.ndarray) -> list[Leg]:
    p = fe.p
    legs: list[Leg] = []
    day_ok = fe.full_day_ok(dec)
    for d, k in zip(*np.where(mask & day_ok[:, None]), strict=True):
        ls = int(p.last_slot[d])
        exit_slot = flatten_slot(ls) if hold == "eod" else min(flatten_slot(ls), dec + 1 + int(hold) // 5)
        legs.append(Leg(int(d), int(cols[k]), dec, int(d), int(exit_slot), float(score[d, k]), stop_bps))
    return legs


def _sub(x: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return x[:, cols] if x.ndim == 2 else x[:, :, cols]


# ------------------------------------------------------------------------------------------------ families
def h1_legs(fe: SessionFeatures, prm: dict[str, Any]) -> list[Leg]:
    cols = fe.p.cols(UNIVERSES[prm["universe"]])
    dec = decision_slot(prm["entry"])
    now = _sub(fe.at(dec), cols)
    if prm["signal"] == "first30":
        sig = _sub(fe.at(decision_slot("10:00")), cols) / _sub(fe.pc, cols) - 1
    elif prm["signal"] == "to_entry":
        sig = now / _sub(fe.pc, cols) - 1
    else:
        sig = now / _sub(fe.op, cols) - 1
    sd = fe.trailing_sd(("h1", prm["signal"], prm["entry"], prm["universe"]), sig)
    thr = float(prm["threshold_sd"])
    mask = np.isfinite(sig) & (sig > (thr * sd if thr > 0 else 0.0)) & (np.isfinite(sd) | (thr == 0))
    score = np.where(np.isfinite(sd) & (sd > 0), sig / np.where(sd > 0, sd, 1.0), sig)
    return _legs_from_mask(fe, mask, score, dec, hold=prm.get("exit", "eod"), cols=cols)


def h2_legs(fe: SessionFeatures, prm: dict[str, Any]) -> list[Leg]:
    cols = fe.p.cols(UNIVERSES[prm["universe"]])
    dec = decision_slot(prm["entry"])
    dev = _sub(fe.at(dec), cols) / _sub(fe.vwap[:, dec, :], cols) - 1
    sd = fe.trailing_sd(("h2", prm["entry"], prm["universe"]), dev)
    mask = np.isfinite(dev) & np.isfinite(sd) & (dev < -float(prm["k_sd"]) * sd)
    if prm["regime"] == "sma50":
        mask &= _sub(fe.ctx.above_sma50, cols)
    score = -dev / np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)
    return _legs_from_mask(fe, mask, score, dec, hold=str(prm["hold"]), cols=cols)


def h3_legs(fe: SessionFeatures, prm: dict[str, Any]) -> list[Leg]:
    cols = fe.p.cols(UNIVERSES[prm["universe"]])
    dec = decision_slot(prm["entry"])
    gap = _sub(fe.op, cols) / _sub(fe.pc, cols) - 1
    sd = fe.trailing_sd(("h3", prm["universe"]), gap, window=60, min_periods=40)
    k = float(prm["k_sd"])
    ok = np.isfinite(gap) & np.isfinite(sd) & np.isfinite(_sub(fe.at(dec), cols))
    if prm["direction"] == "fade_down":
        mask = ok & (gap < -k * sd)
        score = -gap / np.where(sd > 0, sd, np.inf)
    else:
        mask = ok & (gap > k * sd)
        score = gap / np.where(sd > 0, sd, np.inf)
    return _legs_from_mask(fe, mask, score, dec, hold=str(prm["hold"]), cols=cols)


def i1_legs(fe: SessionFeatures, prm: dict[str, Any]) -> list[Leg]:
    p = fe.p
    cols = p.cols(UNIVERSES[prm["universe"]])
    dec = decision_slot("15:25")
    next_eod = prm["exit"] == "next_eod"
    exit_slot = -1 if next_eod else slot_of(prm["exit"])
    now = _sub(fe.at(dec), cols)
    intraday = now / _sub(fe.op, cols) - 1
    mask = np.isfinite(now)
    filt = prm["filter"]
    if filt == "sma200":
        mask &= _sub(fe.ctx.above_sma200, cols)
    elif filt == "intraday_down":
        mask &= np.isfinite(intraday) & (intraday < 0)
    elif filt == "intraday_up":
        mask &= np.isfinite(intraday) & (intraday > 0)
    day_ok = fe.full_day_ok(dec)
    legs: list[Leg] = []
    D = len(p.days)
    for d, k in zip(*np.where(mask & day_ok[:, None]), strict=True):
        if d + 1 >= D:
            continue
        xs = flatten_slot(int(p.last_slot[d + 1])) if next_eod else exit_slot
        legs.append(Leg(int(d), int(cols[k]), dec, int(d) + 1, xs, float(-k), DEFAULT_STOP_BPS))
    return legs


def month_positions(p: IntradayPanel) -> tuple[np.ndarray, np.ndarray]:
    """(D,) k-th LAST session of its month (1 = last) and m-th FIRST session (1 = first), from the exchange calendar
    (published in advance, so knowing that a day is the last session of the month is not look-ahead). Months at the
    panel edges are completed with the calendar, not with data availability."""
    from services.market_data.calendar import get_calendar

    cal = get_calendar()
    months: dict[tuple[int, int], list[date]] = {}
    first = np.zeros(len(p.days), dtype=int)
    last = np.zeros(len(p.days), dtype=int)
    for i, ts in enumerate(p.days):
        d = ts.date()
        key = (d.year, d.month)
        if key not in months:
            start = d.replace(day=1)
            nxt = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
            months[key] = cal.sessions_in_range(start, nxt - timedelta(days=1))
        sess = months[key]
        if d in sess:
            idx = sess.index(d)
            first[i], last[i] = idx + 1, len(sess) - idx
    return last, first


def i2_legs(fe: SessionFeatures, prm: dict[str, Any], positions: tuple[np.ndarray, np.ndarray]) -> list[Leg]:
    p = fe.p
    last_k, first_m = positions
    cols = p.cols(UNIVERSES[prm["universe"]])
    dec = decision_slot("15:25")
    k, m = int(prm["enter_k_last"]), int(prm["exit_m_first"])
    D = len(p.days)
    legs: list[Leg] = []
    day_ok = fe.full_day_ok(dec)
    for d in np.where((last_k == k) & day_ok)[0]:
        e = int(d) + k - 1 + m  # k-1 remaining sessions this month, then the m-th session of the next month
        if e >= D or first_m[e] != m:
            continue
        for kk, j in enumerate(cols):
            if not np.isfinite(fe.cf[d, dec, j]):
                continue
            if prm["regime"] == "sma200" and not fe.ctx.above_sma200[d, j]:
                continue
            legs.append(Leg(int(d), int(j), dec, e, flatten_slot(int(p.last_slot[e])), float(-kk),
                            float(prm["stop_bps"])))
    return legs


def legs_for(family: str, fe: SessionFeatures, prm: dict[str, Any],
             positions: tuple[np.ndarray, np.ndarray] | None = None) -> list[Leg]:
    if family == H1:
        return h1_legs(fe, prm)
    if family == H2:
        return h2_legs(fe, prm)
    if family == H3:
        return h3_legs(fe, prm)
    if family == I1:
        return i1_legs(fe, prm)
    if family == I2:
        return i2_legs(fe, prm, positions if positions is not None else month_positions(fe.p))
    raise KeyError(family)


def max_names(prm: dict[str, Any]) -> int:
    return min(4, len(UNIVERSES[str(prm["universe"])]))
