"""Candidate signal engine: deterministic baseline score (+ optional trained model) net of costs.

``QuantSignalEngine.generate`` scores every eligible symbol of the universe at decision time
``asof`` using **only** bars with ``available_at_ts <= asof``, estimates round-trip spread, fees and
slippage with the ``CostModel``, ranks by net edge and marks **at most one** signal as the candidate —
the highest-ranked one whose ``net_edge_bps > min_net_edge_bps`` and whose own data is fresh and liquid.
Otherwise no signal is a candidate and the caller emits ``NO_TRADE``.

Baseline score ("momentum drift v1", deterministic, *not* a fitted model):

* ``mom = ln(C_t / C_{t-L})``, ``sigma`` = stdev of the last ``L`` one-bar log returns;
* ``score = mom / (sigma * sqrt(L))`` (a trend t-statistic);
* ``expected_return_bps = shrinkage * (mom / L) * H * 10_000`` — the trailing per-bar drift projected
  over the horizon ``H`` (bars) and heavily shrunk. This is a heuristic prior, not an estimate with
  demonstrated skill, and ``probability_up`` is left empty (never presented as a probability).

With a trained model (see ``services.research.ml.TrainedSignalModel``) expected return/probability
come from the model; ``probability_calibrated`` is True only when the model's probabilities were
calibrated on held-out data and evaluated (Brier score recorded): the prediction must say so AND the model
object must declare ``calibration_evaluated is True`` (a model merely claiming calibration is UNCALIBRATED).

Fail-closed eligibility checks (any one blocks the candidate, the signal is still reported): stale data,
unknown/insufficient liquidity, an extreme one-bar move or a multi-session hole inside the lookback, a model
horizon mismatch, a non-positive/undefined expected return and - for intraday bars - a horizon that would end
after the scheduled regular-session close (regular session only; no overnight hold is planned).
"""

from __future__ import annotations

import itertools
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
import pandas as pd

from packages.contracts.enums import Side
from packages.contracts.models import QuantSignal
from packages.core.canonical import canonical_json
from packages.core.clock import Clock
from packages.core.ids import new_id
from packages.core.money import D, floor_shares

from .costs import CostModel
from .frames import (
    float_to_dec,
    horizon_bars,
    is_intraday,
    missing_sessions_between,
    point_in_time,
    prefilter_available,
    prepare_bars,
    session_close_utc,
    timeframe_delta,
    trailing_adv_shares,
    utc_ts,
)

if TYPE_CHECKING:
    from packages.core.db import Database

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
BASELINE_STRATEGY_ID = "quant_baseline_momentum_drift_v1"
DEFAULT_MIN_NET_EDGE_BPS = Decimal(5)
DEFAULT_HORIZON_MINUTES = 60


class ModelPrediction(NamedTuple):
    expected_return_bps: float
    score: float
    probability_up: float | None
    probability_calibrated: bool


class SignalModel(Protocol):
    """A trained model usable by the engine. Must only look at the bars it is given.

    Optional attribute ``calibration_evaluated: bool`` - the engine labels probabilities calibrated only when it
    is ``True`` (held-out calibrator evaluated on the frozen test) and the prediction also says so.
    """

    model_version: str
    strategy_id: str
    horizon_bars: int

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None: ...


@dataclass
class SignalReport:
    asof: datetime
    causal_id: str
    signals: list[QuantSignal]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def candidate(self) -> QuantSignal | None:
        cands = [s for s in self.signals if s.is_candidate]
        return cands[0] if cands else None

    @property
    def outcome(self) -> str:
        return "CANDIDATE" if self.candidate is not None else "NO_TRADE"

    @property
    def no_trade_reason(self) -> str | None:
        if self.candidate is not None:
            return None
        if not self.signals:
            return "no eligible symbols: " + "; ".join(f"{k}: {v}" for k, v in sorted(self.skipped.items()))
        top = min(self.signals, key=lambda s: s.rank)
        return f"no candidate: top-ranked {top.symbol}: {top.reason}"


@dataclass
class _Scored:
    symbol: str
    score: float
    expected_bps: float
    probability: float | None
    calibrated: bool
    spread_bps: Decimal
    fees_bps: Decimal
    slippage_bps: Decimal
    net_edge_bps: Decimal
    features_asof: pd.Timestamp
    dataset_version: str
    is_simulated: bool
    blockers: list[str]
    note: str


class QuantSignalEngine:
    def __init__(
        self,
        cost_model: CostModel,
        *,
        min_net_edge_bps: Decimal | int | str = DEFAULT_MIN_NET_EDGE_BPS,
        horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
        model: SignalModel | None = None,
        lookback_bars: int = 20,
        drift_shrinkage: float = 0.25,
        order_notional: Decimal | int | str = Decimal(1000),
        max_data_age: timedelta | None = None,
        min_adv_usd: Decimal | int | str = Decimal(10_000_000),
        clock: Clock | None = None,
        max_abs_bar_log_return: float = 0.25,
        max_missing_sessions: int = 1,
        horizon_within_session: bool = True,
    ) -> None:
        self.cost_model = cost_model
        self.min_net_edge_bps = D(min_net_edge_bps)
        if self.min_net_edge_bps < 0:
            raise ValueError("min_net_edge_bps must be >= 0 (long-only candidates need positive edge)")
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if lookback_bars < 2:
            raise ValueError("lookback_bars must be >= 2")
        if not 0 < drift_shrinkage <= 1:
            raise ValueError("drift_shrinkage must be in (0, 1]")
        self.horizon_minutes = horizon_minutes
        self.model = model
        self.lookback = lookback_bars
        self.shrinkage = drift_shrinkage
        self.order_notional = D(order_notional)
        if self.order_notional <= 0:
            raise ValueError("order_notional must be positive")
        self.max_data_age = max_data_age
        self.min_adv_usd = D(min_adv_usd)
        self.clock = clock
        if not 0 < max_abs_bar_log_return < 5:
            raise ValueError("max_abs_bar_log_return must be in (0, 5)")
        self.max_abs_bar_log_return = max_abs_bar_log_return
        if max_missing_sessions < 0:
            raise ValueError("max_missing_sessions must be >= 0")
        self.max_missing_sessions = max_missing_sessions
        self.horizon_within_session = horizon_within_session

    @property
    def strategy_id(self) -> str:
        return self.model.strategy_id if self.model is not None else BASELINE_STRATEGY_ID

    @property
    def model_version(self) -> str:
        if self.model is not None:
            return self.model.model_version
        return f"baseline:momentum_drift_v1:L{self.lookback}:shrink{self.shrinkage}:{self.cost_model.version}"

    def _max_age(self, timeframe: str) -> timedelta:
        if self.max_data_age is not None:
            return self.max_data_age
        if is_intraday(timeframe):
            return 2 * timeframe_delta(timeframe) + timedelta(minutes=2)
        return timedelta(days=4)  # weekend + holiday tolerance for daily bars

    def generate(
        self,
        bars_df: pd.DataFrame,
        asof: datetime,
        universe: Sequence[str],
        causal_id: str,
        *,
        quote_spreads_bps: Mapping[str, Decimal] | None = None,
    ) -> list[QuantSignal]:
        return self.generate_report(
            bars_df, asof, universe, causal_id, quote_spreads_bps=quote_spreads_bps
        ).signals

    def generate_report(
        self,
        bars_df: pd.DataFrame,
        asof: datetime,
        universe: Sequence[str],
        causal_id: str,
        *,
        quote_spreads_bps: Mapping[str, Decimal] | None = None,
    ) -> SignalReport:
        asof_ts = utc_ts(asof)
        asof_dt = asof_ts.to_pydatetime()
        if not causal_id:
            raise ValueError("causal_id required")
        report = SignalReport(asof=asof_dt, causal_id=causal_id, signals=[])
        # Rows published after asof are dropped BEFORE validation: future data (even malformed) cannot
        # influence, or make fail, a decision taken at asof. Knowable-but-malformed rows still fail closed.
        visible = prefilter_available(bars_df, asof_ts)
        if not visible.empty and "symbol" in visible.columns:
            # only the symbols being decided on are validated: an unrelated symbol's bad row cannot block them
            visible = visible.loc[visible["symbol"].astype(str).isin(sorted({str(u) for u in universe})).to_numpy()]
        if visible.empty:
            for sym in sorted(set(universe)):
                report.skipped[sym] = "no bars available at or before asof"
            return report
        pit = point_in_time(prepare_bars(visible), asof_ts)  # NOTHING after asof is visible below this line
        spreads = dict(quote_spreads_bps or {})
        scored: list[_Scored] = []
        for sym in sorted(set(universe)):
            if not _SYMBOL_RE.match(sym):
                report.skipped[sym] = "invalid symbol"
                continue
            sb = pit.loc[pit["symbol"] == sym].reset_index(drop=True)
            if sb.empty:
                report.skipped[sym] = "no bars available at or before asof"
                continue
            if len(sb) < self.lookback + 2:
                report.skipped[sym] = f"insufficient history ({len(sb)} bars < {self.lookback + 2})"
                continue
            s = self._score(sb, asof_ts, spreads.get(sym))
            if s is None:
                report.skipped[sym] = "baseline score undefined (zero/NaN volatility)"
                continue
            scored.append(s)

        scored.sort(key=lambda x: (-x.net_edge_bps, x.symbol))
        # The candidate is the highest-ranked signal that passes EVERY check (its own data fresh and liquid,
        # positive expected return, horizon match, net edge above threshold). A stale/illiquid symbol with a
        # large computed edge therefore never blocks, nor becomes, the candidate. At most one is selected.
        selected: str | None = next(
            (s.symbol for s in scored if not s.blockers and s.net_edge_bps > self.min_net_edge_bps), None
        )
        generated_at = self.clock.now() if self.clock is not None else asof_dt
        for rank, s in enumerate(scored, start=1):
            blockers = list(s.blockers)
            if s.net_edge_bps <= self.min_net_edge_bps:
                blockers.append(f"net edge {s.net_edge_bps} bps <= threshold {self.min_net_edge_bps} bps")
            if not blockers and s.symbol != selected:
                blockers.append(f"eligible, but higher-ranked eligible {selected} selected (at most one candidate)")
            is_candidate = s.symbol == selected
            assert is_candidate == (not blockers)
            if is_candidate:
                reason = (
                    f"CANDIDATE: net edge {s.net_edge_bps} bps > threshold {self.min_net_edge_bps} bps; {s.note}"
                )
            else:
                reason = "not candidate: " + "; ".join(blockers) + f"; {s.note}"
            report.signals.append(
                QuantSignal(
                    signal_id=new_id("sig"),
                    strategy_id=self.strategy_id,
                    model_version=self.model_version,
                    symbol=s.symbol,
                    side=Side.BUY,
                    horizon_minutes=self.horizon_minutes,
                    score=float_to_dec(s.score, 6),
                    rank=rank,
                    expected_return_bps=float_to_dec(s.expected_bps, 4),
                    probability_up=None if s.probability is None else float_to_dec(s.probability, 6),
                    probability_calibrated=bool(s.calibrated and s.probability is not None),
                    est_spread_bps=s.spread_bps,
                    est_fees_bps=s.fees_bps,
                    est_slippage_bps=s.slippage_bps,
                    net_edge_bps=s.net_edge_bps,
                    is_candidate=is_candidate,
                    reason=reason[:2000],
                    features_asof=s.features_asof.to_pydatetime(),
                    generated_at=generated_at,
                    dataset_version=s.dataset_version,
                    is_simulated=s.is_simulated,
                )
            )
        assert sum(1 for x in report.signals if x.is_candidate) <= 1
        return report

    def _score(self, sb: pd.DataFrame, asof: pd.Timestamp, spread_bps: Decimal | None) -> _Scored | None:
        sym = str(sb["symbol"].iloc[0])
        tf = str(sb["timeframe"].iloc[0])
        closes = sb["close_f"].to_numpy(dtype=np.float64)
        window = closes[-(self.lookback + 1):]
        logret = np.diff(np.log(window))
        sigma = float(np.std(logret, ddof=1))
        mom = float(math.log(window[-1] / window[0]))
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(mom):
            return None
        h = horizon_bars(self.horizon_minutes, tf)
        tstat = mom / (sigma * math.sqrt(self.lookback))
        expected = self.shrinkage * (mom / self.lookback) * h * 10_000.0
        probability: float | None = None
        calibrated = False
        score = tstat
        note = f"baseline momentum t-stat {tstat:.3f} (L={self.lookback}, H={h} bars, uncalibrated heuristic)"
        if self.model is not None:
            pred = self.model.predict_latest(sb)
            if pred is not None and not _valid_prediction(pred):
                note += f"; model output invalid ({pred!r}) -> not eligible"
                pred = None
                expected = float("nan")
            elif pred is None:
                note += "; model produced no prediction -> not eligible"
                expected = float("nan")
            else:
                expected = pred.expected_return_bps
                score = pred.score
                probability = pred.probability_up
                # a calibrated label needs BOTH the prediction's claim and the model's evaluated calibration
                calibrated = bool(pred.probability_calibrated) and getattr(self.model, "calibration_evaluated",
                                                                           False) is True
                note = (
                    f"model {self.model.model_version}: E[r]={expected:.2f} bps"
                    + (f", p_up={probability:.3f} ({'calibrated' if calibrated else 'UNCALIBRATED'})"
                       if probability is not None else "")
                    + f"; baseline t-stat {tstat:.3f}"
                )
                if pred.probability_calibrated and not calibrated:
                    note += "; model claims calibration without an evaluated calibrator -> labelled UNCALIBRATED"
        last = sb.iloc[-1]
        price: Decimal = last["close"]
        adv_raw = float(trailing_adv_shares(sb)[-1])
        adv = Decimal(f"{adv_raw:.0f}") if math.isfinite(adv_raw) and adv_raw > 0 else None
        qty = max(Decimal(1), floor_shares(self.order_notional / price))
        rt = self.cost_model.estimate_round_trip(sym, qty, price, spread_bps=spread_bps, adv_shares=adv)
        blockers: list[str] = []
        age = asof - last["end_ts"]
        max_age = self._max_age(tf)
        if age > max_age:
            blockers.append(f"stale data: last bar ended {age} before asof (max {max_age})")
        if adv is None:
            blockers.append("liquidity unknown (no trailing completed-session ADV): fail closed")
        elif adv * price < self.min_adv_usd:
            blockers.append(f"ADV ${adv * price:.0f} below liquidity floor ${self.min_adv_usd}")
        jump = float(np.max(np.abs(logret)))
        if jump > self.max_abs_bar_log_return:
            blockers.append(
                f"extreme 1-bar move |ln r|={jump:.3f} > {self.max_abs_bar_log_return} in the lookback: suspected bad "
                "print or unadjusted corporate action (fail closed)"
            )
        wdays = list(sb["session_date"].iloc[-(self.lookback + 1):])
        holes = [missing_sessions_between(a, b) for a, b in itertools.pairwise(wdays) if a != b]
        if holes and max(holes) > self.max_missing_sessions:
            blockers.append(
                f"data gap: {max(holes)} XNYS sessions missing inside the lookback (max {self.max_missing_sessions})"
            )
        if self.horizon_within_session and is_intraday(tf):
            close_utc = session_close_utc(last["session_date"])
            if asof + pd.Timedelta(minutes=self.horizon_minutes) > pd.Timestamp(close_utc):
                blockers.append(
                    f"horizon {self.horizon_minutes} min from asof ends after the scheduled session close "
                    f"{close_utc.isoformat()} (regular session only; no overnight hold planned)"
                )
        if self.model is not None and int(self.model.horizon_bars) != h:
            blockers.append(f"model horizon {self.model.horizon_bars} bars != engine horizon {h} bars")
        if not math.isfinite(expected):
            blockers.append("no expected return")
            expected = 0.0
        elif expected <= 0:
            blockers.append("expected return not positive (long-only)")
        exp_dec = float_to_dec(expected, 4)
        net = (exp_dec - rt.total_bps).quantize(Decimal("0.0001"))
        return _Scored(
            symbol=sym,
            score=score,
            expected_bps=expected,
            probability=probability,
            calibrated=calibrated,
            spread_bps=rt.spread_bps,
            fees_bps=rt.fees_bps,
            slippage_bps=rt.slippage_bps,
            net_edge_bps=net,
            features_asof=last["available_at_ts"],
            dataset_version=str(last["dataset_version"]),
            is_simulated=bool(sb["is_synthetic"].any()),
            blockers=blockers,
            note=note + f"; costs {rt.total_bps} bps round trip for {qty} sh (spread source "
            f"{rt.entry.spread_source})",
        )


def _valid_prediction(pred: ModelPrediction) -> bool:
    """A model output is usable only if every number is finite and a probability lies in [0, 1]."""
    if not (math.isfinite(pred.expected_return_bps) and math.isfinite(pred.score)):
        return False
    p = pred.probability_up
    return p is None or (math.isfinite(p) and 0.0 <= p <= 1.0)


def persist_signals(db: Database, signals: Sequence[QuantSignal], causal_id: str) -> int:
    """Insert signals into the core ``quant_signals`` table through the single-writer queue. Returns rows inserted."""
    from packages.core.db import utc_iso  # keep module import light for research use

    rows = [
        (
            s.signal_id,
            causal_id,
            s.strategy_id,
            s.symbol,
            utc_iso(s.generated_at),
            1 if s.is_candidate else 0,
            format(s.net_edge_bps, "f"),
            s.dataset_version,
            canonical_json(s),
        )
        for s in signals
    ]

    def _write(conn: sqlite3.Connection) -> int:
        # idempotent: re-persisting the same signal ids (retry/replay) inserts nothing and never duplicates
        cursor = conn.executemany(
            "INSERT OR IGNORE INTO quant_signals (signal_id, causal_id, strategy_id, symbol, generated_at, is_candidate,"
            " net_edge_bps, dataset_version, payload_json) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        # Connection.total_changes also counts journal/outbox triggers.
        return cursor.rowcount

    return db.write(_write)
