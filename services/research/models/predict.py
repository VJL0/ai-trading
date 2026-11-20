"""Point-in-time predictions from a verified ``ModelRegistry``.

``PredictionService.predict(bars_df, asof, universe)`` drops every row published after ``asof`` BEFORE validating
anything (``frames.prefilter_available``), so data after ``asof`` - even malformed - can neither change nor break a
prediction. A symbol gets NO PredictionSet (it is simply absent from the result) when its data is stale, its history
is too short for the features / HAR window, its liquidity is unknown, it shows a suspect one-bar move, or the bar
timeframe differs from the registry's. Money-like outputs are ``Decimal`` (``frames.float_to_dec``).

Return forecasts are the selected model per horizon; ``expected_return_bps`` is the expected NET return (after the
cost model's round trip, the training target). A zero-baseline selection forecasts exactly 0 -> NO_TRADE.

The service also implements ``services.research.signals.SignalModel`` (``predict_latest``) so
``QuantSignalEngine(cost_model, model=PredictionService(...))`` scores candidates with the selected model for the
engine horizon. The engine subtracts its own round-trip cost, so the adapter passes a GROSS expected return =
net forecast + (round-trip spread + fees + fixed slippage, excluding impact) - conservative: the engine's own impact
term is then charged on top. With a zero-baseline selection the adapter returns 0 and the engine emits NO_TRADE.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from packages.contracts.enums import MarketRegime
from packages.contracts.models import HorizonForecast, PredictionSet
from packages.core.ids import new_id
from packages.core.money import floor_shares

from ..costs import CostModel
from ..frames import (
    BarDataError,
    float_to_dec,
    horizon_bars,
    is_intraday,
    point_in_time,
    prefilter_available,
    prepare_bars,
    timeframe_delta,
    trailing_adv_shares,
    ts_ns,
    utc_ts,
)
from ..signals import ModelPrediction
from .config import (
    CATBOOST_CLF,
    COST_MODEL_FULL,
    MARKOV_2STATE,
    SPREAD_PLUS_RESIDUAL,
    ZERO_BASELINE,
    ret_target,
)
from .features import intra_session_r1, market_blocks, symbol_model_features, vol_inputs
from .registry import ModelRegistry
from .targets import VOL_COLUMNS

log = logging.getLogger(__name__)

STRATEGY_ID = "quant_model_suite_v1"
MAX_ABS_BAR_LOG_RETURN = 0.25


class PredictionService:
    def __init__(
        self,
        registry: ModelRegistry,
        cost_model: CostModel,
        *,
        signal_horizon_minutes: int = 60,
        max_data_age: timedelta | None = None,
    ) -> None:
        self.registry = registry
        self.cost_model = cost_model
        self.timeframe = registry.timeframe
        self.signal_horizon_minutes = signal_horizon_minutes
        if signal_horizon_minutes not in registry.fitted.returns:
            raise ValueError(f"registry has no {signal_horizon_minutes}-minute return model "
                             f"(horizons {sorted(registry.fitted.returns)})")
        self.horizon_bars = horizon_bars(signal_horizon_minutes, self.timeframe)
        self.max_data_age = max_data_age or (2 * timeframe_delta(self.timeframe) + timedelta(minutes=2))
        self.strategy_id = STRATEGY_ID
        self.calibration_evaluated = registry.probability_calibrated
        cfg = registry.meta["config"]
        self.reference_notional = Decimal(str(cfg["reference_notional"]))
        self.success_horizon_minutes = int(cfg["success_horizon_minutes"])
        self.vol_horizon_minutes = int(cfg["vol_horizon_minutes"])
        self.regime_block_minutes = int(cfg["regime_block_minutes"])
        self.regime_symbols: list[str] = sorted(
            str(x) for x in (registry.meta.get("regime_proxy_symbols") or registry.meta.get("symbols") or []))
        sel = registry.selected(ret_target(signal_horizon_minutes))
        self.model_version = (f"msuite:{registry.registry_version}:{sel}:"
                              f"{registry.model_version_of(ret_target(signal_horizon_minutes))}")

    @property
    def registry_version(self) -> str:
        return self.registry.registry_version

    # ------------------------------------------------------------------ features of the last bar
    def _row(self, sb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(feature row, vol-input row, trailing ADV shares) of the LAST bar, or None when incomplete."""
        spec = self.registry.feature_spec
        if len(sb) < spec.warmup_bars + 1:
            return None
        feats = symbol_model_features(sb, spec)
        x = feats[list(self.registry.feature_columns)].iloc[-1].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            return None
        r1 = intra_session_r1(sb)
        look = r1[-max(spec.warmup_bars, 2):]
        if np.isfinite(look).any() and float(np.nanmax(np.abs(look))) > MAX_ABS_BAR_LOG_RETURN:
            return None  # suspect print / unadjusted corporate action inside the lookback: no prediction
        vrow = np.full(len(VOL_COLUMNS), np.nan)
        vol = self.registry.fitted.vol
        if vol is not None:
            vrow = vol_inputs(sb, vol.vs)[list(VOL_COLUMNS)].iloc[-1].to_numpy(dtype=np.float64)
        adv = float(trailing_adv_shares(sb)[-1])
        if not (math.isfinite(adv) and adv > 0):
            return None  # liquidity unknown: fail closed
        return x, vrow, adv

    def _costs(self, symbol: str, price: Decimal, adv: float, spread_bps: Decimal | None, x: np.ndarray
               ) -> tuple[float, float, float, float]:
        """(one-way est cost bps, est slippage bps, round-trip spread+fees+fixed-slippage bps, round-trip bps)."""
        price_q = price.quantize(Decimal("0.01"))
        qty = max(Decimal(1), floor_shares(self.reference_notional / price_q))
        adv_d = Decimal(f"{float(f'{adv:.3g}'):.0f}")
        rt = self.cost_model.estimate_round_trip(symbol, qty, price_q, spread_bps=spread_bps, adv_shares=adv_d)
        entry = rt.entry
        floor = float(entry.spread_cost_bps + entry.fees_bps)
        kind = self.registry.fitted.cost_selected
        if kind == COST_MODEL_FULL:
            est = float(entry.total_bps)
        elif kind == SPREAD_PLUS_RESIDUAL and self.registry.fitted.cost_residual is not None:
            cols = self.registry.feature_columns
            spec = self.registry.feature_spec
            cx = np.asarray([[
                float(entry.quoted_spread_bps),
                x[cols.index(f"rv_{spec.vol_window}")],
                float(entry.participation) if entry.participation is not None else 0.0,
                math.log10(max(adv * float(price_q), 1.0)),
                x[cols.index("tod")],
                x[cols.index(f"ret_{min(spec.return_horizons)}")],
            ]], dtype=np.float64)
            est = floor + max(float(self.registry.fitted.cost_residual.predict(cx)[0]), 0.0)
        else:
            est = floor
        fixed_rt = 2.0 * float(self.cost_model.slippage.fixed_bps)
        rt_no_impact = float(rt.spread_bps + rt.fees_bps) + fixed_rt
        return est, est - floor, rt_no_impact, float(rt.total_bps)

    def _regime(self, visible: pd.DataFrame, asof_ts: pd.Timestamp, prepared: pd.DataFrame | None = None
                ) -> tuple[MarketRegime, float | None, str | None]:
        """Filtered regime at ``asof`` from the TRAINING proxy symbols (same observation as in training).

        UNKNOWN (with a reason) when the Markov model was not selected, a proxy symbol is missing or stale, its
        knowable bars are malformed, or no complete block is recent enough - never a guess.
        """
        fitted = self.registry.fitted
        if fitted.regime is None:
            return MarketRegime.UNKNOWN, None, None
        proxy = self.regime_symbols
        if not proxy:
            return MarketRegime.UNKNOWN, None, "regime UNKNOWN: registry records no proxy symbols"
        src = visible if prepared is None else prepared
        sub = src.loc[src["symbol"].astype(str).isin(proxy).to_numpy()]
        missing = sorted(set(proxy) - set(sub["symbol"].astype(str)))
        if missing:
            return MarketRegime.UNKNOWN, None, f"regime UNKNOWN: proxy symbols without data {missing}"
        try:
            pb = point_in_time(prepare_bars(sub) if prepared is None else sub.reset_index(drop=True), asof_ts)
        except BarDataError as exc:
            return MarketRegime.UNKNOWN, None, f"regime UNKNOWN: proxy bars rejected ({exc})"
        if pb.empty or str(pb["timeframe"].iloc[0]) != self.timeframe:
            return MarketRegime.UNKNOWN, None, "regime UNKNOWN: proxy bars missing or of another timeframe"
        max_age_ns = int(pd.Timedelta(self.max_data_age).value)
        sym_arr = pb["symbol"].astype(str).to_numpy()
        end_arr = ts_ns(pb["end_ts"])
        last_end = {sym: int(end_arr[sym_arr == sym].max()) for sym in proxy if (sym_arr == sym).any()}
        stale = sorted(k for k, v in last_end.items() if int(asof_ts.value) - v > max_age_ns)
        if stale or len(last_end) != len(proxy):
            return MarketRegime.UNKNOWN, None, f"regime UNKNOWN: stale proxy symbols {stale}"
        blocks = market_blocks(pb, self.regime_block_minutes)
        if len(blocks) < 2:
            return MarketRegime.UNKNOWN, None, "regime UNKNOWN: fewer than two complete blocks"
        block_age = int(asof_ts.value) - int(blocks.avail_ns[-1])
        max_block_age = pd.Timedelta(minutes=self.regime_block_minutes) + pd.Timedelta(self.max_data_age)
        if block_age > max_block_age.value:
            return MarketRegime.UNKNOWN, None, "regime UNKNOWN: last complete block is stale"
        p_t = float(fitted.regime.p_turbulent(blocks.ret_bps)[-1])
        regime, pr = fitted.regime.classify(p_t)
        return regime, (pr if math.isfinite(pr) else None), None

    # ------------------------------------------------------------------ public API
    def predict(
        self,
        bars_df: pd.DataFrame,
        asof: datetime,
        universe: Sequence[str],
        *,
        quote_spreads_bps: Mapping[str, Decimal] | None = None,
    ) -> list[PredictionSet]:
        asof_ts = utc_ts(asof)
        wanted = sorted({str(u) for u in universe})
        knowable = prefilter_available(bars_df, asof_ts)  # nothing published after asof survives this line
        if knowable.empty or "symbol" not in knowable.columns:
            return []
        visible = knowable.loc[knowable["symbol"].astype(str).isin(wanted).to_numpy()]
        if visible.empty:
            return []
        try:
            bars = point_in_time(prepare_bars(visible), asof_ts)
        except BarDataError as exc:
            log.warning("predict: knowable bars rejected (%s); no predictions (fail closed)", exc)
            return []
        if bars.empty:
            return []
        tf = str(bars["timeframe"].iloc[0])
        if tf != self.timeframe or not is_intraday(tf):
            log.warning("predict: bars timeframe %s != registry timeframe %s; no predictions", tf, self.timeframe)
            return []
        reg = self.registry
        fitted = reg.fitted
        spreads = dict(quote_spreads_bps or {})
        rows: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray, float]] = []
        for sym in wanted:
            sb = bars.loc[bars["symbol"] == sym].reset_index(drop=True)
            if sb.empty:
                continue
            if asof_ts - sb["end_ts"].iloc[-1] > pd.Timedelta(self.max_data_age):
                continue  # stale
            got = self._row(sb)
            if got is None:
                continue
            rows.append((sym, sb, *got))
        if not rows:
            return []
        X = np.vstack([r[2] for r in rows])
        V = np.vstack([r[3] for r in rows])
        returns = {h: (mid, model.predict(X)) for h, (mid, model) in sorted(fitted.returns.items())}
        p_succ = fitted.success.calibrated(X) if fitted.success is not None else None
        rank_scores = fitted.rank[1].predict(X) if fitted.rank is not None else None
        ranks: list[int | None] = [None] * len(rows)
        if rank_scores is not None:
            order = sorted(range(len(rows)), key=lambda i: (-float(rank_scores[i]), rows[i][0]))
            for pos, i in enumerate(order, start=1):
                ranks[i] = pos
        vol_bps = None
        if fitted.vol is not None:
            var = fitted.vol.forecast(V)
            vol_bps = np.where(np.isfinite(var) & (var > 0), np.sqrt(np.maximum(var, 0.0)) * 10_000.0, np.nan)
        # proxy symbols already validated with the universe -> reuse the prepared frame; else validate them separately
        # (a malformed proxy-only symbol then makes the regime UNKNOWN instead of blocking every prediction)
        regime, regime_p, regime_note = self._regime(
            knowable, asof_ts, prepared=bars if set(self.regime_symbols) <= set(wanted) else None)
        sel_success = reg.selected("success") == CATBOOST_CLF and p_succ is not None
        out: list[PredictionSet] = []
        for i, (sym, sb, x, _v, adv) in enumerate(rows):
            last = sb.iloc[-1]
            if not all(math.isfinite(float(pred[i])) for _, pred in returns.values()) or (
                    p_succ is not None and not math.isfinite(float(p_succ[i]))) or (
                    rank_scores is not None and not math.isfinite(float(rank_scores[i]))):
                log.warning("predict: non-finite model output for %s; no prediction (fail closed)", sym)
                continue
            est, slip, _rt_ni, _rt = self._costs(sym, last["close"], adv, spreads.get(sym), x)
            if not (math.isfinite(est) and math.isfinite(slip)):
                continue
            forecasts = tuple(
                HorizonForecast(
                    horizon_minutes=h,
                    expected_return_bps=float_to_dec(float(pred[i]) if mid != ZERO_BASELINE else 0.0, 4),
                    model_id=mid,
                    model_version=reg.model_version_of(ret_target(h)),
                )
                for h, (mid, pred) in returns.items()
            )
            notes = [f"registry trained on {reg.dataset_version} ({'SIMULATED' if reg.is_simulated else 'historical'})",
                     "expected_return_bps are NET of the cost model round trip"]
            if any(mid == ZERO_BASELINE for mid, _ in returns.values()):
                notes.append("zero_baseline horizons: no model beat the zero baseline after multiple-testing control")
            if regime_note:
                notes.append(regime_note)
            if fitted.vol is not None and vol_bps is not None and not math.isfinite(float(vol_bps[i])):
                notes.append("vol forecast unavailable: history shorter than the HAR/EWMA window")
            vb = None if vol_bps is None or not math.isfinite(float(vol_bps[i])) else float(vol_bps[i])
            out.append(
                PredictionSet(
                    prediction_id=new_id("pred"),
                    symbol=sym,
                    asof=asof_ts.to_pydatetime(),
                    features_asof=last["available_at_ts"].to_pydatetime(),
                    dataset_version=str(last["dataset_version"]),
                    registry_version=reg.registry_version,
                    return_forecasts=forecasts,
                    success_probability=float_to_dec(float(p_succ[i]), 6) if sel_success and p_succ is not None else None,
                    success_horizon_minutes=self.success_horizon_minutes if sel_success else None,
                    probability_calibrated=bool(sel_success and reg.probability_calibrated),
                    calibration_method=(fitted.success.method if sel_success and fitted.success is not None else None),
                    rank=ranks[i],
                    rank_score=None if rank_scores is None else float_to_dec(float(rank_scores[i]), 6),
                    universe_size=len(rows),
                    vol_forecast_bps=None if vb is None else float_to_dec(vb, 4),
                    vol_horizon_minutes=self.vol_horizon_minutes if vb is not None else None,
                    vol_model=reg.selected("vol") if vb is not None else None,
                    regime=regime,
                    regime_probability=None if regime_p is None else float_to_dec(regime_p, 6),
                    est_cost_bps=float_to_dec(est, 4),
                    est_slippage_bps=float_to_dec(slip, 4),
                    cost_model_id=f"{reg.selected('cost')}:{self.cost_model.version}",
                    is_simulated=bool(reg.is_simulated or bool(sb["is_synthetic"].any())),
                    notes=tuple(notes),
                )
            )
        return out

    # ------------------------------------------------------------------ SignalModel protocol
    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None:
        """Score the last bar of a prepared single-symbol PIT frame (``QuantSignalEngine`` adapter)."""
        if sym_bars.empty or str(sym_bars["timeframe"].iloc[0]) != self.timeframe:
            return None
        got = self._row(sym_bars)
        if got is None:
            return None
        x, _v, adv = got
        mid, model = self.registry.fitted.returns[self.signal_horizon_minutes]
        prob: float | None = None
        fitted = self.registry.fitted
        if (fitted.success is not None and self.registry.selected("success") == CATBOOST_CLF
                and self.success_horizon_minutes == self.signal_horizon_minutes):
            prob = float(fitted.success.calibrated(x.reshape(1, -1))[0])
        calibrated = bool(prob is not None and self.calibration_evaluated)
        if mid == ZERO_BASELINE:
            return ModelPrediction(expected_return_bps=0.0, score=0.0, probability_up=prob, probability_calibrated=calibrated)
        net = float(model.predict(x.reshape(1, -1))[0])
        if not math.isfinite(net):
            return None
        sym = str(sym_bars["symbol"].iloc[0])
        _est, _slip, rt_no_impact, _rt = self._costs(sym, sym_bars["close"].iloc[-1], adv, None, x)
        return ModelPrediction(expected_return_bps=net + rt_no_impact, score=net, probability_up=prob,
                               probability_calibrated=calibrated)


__all__ = ["MARKOV_2STATE", "STRATEGY_ID", "PredictionService"]
