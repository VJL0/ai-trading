"""Strategy E — options research: pricing, Greeks, implied volatility, early exercise, defined-risk spreads and a
live chain scan (implied vs forecast realized volatility, term structure, skew).

Data reality (verified by the capability probe): the account sees Alpaca's free *indicative* option feed (derived,
delayed) and contract metadata; OPRA is not entitled ("OPRA agreement is not signed"), and historical option
*quotes* are unavailable. Executable-NBBO backtests are therefore impossible here, so every option opportunity is
RESEARCH_ONLY and enters the opportunity ranking with zero evidence weight; prospective shadow logging is the only
honest way to accumulate evidence. Index options (SPX/XSP/VIX roots) are European-style; equity/ETF options are
American (early exercise matters for short calls before ex-dividend dates and deep ITM puts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
from scipy.optimize import brentq

from services.capabilities.http import DATA_BASE, TRADING_BASE, AlpacaHttpError, ReadOnlyAlpacaHttp

from .costs import OptionCosts

MULTIPLIER = 100.0


# ------------------------------------------------------------------------------------------- pricing
def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = _ncdf
    if call:
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool) -> dict[str, float]:
    if T <= 0 or sigma <= 0:
        intrinsic_delta = (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
        return {"delta": intrinsic_delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = _npdf(d1)
    delta = math.exp(-q * T) * (_ncdf(d1) if call else _ncdf(d1) - 1)
    gamma = math.exp(-q * T) * pdf / (S * sigma * sq)
    vega = S * math.exp(-q * T) * pdf * sq / 100.0  # per vol point
    if call:
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * sq) - r * K * math.exp(-r * T) * _ncdf(d2)
                 + q * S * math.exp(-q * T) * _ncdf(d1))
    else:
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * sq) + r * K * math.exp(-r * T) * _ncdf(-d2)
                 - q * S * math.exp(-q * T) * _ncdf(-d1))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365.0}


def crr_american(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool,
                 steps: int = 300) -> tuple[float, bool]:
    """Cox-Ross-Rubinstein American price and whether immediate exercise is optimal at t=0."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S)), True
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    j = np.arange(steps + 1)
    prices = S * u ** (steps - j) * d ** j
    vals = np.maximum(prices - K, 0.0) if call else np.maximum(K - prices, 0.0)
    for i in range(steps - 1, -1, -1):
        prices = prices[: i + 1] / u
        cont = disc * (p * vals[: i + 1] + (1 - p) * vals[1: i + 2])
        ex = np.maximum(prices - K, 0.0) if call else np.maximum(K - prices, 0.0)
        vals = np.maximum(cont, ex)
    intrinsic = max(0.0, (S - K) if call else (K - S))
    return float(vals[0]), bool(intrinsic > 0 and abs(vals[0] - intrinsic) < 1e-6)


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, call: bool) -> float | None:
    intrinsic = max(0.0, (S - K) if call else (K - S)) * math.exp(-r * T)
    if T <= 0 or price <= intrinsic + 1e-6 or price <= 0:
        return None
    f = lambda s: bs_price(S, K, T, r, q, s, call) - price  # noqa: E731
    try:
        if f(1e-4) > 0 or f(5.0) < 0:
            return None
        root = brentq(f, 1e-4, 5.0, xtol=1e-6)
        return float(root[0] if isinstance(root, tuple) else root)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------- structures
@dataclass(frozen=True)
class Leg:
    symbol: str
    strike: float
    call: bool
    qty: int  # +1 long, -1 short (contracts)
    bid: float
    ask: float
    expiry: date
    style: str = "american"

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


def fhs_terminal(S: float, days: int, sigma_annual: float, daily_returns: np.ndarray | None, n_paths: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Terminal prices by filtered historical simulation: daily returns standardized by their own EWMA volatility
    (lambda 0.94), demeaned, resampled with replacement and rescaled to the forecast volatility. This keeps the
    empirical skew and fat tails that a lognormal model (and a flat-vol view of a skewed surface) ignores.
    Falls back to lognormal without enough history."""
    r = np.asarray(daily_returns if daily_returns is not None else [], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 250 or days < 1:
        z = rng.standard_normal(n_paths)
        T = max(days, 1) / 252
        return S * np.exp(-0.5 * sigma_annual ** 2 * T + sigma_annual * math.sqrt(T) * z), "lognormal"
    var = np.empty_like(r)
    var[0] = r[:20].var()
    for i in range(1, len(r)):
        var[i] = 0.94 * var[i - 1] + 0.06 * r[i - 1] ** 2
    zs = r / np.sqrt(np.maximum(var, 1e-12))
    zs = (zs - zs.mean()) / zs.std()
    draws = rng.choice(zs, size=(n_paths, days), replace=True)
    sd = sigma_annual / math.sqrt(252)
    log_ret = np.log1p(np.clip(draws * sd, -0.95, None)).sum(axis=1)
    log_ret -= np.log(np.mean(np.exp(log_ret)))  # zero drift: E[S_T] = S
    return S * np.exp(log_ret), "filtered_historical_simulation"


def vertical_spread(S: float, long_leg: Leg, short_leg: Leg, *, sigma_physical: float, mu_annual: float = 0.0,
                    r: float = 0.04, asof: date | None = None, costs: OptionCosts | None = None,
                    n_paths: int = 20000, seed: int = 3, daily_returns: np.ndarray | None = None) -> dict[str, Any]:
    """Debit/credit vertical (same expiry, same type) at executable touch (buy ask, sell bid) plus fees. Payoff at
    expiry under a *physical* model: filtered historical simulation when ``daily_returns`` is given (empirical
    tails), otherwise lognormal with ``sigma_physical`` and drift ``mu_annual``."""
    costs = costs or OptionCosts()
    asof = asof or datetime.now(UTC).date()
    T = max((long_leg.expiry - asof).days, 0) / 365.0
    call = long_leg.call
    entry = long_leg.ask - short_leg.bid  # >0 debit, <0 credit (per share)
    fees = (costs.per_contract_fees(sell=False, premium=long_leg.ask)
            + costs.per_contract_fees(sell=True, premium=short_leg.bid)) / MULTIPLIER
    width = abs(short_leg.strike - long_leg.strike)
    if entry > 0:  # debit spread
        max_loss = entry + fees
        max_gain = width - entry - fees
        capital = max_loss
    else:  # credit spread: margin = width - credit
        max_loss = width + entry + fees
        max_gain = -entry - fees
        capital = max_loss
    rng = np.random.default_rng(seed)
    if daily_returns is not None:
        days = max(1, round(T * 252))
        ST, model_kind = fhs_terminal(S, days, sigma_physical, daily_returns, n_paths, rng)
    else:
        z = rng.standard_normal(n_paths)
        ST = S * np.exp((mu_annual - 0.5 * sigma_physical ** 2) * T + sigma_physical * math.sqrt(T) * z)
        model_kind = "lognormal"
    lp = np.maximum(ST - long_leg.strike, 0) if call else np.maximum(long_leg.strike - ST, 0)
    sp = np.maximum(ST - short_leg.strike, 0) if call else np.maximum(short_leg.strike - ST, 0)
    exit_fees = (costs.per_contract_fees(sell=True, premium=0.0) + costs.per_contract_fees(sell=False, premium=0.0)) \
        / MULTIPLIER
    pnl = lp - sp - entry - fees - np.where(np.abs(lp - sp) > 0, exit_fees, 0.0)
    pnl_frac = pnl / capital if capital > 0 else pnl * 0
    q05 = np.quantile(pnl_frac, 0.05)
    early = early_exercise_flags(S, short_leg, T, r)
    return {
        "type": ("call" if call else "put") + ("_debit" if entry > 0 else "_credit") + "_vertical",
        "entry_per_share": entry, "fees_per_share": fees, "width": width, "T_years": T,
        "max_loss_per_spread": max_loss * MULTIPLIER, "max_gain_per_spread": max_gain * MULTIPLIER,
        "capital_per_spread": capital * MULTIPLIER,
        "expected_pnl_frac": float(pnl_frac.mean()), "std_pnl_frac": float(pnl_frac.std()),
        "cvar95_frac": float(pnl_frac[pnl_frac <= q05].mean()), "prob_profit": float((pnl > 0).mean()),
        "early_exercise": early, "model": {"kind": model_kind, "sigma_physical": sigma_physical,
                                           "mu_annual": mu_annual, "paths": n_paths},
    }


def early_exercise_flags(S: float, short_leg: Leg, T: float, r: float, dividend_before_expiry: float = 0.0) -> list[str]:
    flags = []
    if short_leg.style != "american":
        return flags
    itm = (short_leg.strike < S) if short_leg.call else (short_leg.strike > S)
    if not itm:
        return flags
    intrinsic = (S - short_leg.strike) if short_leg.call else (short_leg.strike - S)
    extrinsic = max(0.0, short_leg.mid - intrinsic)
    if short_leg.call and dividend_before_expiry > extrinsic:
        flags.append("short ITM call: dividend exceeds extrinsic value -> early assignment likely before ex-date")
    if not short_leg.call and extrinsic < short_leg.strike * (1 - math.exp(-r * T)):
        flags.append("short deep-ITM put: extrinsic below interest on strike -> early assignment risk")
    if extrinsic < 0.05:
        flags.append("short leg has < $0.05 extrinsic value: assignment risk")
    return flags


# ------------------------------------------------------------------------------------------- chain + scan
def fetch_chain(http: ReadOnlyAlpacaHttp, underlying: str, *, min_days: int = 7, max_days: int = 75,
                asof: datetime | None = None) -> dict[str, Any]:
    asof = asof or datetime.now(UTC)
    lo = (asof + timedelta(days=min_days)).date().isoformat()
    hi = (asof + timedelta(days=max_days)).date().isoformat()
    contracts = http.paginate(f"{TRADING_BASE}/v2/options/contracts",
                              {"underlying_symbols": underlying, "expiration_date_gte": lo,
                               "expiration_date_lte": hi, "limit": 1000}, items_key="option_contracts", max_pages=20)
    snaps = http.paginate(f"{DATA_BASE}/v1beta1/options/snapshots/{underlying}",
                          {"feed": "indicative", "limit": 1000, "expiration_date_gte": lo,
                           "expiration_date_lte": hi}, items_key="snapshots", max_pages=20)
    return {"underlying": underlying, "asof": asof.isoformat(), "contracts": contracts, "snapshots": snaps,
            "feed": "indicative"}


def parse_occ(symbol: str) -> tuple[str, date, bool, float]:
    """OCC option symbol -> (root, expiry, is_call, strike)."""
    i = len(symbol) - 15
    root = symbol[:i]
    exp = datetime.strptime(symbol[i:i + 6], "%y%m%d").replace(tzinfo=UTC).date()
    return root, exp, symbol[i + 6] == "C", int(symbol[i + 7:]) / 1000.0


def chain_quotes(chain: dict[str, Any], S: float, r: float = 0.04, q: float = 0.0,
                 asof: date | None = None) -> list[dict[str, Any]]:
    asof = asof or datetime.now(UTC).date()
    style = {c["symbol"]: c.get("style", "american") for c in chain.get("contracts") or []}
    oi = {c["symbol"]: c.get("open_interest") for c in chain.get("contracts") or []}
    out = []
    snaps = chain.get("snapshots") or {}
    for sym, snap in (snaps.items() if isinstance(snaps, dict) else []):
        qt = snap.get("latestQuote") or {}
        bid, ask = float(qt.get("bp") or 0), float(qt.get("ap") or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        root, exp, call, K = parse_occ(sym)
        T = max((exp - asof).days, 0) / 365.0
        if T <= 0:
            continue
        mid = 0.5 * (bid + ask)
        iv = snap.get("impliedVolatility") or implied_vol(mid, S, K, T, r, q, call)
        if iv is None:
            continue
        g = snap.get("greeks") or bs_greeks(S, K, T, r, q, float(iv), call)
        out.append({"symbol": sym, "root": root, "expiry": exp, "call": call, "strike": K, "T": T, "bid": bid,
                    "ask": ask, "mid": mid, "spread_pct": (ask - bid) / mid, "iv": float(iv),
                    "delta": float(g.get("delta", 0.0)), "style": style.get(sym, "american"),
                    "open_interest": oi.get(sym), "quote_ts": qt.get("t")})
    return out


def parity_spot(chain: dict[str, Any], r: float = 0.04, asof: date | None = None, min_days: int = 7) -> float | None:
    """Spot implied by the quotes themselves: median over near-ATM strikes of C - P + K e^{-rT} (nearest expiry
    >= ``min_days``). Indicative quotes are delayed, so pricing them against a real-time stock print mixes clocks;
    the parity spot keeps model and quotes on one timestamp (dividends before expiry ignored: small at 30 DTE)."""
    asof = asof or datetime.now(UTC).date()
    snaps = chain.get("snapshots") or {}
    mids: dict[tuple[date, float, bool], float] = {}
    for sym, snap in (snaps.items() if isinstance(snaps, dict) else []):
        qt = snap.get("latestQuote") or {}
        bid, ask = float(qt.get("bp") or 0), float(qt.get("ap") or 0)
        if bid > 0 and ask >= bid:
            _, exp, call, k = parse_occ(sym)
            mids[(exp, k, call)] = 0.5 * (bid + ask)
    exps = sorted({e for (e, _, _) in mids if (e - asof).days >= min_days})
    for exp in exps:
        T = (exp - asof).days / 365.0
        pairs = [(k, mids[(exp, k, True)] - mids[(exp, k, False)] + k * math.exp(-r * T))
                 for (e, k, c) in list(mids) if e == exp and c and (exp, k, False) in mids]
        if len(pairs) < 3:
            continue
        guess = float(np.median([v for _, v in pairs]))
        near = sorted(pairs, key=lambda kv: abs(kv[0] - guess))[:6]
        return float(np.median([v for _, v in near]))
    return None


def har_rv_forecast(daily_returns: np.ndarray, horizon: int = 21) -> float:
    """Annualized HAR-RV forecast (Corsi 2009): OLS of the forward ``horizon``-day mean squared return on the
    daily / weekly / monthly trailing means, fitted on the past only, evaluated at the last observation."""
    import pandas as pd

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 120:
        return float(np.std(r) * math.sqrt(252)) if len(r) > 5 else float("nan")
    rv = pd.Series(r ** 2)
    feats = pd.DataFrame({"d": rv, "w": rv.rolling(5).mean(), "m": rv.rolling(22).mean()})
    rev = pd.Series(rv.to_numpy()[::-1])
    fwd_vals = np.asarray(rev.rolling(horizon).mean(), dtype=float)[::-1]  # mean of rv[t .. t+horizon-1]
    fwd = pd.Series(np.r_[fwd_vals[1:], np.nan], name="y")  # shift: mean of rv[t+1 .. t+horizon]
    data = pd.concat([feats, fwd], axis=1).dropna()
    X = np.column_stack([np.ones(len(data)), data[["d", "w", "m"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, data["y"].to_numpy(), rcond=None)
    last = feats.iloc[-1]
    pred = float(beta[0] + beta[1] * last["d"] + beta[2] * last["w"] + beta[3] * last["m"])
    if not math.isfinite(pred) or pred <= 0:
        pred = float(last["m"])
    return math.sqrt(max(pred, 1e-10) * 252)


def scan_vol_surface(quotes: list[dict[str, Any]], S: float, rv_forecast: float) -> dict[str, Any]:
    """ATM IV term structure, 25-delta skew, implied-minus-forecast variance premium and liquidity."""
    if not quotes:
        return {"status": "NO_QUOTES"}
    by_exp: dict[date, list[dict[str, Any]]] = {}
    for qd in quotes:
        by_exp.setdefault(qd["expiry"], []).append(qd)
    term = []
    for exp, qs in sorted(by_exp.items()):
        atm = sorted(qs, key=lambda x: abs(x["strike"] - S))[:4]
        put25 = [x for x in qs if not x["call"] and -0.35 < x["delta"] < -0.15]
        call25 = [x for x in qs if x["call"] and 0.15 < x["delta"] < 0.35]
        term.append({"expiry": exp.isoformat(), "T": atm[0]["T"], "atm_iv": float(np.mean([x["iv"] for x in atm])),
                     "atm_spread_pct": float(np.mean([x["spread_pct"] for x in atm])),
                     "skew_25d": (float(np.mean([x["iv"] for x in put25])) - float(np.mean([x["iv"] for x in call25])))
                     if put25 and call25 else None, "n": len(qs)})
    near = min(term, key=lambda x: abs(x["T"] - 30 / 365))
    far = max(term, key=lambda x: x["T"])
    vrp = near["atm_iv"] ** 2 - rv_forecast ** 2
    return {"status": "OK", "spot": S, "rv_forecast": rv_forecast, "atm_iv_30d": near["atm_iv"],
            "term_slope": far["atm_iv"] - near["atm_iv"], "variance_premium": vrp,
            "iv_rv_ratio": near["atm_iv"] / rv_forecast if rv_forecast > 0 else None,
            "skew_25d_30d": near["skew_25d"], "atm_spread_pct_30d": near["atm_spread_pct"], "term": term}


def candidate_spreads(quotes: list[dict[str, Any]], S: float, rv_forecast: float, *, target_days: int = 30,
                      asof: date | None = None, daily_returns: np.ndarray | None = None) -> list[dict[str, Any]]:
    """A small, pre-declared set of defined-risk structures near 30 DTE: bull call debit, bear put debit,
    bull put credit (sells the variance premium) — evaluated at executable touch with fees."""
    if not quotes:
        return []
    exps = sorted({q["expiry"] for q in quotes}, key=lambda e: abs((e - (asof or datetime.now(UTC).date())).days
                                                                    - target_days))
    exp = exps[0]
    qs = [q for q in quotes if q["expiry"] == exp]

    def pick(call: bool, target_delta: float) -> dict[str, Any] | None:
        side = [q for q in qs if q["call"] == call]
        return min(side, key=lambda q: abs(q["delta"] - target_delta)) if side else None

    def leg(qd: dict[str, Any], qty: int) -> Leg:
        return Leg(qd["symbol"], qd["strike"], qd["call"], qty, qd["bid"], qd["ask"], qd["expiry"], qd["style"])

    out = []
    plans = [("bull_call_debit", True, 0.5, 0.3), ("bear_put_debit", False, -0.5, -0.3),
             ("bull_put_credit", False, -0.3, -0.15)]
    for name, call, d_long, d_short in plans:
        a, b = pick(call, d_long), pick(call, d_short)
        if not a or not b or a["symbol"] == b["symbol"]:
            continue
        if name == "bull_put_credit":  # long the further OTM put, short the nearer one
            a, b = b, a
        res = vertical_spread(S, leg(a, 1), leg(b, -1), sigma_physical=rv_forecast, asof=asof,
                              daily_returns=daily_returns)
        res.update({"name": name, "expiry": exp.isoformat(), "long": a["symbol"], "short": b["symbol"]})
        out.append(res)
    return out


def scan_underlying(http: ReadOnlyAlpacaHttp, underlying: str, spot: float, daily_returns: np.ndarray) -> dict[str, Any]:
    try:
        chain = fetch_chain(http, underlying)
    except AlpacaHttpError as exc:
        return {"underlying": underlying, "status": "ERROR", "error": str(exc)[:200]}
    rv = har_rv_forecast(daily_returns)
    implied = parity_spot(chain)
    stock_print = spot
    if implied is not None and abs(implied / spot - 1) < 0.05:
        spot = implied
    quotes = chain_quotes(chain, spot)
    surface = scan_vol_surface(quotes, spot, rv)
    spreads = candidate_spreads(quotes, spot, rv, daily_returns=daily_returns)
    return {"underlying": underlying, "feed": "indicative", "asof": chain["asof"], "n_quotes": len(quotes),
            "stock_print": stock_print, "parity_spot": implied, "model_spot": spot,
            "n_contracts": len(chain.get("contracts") or []), "surface": surface, "spreads": spreads,
            "evidence": "RESEARCH_ONLY: indicative quotes; no executable quote history for validation"}


# ------------------------------------------------------------------------------------------- prospective journal
def journal_candidates(path: Any, scan: dict[str, Any]) -> int:
    """Append each scanned spread (entry at executable touch, model stats) once per underlying/day/structure."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if p.is_file():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["underlying"], r["day"], r["name"], r["expiry"]))
    day = str(scan.get("asof", ""))[:10]
    n = 0
    with p.open("a") as fh:
        for u in scan.get("underlyings", []):
            for sp in u.get("spreads", []):
                key = (u["underlying"], day, sp["name"], sp["expiry"])
                if key in seen:
                    continue
                rec = {"underlying": u["underlying"], "day": day, "spot": u.get("surface", {}).get("spot"),
                       "name": sp["name"], "expiry": sp["expiry"], "long": sp["long"], "short": sp["short"],
                       "entry_per_share": sp["entry_per_share"], "fees_per_share": sp["fees_per_share"],
                       "width": sp["width"], "capital_per_spread": sp["capital_per_spread"],
                       "expected_pnl_frac": sp["expected_pnl_frac"], "prob_profit": sp["prob_profit"],
                       "settled": None}
                fh.write(json.dumps(rec, default=str) + "\n")
                n += 1
    return n


def settle_journal(path: Any, closes: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Settle journal entries whose expiry has a known underlying close: payoff at expiry vs entry at touch."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return {"settled": 0, "open": 0}
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    changed = 0
    for r in rows:
        if r.get("settled") is not None:
            continue
        px = closes.get((r["underlying"], r["expiry"]))
        if px is None:
            continue
        _, _, call, k_long = parse_occ(r["long"])
        _, _, _, k_short = parse_occ(r["short"])
        lp = max(px - k_long, 0.0) if call else max(k_long - px, 0.0)
        spv = max(px - k_short, 0.0) if call else max(k_short - px, 0.0)
        pnl = lp - spv - r["entry_per_share"] - r["fees_per_share"]
        cap = r["capital_per_spread"] / MULTIPLIER
        r["settled"] = {"underlying_close": px, "pnl_per_share": pnl, "pnl_frac": pnl / cap if cap > 0 else None}
        changed += 1
    if changed:
        p.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    done = [r for r in rows if r.get("settled")]
    return {"settled": len(done), "open": len(rows) - len(done), "newly_settled": changed,
            "mean_pnl_frac": float(np.mean([r["settled"]["pnl_frac"] for r in done])) if done else None}
