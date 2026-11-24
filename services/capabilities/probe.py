"""Dated, read-only verification of what the Alpaca PAPER account can trade and see.

``CapabilityProbe(http).run()`` returns a JSON-able snapshot; ``save_snapshot`` writes it (content-hashed) under
``datasets/quantlab/capabilities``. ``asset_frame`` returns the full equity (active + inactive) and crypto asset list
with an inferred instrument kind for point-in-time universe construction.

Nothing is assumed: an endpoint that fails is recorded as such (``ERROR``/``DENIED``) and the asset class is then
not research- or execution-eligible through that layer.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from packages.contracts.enums import OrderType
from packages.contracts.policy import RiskPolicy
from packages.core.canonical import canonical_json, sha256_hex
from services.risk import limits

from .http import DATA_BASE, TRADING_BASE, AlpacaHttpError, ReadOnlyAlpacaHttp
from .instruments import EQUITY_RESEARCH_KINDS, InstrumentKind, classify

SCHEMA = "capabilities/1"
OPTION_UNDERLYINGS = ("SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "SPX", "XSP", "VIX", "DJX", "NDX", "RUT")
PDT_NOTE = ("FINRA replaced the pattern-day-trader provisions of Rule 4210 with intraday margin standards "
            "(effective 2026-06-04; SEC approval 2026-04-15). Alpaca deprecated pattern_day_trader/daytrade_count. "
            "A same-day round trip is limited by buying power, not a trade count; this build never borrows.")


class Layer:
    OK = "OK"
    DENIED = "DENIED"
    ABSENT = "ABSENT"
    ERROR = "ERROR"
    DISABLED_BY_POLICY = "DISABLED_BY_POLICY"
    DENYLISTED = "DENYLISTED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    EXECUTABLE = "EXECUTABLE"


@dataclass(frozen=True)
class Check:
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "detail": self.detail}


def _try(http: ReadOnlyAlpacaHttp, url: str, params: dict[str, Any] | None = None) -> tuple[Check, Any]:
    try:
        resp = http.get_response(url, params)
    except AlpacaHttpError as exc:
        return Check(Layer.ERROR, str(exc)[:200]), None
    if resp.status_code == 200:
        return Check(Layer.OK, "HTTP 200"), resp.json()
    if resp.status_code in (401, 403):
        return Check(Layer.DENIED, f"HTTP {resp.status_code}: {resp.text[:160]}"), None
    if resp.status_code == 404:
        return Check(Layer.ABSENT, "HTTP 404"), None
    return Check(Layer.ERROR, f"HTTP {resp.status_code}: {resp.text[:160]}"), None


class CapabilityProbe:
    def __init__(self, http: ReadOnlyAlpacaHttp, *, policy: RiskPolicy | None = None,
                 now: datetime | None = None, policy_source: str = "argument") -> None:
        self.http = http
        self.policy = policy or RiskPolicy()
        self.policy_verified = policy is not None
        self.policy_source = policy_source if policy is not None else "build default (no active policy read)"
        self.now = now or datetime.now(UTC)
        self.assets: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ raw collection
    def _account(self) -> tuple[Check, dict[str, Any]]:
        chk, acct = _try(self.http, f"{TRADING_BASE}/v2/account")
        if acct is None:
            return chk, {}
        keep = ("status", "crypto_status", "crypto_tier", "options_approved_level", "options_trading_level",
                "multiplier", "shorting_enabled", "trading_blocked", "account_blocked", "transfers_blocked",
                "trade_suspended_by_user", "currency", "equity", "cash", "buying_power", "non_marginable_buying_power",
                "options_buying_power", "balance_asof", "pattern_day_trader", "daytrade_count")
        out = {k: acct.get(k) for k in keep}
        out["pdt_fields_reported"] = "pattern_day_trader" in acct or "daytrade_count" in acct
        return chk, out

    def _config(self) -> tuple[Check, dict[str, Any]]:
        chk, cfg = _try(self.http, f"{TRADING_BASE}/v2/account/configurations")
        return chk, dict(cfg or {})

    def _asset_list(self, asset_class: str, status: str) -> tuple[Check, list[dict[str, Any]]]:
        chk, data = _try(self.http, f"{TRADING_BASE}/v2/assets", {"asset_class": asset_class, "status": status})
        return chk, list(data or [])

    def _option_roots(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        exp_to = (self.now + timedelta(days=60)).date().isoformat()
        for u in OPTION_UNDERLYINGS:
            chk, data = _try(self.http, f"{TRADING_BASE}/v2/options/contracts",
                             {"underlying_symbols": u, "limit": 1000, "expiration_date_lte": exp_to})
            contracts = list((data or {}).get("option_contracts") or [])
            out[u] = {
                "check": chk.as_dict(),
                "contracts_first_page": len(contracts),
                "roots": sorted({str(c.get("root_symbol")) for c in contracts if c.get("root_symbol")}),
                "styles": sorted({str(c.get("style")) for c in contracts if c.get("style")}),
                "multipliers": sorted({str(c.get("multiplier") or c.get("size")) for c in contracts}),
                "tradable": sum(bool(c.get("tradable")) for c in contracts),
                "has_more": bool((data or {}).get("next_page_token")),
            }
        return out

    def _data_checks(self) -> dict[str, dict[str, str]]:
        day0 = (self.now - timedelta(days=10)).date().isoformat()
        checks: dict[str, tuple[str, dict[str, Any]]] = {
            "stocks_sip_historical_daily": (f"{DATA_BASE}/v2/stocks/bars",
                                            {"symbols": "SPY", "timeframe": "1Day", "start": day0, "feed": "sip",
                                             "end": (self.now - timedelta(minutes=20)).isoformat()}),
            "stocks_iex_latest_quote": (f"{DATA_BASE}/v2/stocks/quotes/latest", {"symbols": "SPY", "feed": "iex"}),
            "stocks_sip_latest_quote": (f"{DATA_BASE}/v2/stocks/quotes/latest", {"symbols": "SPY", "feed": "sip"}),
            "crypto_bars": (f"{DATA_BASE}/v1beta3/crypto/us/bars",
                            {"symbols": "BTC/USD", "timeframe": "1Hour", "start": day0}),
            "crypto_orderbook": (f"{DATA_BASE}/v1beta3/crypto/us/latest/orderbooks", {"symbols": "BTC/USD"}),
            "options_indicative_snapshots": (f"{DATA_BASE}/v1beta1/options/snapshots/SPY",
                                             {"feed": "indicative", "limit": 1}),
            "options_opra_snapshots": (f"{DATA_BASE}/v1beta1/options/snapshots/SPY", {"feed": "opra", "limit": 1}),
            "news": (f"{DATA_BASE}/v1beta1/news", {"symbols": "SPY", "limit": 1}),
            "corporate_actions": (f"{DATA_BASE}/v1/corporate-actions", {"symbols": "SPY", "start": day0}),
            "screener_most_actives": (f"{DATA_BASE}/v1beta1/screener/stocks/most-actives", {"top": 1}),
        }
        return {name: _try(self.http, url, params)[0].as_dict() for name, (url, params) in checks.items()}

    # ------------------------------------------------------------------ layering
    def run(self) -> dict[str, Any]:
        acct_chk, acct = self._account()
        cfg_chk, cfg = self._config()
        eq_chk, equities = self._asset_list("us_equity", "active")
        inact_chk, inactive = self._asset_list("us_equity", "inactive")
        cr_chk, crypto = self._asset_list("crypto", "active")
        self.assets = {"us_equity_active": equities, "us_equity_inactive": inactive, "crypto_active": crypto}
        options = self._option_roots()
        data = self._data_checks()

        kinds = Counter(classify(a["symbol"], a.get("name"), a.get("exchange")) for a in equities if a.get("tradable"))
        attrs = Counter(at for a in equities for at in (a.get("attributes") or []))
        active_ok = (acct_chk.status == Layer.OK and acct.get("status") == "ACTIVE"
                     and acct.get("trading_blocked") is False and acct.get("account_blocked") is False
                     and acct.get("trade_suspended_by_user") is False)
        opt_level = int(acct.get("options_trading_level") or 0)
        multiplier = float(acct.get("multiplier") or 1)
        pol = self.policy
        policy_valid = self.policy_verified and not limits.validate_policy(pol)

        def acct_layer(ok: bool, detail: str) -> dict[str, str]:
            return Check(Layer.OK if ok else Layer.DENIED, detail).as_dict()

        def data_ok(*names: str) -> bool:
            return all(data.get(n, {}).get("status") == Layer.OK for n in names)

        classes: dict[str, dict[str, Any]] = {
            "us_equity_long": {
                "api_schema": Check(eq_chk.status, f"{sum(bool(a.get('tradable')) for a in equities)} tradable "
                                                   f"active equities").as_dict(),
                "market_data": Check(Layer.OK if data_ok("stocks_sip_historical_daily") else Layer.DENIED,
                                     "historical SIP bars (free plan, >15 min old); real-time IEX only").as_dict(),
                "account_permission": acct_layer(active_ok, f"account status {acct.get('status')}"),
                "policy_execution": Check(
                    Layer.EXECUTABLE if policy_valid else Layer.DISABLED_BY_POLICY,
                    f"LIMIT DAY whole-share long-only, regular session, allowlist of {len(pol.symbol_allowlist)} "
                    f"symbols, max order ${pol.max_order_notional}, max {pol.max_positions} position(s)").as_dict(),
            },
            "us_equity_short": {
                "api_schema": Check(Layer.OK, "shortable flag on assets").as_dict(),
                "market_data": Check(Layer.OK, "same as long").as_dict(),
                "account_permission": acct_layer(active_ok and bool(acct.get("shorting_enabled"))
                                                 and not cfg.get("no_shorting"),
                                                 f"shorting_enabled={acct.get('shorting_enabled')}"),
                "policy_execution": Check(Layer.DISABLED_BY_POLICY,
                                          "long_only is a structural invariant (services/risk/limits.py)").as_dict(),
            },
            "margin_borrowing": {
                "api_schema": Check(Layer.OK, "margin account fields").as_dict(),
                "market_data": Check(Layer.OK, "n/a").as_dict(),
                "account_permission": acct_layer(active_ok and multiplier > 1, f"multiplier={acct.get('multiplier')}"),
                "policy_execution": Check(Layer.DISABLED_BY_POLICY,
                                          "allow_margin cannot be enabled; BUYs must fit settled/non-marginable "
                                          "buying power").as_dict(),
            },
            "leveraged_inverse_etp": {
                "api_schema": Check(Layer.OK, f"{kinds[InstrumentKind.LEVERAGED_ETF]} leveraged, "
                                              f"{kinds[InstrumentKind.INVERSE_ETF]} inverse, "
                                              f"{kinds[InstrumentKind.VOLATILITY_ETP]} volatility ETPs (name-inferred)"
                                    ).as_dict(),
                "market_data": Check(Layer.OK if data_ok("stocks_sip_historical_daily") else Layer.DENIED,
                                     "equity bars").as_dict(),
                "account_permission": acct_layer(active_ok, "equity trading"),
                "policy_execution": Check(Layer.DENYLISTED,
                                          f"hard denylist of {len(limits.HARD_DENYLIST)} symbols; research-only"
                                          ).as_dict(),
            },
            "us_equity_multi_day": {
                "api_schema": Check(Layer.OK, "equity positions persist across sessions at the broker").as_dict(),
                "market_data": Check(Layer.OK if data_ok("stocks_sip_historical_daily") else Layer.DENIED,
                                     "daily bars").as_dict(),
                "account_permission": acct_layer(active_ok, "equity trading"),
                "policy_execution": Check(
                    Layer.EXECUTABLE if pol.allow_overnight_positions and policy_valid else Layer.DISABLED_BY_POLICY,
                    "multi-day holding implemented (DAY stops re-armed each session, gap exits, session deadline); "
                    + ("enabled for " + ", ".join(pol.multi_day_strategy_allowlist) if pol.allow_overnight_positions
                       else "disabled: allow_overnight_positions is False in the active risk policy")).as_dict(),
            },
            "crypto_spot": {
                "api_schema": Check(cr_chk.status, f"{len(crypto)} active pairs").as_dict(),
                "market_data": Check(Layer.OK if data_ok("crypto_bars", "crypto_orderbook") else Layer.DENIED,
                                     "crypto bars + latest order book (free)").as_dict(),
                "account_permission": acct_layer(active_ok and acct.get("crypto_status") == "ACTIVE",
                                                 f"crypto_status={acct.get('crypto_status')}, "
                                                 f"tier={acct.get('crypto_tier')}"),
                "policy_execution": Check(
                    Layer.NOT_IMPLEMENTED,
                    "crypto contracts exist, but risk, gateway, continuous feed, received-asset fees and net "
                    "quantity reconciliation are not implemented end to end; research-only regardless of "
                    "crypto_enabled").as_dict(),
            },
            "us_options": {
                "api_schema": Check(Layer.OK if any(o["contracts_first_page"] for o in options.values())
                                    else Layer.ABSENT,
                                    "option contracts endpoint; per-underlying detail in option_underlyings").as_dict(),
                "market_data": Check(
                    Layer.OK if data_ok("options_indicative_snapshots") else Layer.DENIED,
                    "indicative (delayed/derived) quotes only; OPRA "
                    f"{data.get('options_opra_snapshots', {}).get('status')}: executable NBBO history unavailable"
                ).as_dict(),
                "account_permission": acct_layer(active_ok and opt_level >= 1, f"options_trading_level={opt_level} "
                                                 "(1 covered/cash-secured, 2 long, 3 spreads); contract discovery "
                                                 "does not verify account-specific index option enablement"),
                "policy_execution": Check(Layer.DISABLED_BY_POLICY,
                                          "allowed_order_types must be exactly (LIMIT,) on equity symbols; options "
                                          "are research/shadow-only").as_dict(),
            },
            "extended_and_overnight_hours": {
                "api_schema": Check(Layer.OK, f"{attrs.get('overnight_tradable', 0)} overnight_tradable assets"
                                    ).as_dict(),
                "market_data": Check(Layer.OK, "IEX").as_dict(),
                "account_permission": acct_layer(active_ok and cfg_chk.status == Layer.OK
                                                 and not cfg.get("disable_overnight_trading", False),
                                                 f"disable_overnight_trading={cfg.get('disable_overnight_trading')}"),
                "policy_execution": Check(Layer.DISABLED_BY_POLICY, "regular_session_only is an invariant").as_dict(),
            },
        }
        for cls in classes.values():
            cls["research_eligible"] = cls["market_data"]["status"] == Layer.OK
            cls["execution_eligible"] = (cls["api_schema"]["status"] == Layer.OK
                                         and cls["market_data"]["status"] == Layer.OK
                                         and cls["account_permission"]["status"] == Layer.OK
                                         and cls["policy_execution"]["status"] == Layer.EXECUTABLE)

        body: dict[str, Any] = {
            "schema": SCHEMA,
            "asof": self.now.isoformat(),
            "endpoint": "paper",
            "account": {**acct, "type": "margin" if multiplier > 1 else "cash", "pdt_note": PDT_NOTE,
                        "check": acct_chk.as_dict()},
            "configuration": {**cfg, "check": cfg_chk.as_dict()},
            "assets": {
                "us_equity_active": len(equities),
                "us_equity_tradable": sum(bool(a.get("tradable")) for a in equities),
                "us_equity_inactive": len(inactive), "inactive_check": inact_chk.as_dict(),
                "by_exchange": dict(Counter(str(a.get("exchange")) for a in equities).most_common()),
                "by_kind_tradable": {str(k): v for k, v in kinds.most_common()},
                "attributes": dict(attrs.most_common()),
                "fractionable": sum(bool(a.get("fractionable")) for a in equities),
                "shortable": sum(bool(a.get("shortable")) for a in equities),
                "crypto_pairs": sorted(a["symbol"] for a in crypto if a.get("tradable")),
            },
            "option_underlyings": options,
            "market_data": data,
            "policy": {"version": pol.version, "valid": policy_valid, "source": self.policy_source,
                       "verified": self.policy_verified,
                       "allow_overnight_positions": pol.allow_overnight_positions,
                       "crypto_enabled": pol.crypto_enabled,
                       "order_types": [str(t) for t in pol.allowed_order_types],
                       "long_only": pol.long_only, "allow_margin": pol.allow_margin,
                       "entry_order_type": str(OrderType.LIMIT)},
            "asset_classes": classes,
        }
        body["content_hash"] = sha256_hex(canonical_json({k: v for k, v in body.items() if k != "asof"}))[:16]
        return body

    # ------------------------------------------------------------------ universe frame
    def asset_frame(self) -> pd.DataFrame:
        if not self.assets:
            raise RuntimeError("run() first")
        rows: list[dict[str, Any]] = []
        for key, items in self.assets.items():
            for a in items:
                cls = str(a.get("class") or ("crypto" if key.startswith("crypto") else "us_equity"))
                kind = classify(a["symbol"], a.get("name"), a.get("exchange"), cls)
                rows.append({
                    "symbol": a["symbol"], "asset_class": cls, "name": a.get("name"), "exchange": a.get("exchange"),
                    "status": a.get("status"), "tradable": bool(a.get("tradable")),
                    "marginable": bool(a.get("marginable")), "shortable": bool(a.get("shortable")),
                    "easy_to_borrow": bool(a.get("easy_to_borrow")), "fractionable": bool(a.get("fractionable")),
                    "attributes": ",".join(a.get("attributes") or []), "kind": str(kind),
                    "research_kind": kind in EQUITY_RESEARCH_KINDS or cls == "crypto",
                    "min_order_size": a.get("min_order_size"), "min_trade_increment": a.get("min_trade_increment"),
                    "price_increment": a.get("price_increment"), "asof": self.now.isoformat(),
                })
        return pd.DataFrame(rows).drop_duplicates(subset=["symbol", "asset_class", "status"])


def capabilities_dir(datasets_dir: Path) -> Path:
    return datasets_dir / "quantlab" / "capabilities"


def save_snapshot(snapshot: dict[str, Any], datasets_dir: Path, assets: pd.DataFrame | None = None) -> Path:
    out = capabilities_dir(datasets_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = snapshot["asof"][:10].replace("-", "")
    path = out / f"cap_{stamp}_{snapshot['content_hash']}.json"
    text = json.dumps(snapshot, indent=2, sort_keys=True, default=str)
    path.write_text(text)
    (out / "latest.json").write_text(text)
    if assets is not None:
        adir = datasets_dir / "quantlab" / "assets"
        adir.mkdir(parents=True, exist_ok=True)
        assets.to_parquet(adir / f"assets_{stamp}.parquet", index=False)
        assets.to_parquet(adir / "assets_latest.parquet", index=False)
    return path


def load_latest(datasets_dir: Path) -> dict[str, Any] | None:
    path = capabilities_dir(datasets_dir) / "latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_assets(datasets_dir: Path) -> pd.DataFrame:
    path = datasets_dir / "quantlab" / "assets" / "assets_latest.parquet"
    if not path.is_file():
        raise FileNotFoundError("no asset snapshot; run `python -m services.capabilities probe`")
    return pd.read_parquet(path)
