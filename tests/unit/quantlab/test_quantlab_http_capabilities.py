from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from packages.contracts.policy import RiskPolicy
from services.capabilities.http import ReadOnlyAlpacaHttp, ReadOnlyViolation
from services.capabilities.instruments import InstrumentKind, classify, leverage_hint
from services.capabilities.probe import CapabilityProbe, Layer, save_snapshot

PK = "PKTESTKEY000000000"


def _client(handler, **kw) -> ReadOnlyAlpacaHttp:  # type: ignore[no-untyped-def]
    return ReadOnlyAlpacaHttp(PK, "secret", transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_refuses_live_and_non_paper_keys() -> None:
    with pytest.raises(ReadOnlyViolation):
        ReadOnlyAlpacaHttp("AKLIVEKEY", "s")
    with pytest.raises(ReadOnlyViolation):
        ReadOnlyAlpacaHttp("XXSOMETHING", "s")


def test_get_only_and_host_allowlist() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.method)
        return httpx.Response(200, json={"ok": True})

    c = _client(handler)
    assert c.get("https://data.alpaca.markets/v2/stocks/bars") == {"ok": True}
    with pytest.raises(ReadOnlyViolation):
        c.get("https://api.alpaca.markets/v2/orders")  # live trading host
    with pytest.raises(ReadOnlyViolation):
        c.get("http://paper-api.alpaca.markets/v2/account")  # not https
    assert not hasattr(c, "post") and not hasattr(c, "delete")
    assert seen == ["GET"]
    assert "secret" not in repr(c)


def test_retries_429_then_succeeds_and_paginates() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        token = req.url.params.get("page_token")
        if token is None:
            return httpx.Response(200, json={"bars": {"A": [1, 2]}, "next_page_token": "p2"})
        return httpx.Response(200, json={"bars": {"A": [3], "B": [4]}, "next_page_token": None})

    c = _client(handler)
    out = c.paginate("https://data.alpaca.markets/v2/stocks/bars", {"symbols": "A,B"}, items_key="bars")
    assert out == {"A": [1, 2, 3], "B": [4]}
    assert c.retries == 1


def _probe_handler(*, crypto_status: str = "ACTIVE", options_403: bool = True):  # type: ignore[no-untyped-def]
    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.path
        if p == "/v2/account":
            return httpx.Response(200, json={"status": "ACTIVE", "crypto_status": crypto_status,
                                             "options_trading_level": 3, "multiplier": "4", "shorting_enabled": True,
                                             "trading_blocked": False, "account_blocked": False, "equity": "5000",
                                             "trade_suspended_by_user": False,
                                             "account_number": "PA123456789"})
        if p == "/v2/account/configurations":
            return httpx.Response(200, json={"fractional_trading": True, "no_shorting": False})
        if p == "/v2/assets":
            ac = req.url.params.get("asset_class")
            if ac == "crypto":
                return httpx.Response(200, json=[{"symbol": "BTC/USD", "class": "crypto", "tradable": True,
                                                  "exchange": "CRYPTO", "name": "Bitcoin"}])
            if req.url.params.get("status") == "inactive":
                return httpx.Response(200, json=[{"symbol": "OLDCO", "class": "us_equity", "tradable": False,
                                                  "exchange": "NYSE", "name": "Old Co", "status": "inactive"}])
            return httpx.Response(200, json=[
                {"symbol": "SPY", "class": "us_equity", "tradable": True, "exchange": "ARCA",
                 "name": "SPDR S&P 500 ETF Trust", "status": "active", "attributes": ["has_options"]},
                {"symbol": "SOXL", "class": "us_equity", "tradable": True, "exchange": "ARCA",
                 "name": "Direxion Daily Semiconductor Bull 3X ETF", "status": "active"}])
        if p == "/v2/options/contracts":
            return httpx.Response(200, json={"option_contracts": [
                {"symbol": "SPY261218C00500000", "root_symbol": "SPY", "style": "american", "multiplier": "100",
                 "tradable": True}], "next_page_token": None})
        if "options/snapshots" in p and req.url.params.get("feed") == "opra" and options_403:
            return httpx.Response(403, json={"message": "OPRA agreement is not signed"})
        if p == "/v2/stocks/quotes/latest" and req.url.params.get("feed") == "sip":
            return httpx.Response(403, json={"message": "subscription does not permit querying recent SIP data"})
        return httpx.Response(200, json={})
    return handler


def test_probe_separates_research_from_execution(tmp_path) -> None:  # type: ignore[no-untyped-def]
    probe = CapabilityProbe(_client(_probe_handler()), policy=RiskPolicy(), now=datetime(2026, 11, 23, 15, tzinfo=UTC))
    snap = probe.run()
    cls = snap["asset_classes"]
    assert cls["us_equity_long"]["execution_eligible"] is True
    assert cls["crypto_spot"]["research_eligible"] is True
    assert cls["crypto_spot"]["execution_eligible"] is False
    assert cls["crypto_spot"]["policy_execution"]["status"] == Layer.NOT_IMPLEMENTED
    assert cls["us_options"]["policy_execution"]["status"] == Layer.DISABLED_BY_POLICY
    assert cls["us_options"]["account_permission"]["status"] == Layer.OK
    assert cls["us_equity_short"]["execution_eligible"] is False
    assert cls["leveraged_inverse_etp"]["policy_execution"]["status"] == Layer.DENYLISTED
    assert snap["market_data"]["options_opra_snapshots"]["status"] == Layer.DENIED
    assert snap["market_data"]["stocks_sip_latest_quote"]["status"] == Layer.DENIED
    assert "account_number" not in json.dumps(snap)  # no account identifiers persisted
    assert snap["account"]["pdt_fields_reported"] is False
    frame = probe.asset_frame()
    assert set(frame["symbol"]) == {"SPY", "SOXL", "OLDCO", "BTC/USD"}
    assert frame.set_index("symbol").loc["SOXL", "kind"] == InstrumentKind.LEVERAGED_ETF
    path = save_snapshot(snap, tmp_path, frame)
    assert path.is_file() and (tmp_path / "quantlab" / "assets" / "assets_latest.parquet").is_file()


def test_probe_denied_crypto_account_blocks_crypto_layer() -> None:
    snap = CapabilityProbe(_client(_probe_handler(crypto_status="INACTIVE"))).run()
    assert snap["asset_classes"]["crypto_spot"]["account_permission"]["status"] == Layer.DENIED


def test_missing_active_policy_cannot_claim_execution_readiness() -> None:
    snap = CapabilityProbe(_client(_probe_handler())).run()
    assert snap["policy"]["verified"] is False
    assert all(not c["execution_eligible"] for c in snap["asset_classes"].values())
    assert snap["asset_classes"]["us_equity_long"]["research_eligible"] is True


def test_crypto_enable_flag_cannot_make_missing_adapter_executable() -> None:
    snap = CapabilityProbe(_client(_probe_handler()), policy=RiskPolicy(crypto_enabled=True)).run()
    assert snap["policy"]["crypto_enabled"] is True
    assert snap["asset_classes"]["crypto_spot"]["policy_execution"]["status"] == Layer.NOT_IMPLEMENTED
    assert not snap["asset_classes"]["crypto_spot"]["execution_eligible"]


@pytest.mark.parametrize("flag", ["trading_blocked", "account_blocked", "trade_suspended_by_user"])
@pytest.mark.parametrize("value", [True, None, "false"])
def test_account_blocks_and_unknown_permission_flags_fail_closed(flag: str, value: object) -> None:
    base = _probe_handler()

    def handler(req: httpx.Request) -> httpx.Response:
        response = base(req)
        if req.url.path == "/v2/account":
            account = response.json()
            account[flag] = value
            return httpx.Response(200, json=account)
        return response

    snap = CapabilityProbe(_client(handler), policy=RiskPolicy()).run()
    assert all(not c["execution_eligible"] for c in snap["asset_classes"].values())
    assert snap["asset_classes"]["crypto_spot"]["account_permission"]["status"] == Layer.DENIED


@pytest.mark.parametrize(("sym", "name", "exch", "kind"), [
    ("SPY", "State Street SPDR S&P 500 ETF Trust", "ARCA", InstrumentKind.ETF),
    ("QQQ", "Invesco QQQ Trust, Series 1", "NASDAQ", InstrumentKind.ETF),
    ("TQQQ", "ProShares UltraPro QQQ", "NASDAQ", InstrumentKind.LEVERAGED_ETF),
    ("SQQQ", "ProShares UltraPro Short QQQ", "NASDAQ", InstrumentKind.INVERSE_ETF),
    ("NVDL", "GraniteShares 2x Long NVDA Daily ETF", "NASDAQ", InstrumentKind.LEVERAGED_ETF),
    ("UVXY", "ProShares Ultra VIX Short-Term Futures ETF", "BATS", InstrumentKind.VOLATILITY_ETP),
    ("JPM", "JPMorgan Chase & Co.", "NYSE", InstrumentKind.COMMON_STOCK),
    ("NTRS", "Northern Trust Corporation Common Stock", "NASDAQ", InstrumentKind.COMMON_STOCK),
    ("ABCDW", "Acme Acquisition Corp Warrant", "NASDAQ", InstrumentKind.WARRANT),
    ("XYZ", "Xyz Inc", "OTC", InstrumentKind.OTC),
])
def test_classify(sym: str, name: str, exch: str, kind: InstrumentKind) -> None:
    assert classify(sym, name, exch) == kind


def test_leverage_hint() -> None:
    assert leverage_hint("Direxion Daily Semiconductor Bull 3X ETF") == 3.0
    assert leverage_hint("SPDR S&P 500 ETF") is None


def test_paginate_keeps_object_valued_dicts() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("page_token") is None:
            return httpx.Response(200, json={"snapshots": {"A": {"latestQuote": {"bp": 1}}}, "next_page_token": "x"})
        return httpx.Response(200, json={"snapshots": {"B": {"latestQuote": {"bp": 2}}}, "next_page_token": None})

    out = _client(handler).paginate("https://data.alpaca.markets/v1beta1/options/snapshots/SPY", {},
                                    items_key="snapshots")
    assert out == {"A": {"latestQuote": {"bp": 1}}, "B": {"latestQuote": {"bp": 2}}}


@pytest.mark.parametrize(("name", "kind"), [
    ("The RBB Fund, Inc. F/m US Treasury 3 Month Bill Fund", InstrumentKind.ETF),
    ("PIMCO Enhanced Short Maturity Active Exchange-Traded Fund", InstrumentKind.ETF),
    ("iShares Short Treasury Bond ETF", InstrumentKind.ETF),
    ("Vanguard Short-Term Treasury ETF", InstrumentKind.ETF),
    ("ProShares UltraShort Silver", InstrumentKind.INVERSE_ETF),
    ("ProShares Short S&P500", InstrumentKind.INVERSE_ETF),
    ("ProShares Ultra NVDA", InstrumentKind.LEVERAGED_ETF),
    ("Invesco LTD", InstrumentKind.COMMON_STOCK),
    ("WisdomTree, Inc.", InstrumentKind.COMMON_STOCK),
])
def test_fund_detection_regressions(name: str, kind: InstrumentKind) -> None:
    assert classify("XYZ", name, "ARCA") == kind
