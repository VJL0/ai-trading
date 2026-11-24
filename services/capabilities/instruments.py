"""Instrument classification from Alpaca asset metadata (name/symbol/exchange heuristics, documented and testable).

Alpaca's asset endpoint has no ETF/leverage field, so the class is inferred from the issuer name. The rules are
conservative for *execution*: anything that looks leveraged/inverse/volatility-linked is flagged even if the name
is ambiguous, and the hard denylist in ``services.risk.limits`` still applies independently.
"""

from __future__ import annotations

import re
from enum import StrEnum

from packages.contracts.policy import DEFAULT_DENYLIST


class InstrumentKind(StrEnum):
    COMMON_STOCK = "COMMON_STOCK"
    ETF = "ETF"
    LEVERAGED_ETF = "LEVERAGED_ETF"
    INVERSE_ETF = "INVERSE_ETF"
    VOLATILITY_ETP = "VOLATILITY_ETP"
    ETN = "ETN"
    PREFERRED = "PREFERRED"
    WARRANT = "WARRANT"
    RIGHT = "RIGHT"
    UNIT = "UNIT"
    OTC = "OTC"
    CRYPTO = "CRYPTO"
    OTHER = "OTHER"


_ETF_WORD = re.compile(r"\b(ETF|ETFS|ETP)\b", re.I)
_FUND_BRANDS = re.compile(
    r"\b(ISHARES|SPDR|VANGUARD|INVESCO|PROSHARES|DIREXION|GLOBAL X|WISDOMTREE|VANECK|FIRST TRUST|SCHWAB|"
    r"GRANITESHARES|SELECT SECTOR|DEFIANCE|TRADR|ROUNDHILL|YIELDMAX|AMPLIFY|T-REX|LEVERAGE SHARES|MICROSECTORS|"
    r"INNOVATOR|PACER|ARK|GRAYSCALE|BITWISE|21SHARES|FRANKLIN|ABRDN|SPROTT|UNITED STATES OIL|TEUCRIUM|KRANESHARES|"
    r"XTRACKERS|COLUMBIA|ALPS|MASSMUTUAL|JANUS|NUVEEN|DIMENSIONAL|AVANTIS|FIDELITY|JPMORGAN|GOLDMAN SACHS)\b", re.I)
_FUND_NOUNS = re.compile(r"\b(FUND|TRUST|SHARES|INDEX|PORTFOLIO|STRATEGY|PREMIUM INCOME|INCOME|BOND|TREASURY)\b",
                         re.I)


_ETF_ONLY_BRANDS = re.compile(
    r"\b(PROSHARES|DIREXION|ISHARES|SPDR|GRANITESHARES|GLOBAL X|VANECK|ROUNDHILL|YIELDMAX|DEFIANCE|TRADR|T-REX|"
    r"LEVERAGE SHARES|MICROSECTORS|INNOVATOR|PACER|AMPLIFY|KRANESHARES|XTRACKERS|21SHARES|ULTRAPRO|ULTRASHORT)\b",
    re.I)
_CORPORATE = re.compile(r"\b(INC|INCORPORATED|CORP|CORPORATION|HOLDINGS|LTD|LIMITED|PLC|GROUP)\b\.?", re.I)
_EXCHANGE_TRADED = re.compile(r"EXCHANGE[- ]TRADED", re.I)
_CASHLIKE_FUND = re.compile(r"\bFUND\b.*\b(TREASURY|BILL|BOND|MUNICIPAL|INCOME|INDEX|MONEY MARKET)\b|"
                            r"\b(TREASURY|BILL|BOND|MUNICIPAL|INCOME|INDEX)\b.*\bFUND\b", re.I)


def _is_fund(text: str) -> bool:
    if _ETF_WORD.search(text) or _EXCHANGE_TRADED.search(text) or _CASHLIKE_FUND.search(text):
        return True
    if _ETF_ONLY_BRANDS.search(text) and not _CORPORATE.search(text.replace("Fund, Inc.", "")):
        return True
    return bool(_FUND_BRANDS.search(text) and _FUND_NOUNS.search(text))


_LEVERAGE = re.compile(r"(\b[2-5](\.\d+)?X\b|\b[2-5](\.\d+)?x\b|ULTRA|LEVERAGED|DAILY TARGET|BULL\b|DAILY 2X|DAILY 3X|"
                       r"\b(200|300)%)", re.I)
_NOT_INVERSE_AFTER_SHORT = r"(TERM|MATURITY|DURATION|DATED|DUR\b|TREASURY|BOND|INCOME|CORPORATE|MUNI|HIGH YIELD|VOL)"
_INVERSE = re.compile(rf"(INVERSE|ULTRASHORT|\bSHORT\b(?![\s-]*{_NOT_INVERSE_AFTER_SHORT})|\bBEAR\b|-1X|"
                      r"\b-[1-3]X\b)", re.I)
_VOL = re.compile(r"(VIX|VOLATILITY)", re.I)
_ETN = re.compile(r"\bETN\b|EXCHANGE TRADED NOTE", re.I)
_PREF = re.compile(r"(PREFERRED|PFD|DEPOSITARY SHARES|\bSERIES [A-Z]\b|NOTES DUE|\bPERP\b|FIXED[- ]RATE)", re.I)
_WARRANT = re.compile(r"WARRANT", re.I)
_RIGHT = re.compile(r"\bRIGHTS?\b", re.I)
_UNIT = re.compile(r"\bUNITS?\b", re.I)


def classify(symbol: str, name: str | None, exchange: str | None, asset_class: str = "us_equity") -> InstrumentKind:
    if asset_class == "crypto":
        return InstrumentKind.CRYPTO
    if (exchange or "").upper() == "OTC":
        return InstrumentKind.OTC
    sym = symbol.upper()
    text = (name or "").strip()
    if sym in DEFAULT_DENYLIST:
        if _VOL.search(text) or sym in {"UVXY", "SVXY", "VXX", "VIXY"}:
            return InstrumentKind.VOLATILITY_ETP
        if _INVERSE.search(text) or sym in {"SH", "PSQ", "DOG", "RWM", "SQQQ", "SPXS", "SPXU", "SDS", "QID", "SDOW",
                                             "TZA", "LABD", "SOXS", "TECS", "FAZ", "DUST", "JDST", "TMV", "KOLD",
                                             "SCO", "YANG", "ERY"}:
            return InstrumentKind.INVERSE_ETF
        return InstrumentKind.LEVERAGED_ETF
    if _WARRANT.search(text):
        return InstrumentKind.WARRANT
    if _RIGHT.search(text) and not _is_fund(text):
        return InstrumentKind.RIGHT
    if _UNIT.search(text) and not _is_fund(text):
        return InstrumentKind.UNIT
    if _PREF.search(text):
        return InstrumentKind.PREFERRED
    if _ETN.search(text):
        return InstrumentKind.ETN
    is_fund = _is_fund(text)
    if is_fund and _VOL.search(text):
        return InstrumentKind.VOLATILITY_ETP
    if is_fund and _INVERSE.search(text):
        return InstrumentKind.INVERSE_ETF
    if is_fund and _LEVERAGE.search(text):
        return InstrumentKind.LEVERAGED_ETF
    if is_fund:
        return InstrumentKind.ETF
    return InstrumentKind.COMMON_STOCK


EQUITY_RESEARCH_KINDS = frozenset({InstrumentKind.COMMON_STOCK, InstrumentKind.ETF, InstrumentKind.LEVERAGED_ETF,
                                   InstrumentKind.INVERSE_ETF, InstrumentKind.VOLATILITY_ETP, InstrumentKind.ETN})
"""Kinds included in research panels (OTC, warrants, rights, units and preferreds are excluded: thin, unusual
settlement or not tradable by this stack)."""

LEVERAGE_FACTOR_HINT = re.compile(r"\b([2-5])(?:\.\d+)?X\b", re.I)


def leverage_hint(name: str | None) -> float | None:
    """Stated daily leverage multiple from the fund name (e.g. '3X' -> 3.0); None when not stated."""
    m = LEVERAGE_FACTOR_HINT.search(name or "")
    return float(m.group(1)) if m else None
