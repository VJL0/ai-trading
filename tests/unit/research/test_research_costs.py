"""Cost model: exact Decimal expectations (hand-computed in the comments)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.contracts.enums import Side
from services.research.costs import (
    CostModel,
    FeeSchedule,
    LiquidityBucket,
    SlippageModel,
    SpreadModel,
    spread_bps_from_quote,
)


def test_buy_has_no_regulatory_fees_and_uses_unknown_liquidity_defaults() -> None:
    est = CostModel().estimate("AAA", Side.BUY, 100, "50")
    # notional 5000; commission 0; no SEC/TAF on buys
    assert est.notional == Decimal("5000")
    assert est.sec_fee_usd == 0 and est.finra_taf_usd == 0 and est.commission_usd == 0
    assert est.fees_usd == 0 and est.fees_bps == 0
    # unknown liquidity: full spread 30 bps -> crossing half = 15 bps = $7.50
    assert est.quoted_spread_bps == Decimal("30")
    assert est.spread_source == "UNKNOWN_LIQUIDITY_DEFAULT"
    assert est.spread_cost_bps == Decimal("15")
    assert est.spread_cost_usd == Decimal("7.5")
    # slippage: fixed 2 + unknown-ADV impact 10 = 12 bps = $6.00
    assert est.participation is None
    assert est.slippage_bps == Decimal("12")
    assert est.slippage_usd == Decimal("6")
    assert est.total_usd == Decimal("13.5")
    assert est.total_bps == Decimal("27")


def test_sell_pays_sec_and_taf_rounded_up_to_cent_with_quote_spread_and_sqrt_impact() -> None:
    est = CostModel().estimate("AAA", Side.SELL, 100, "50", spread_bps="4", adv_shares=1_000_000)
    # SEC: 5000 * 0.0000278 = 0.139 -> ceil to cent 0.14 ; TAF: 100 * 0.000166 = 0.0166 -> 0.02
    assert est.sec_fee_usd == Decimal("0.14")
    assert est.finra_taf_usd == Decimal("0.02")
    assert est.fees_usd == Decimal("0.16")
    assert est.fees_bps == Decimal("0.32")  # 0.16 / 5000 * 1e4
    # quote spread 4 bps full -> 2 bps crossing = $1.00
    assert est.spread_source == "QUOTE"
    assert est.spread_cost_bps == Decimal("2")
    assert est.spread_cost_usd == Decimal("1")
    # participation 100/1e6 = 1e-4 ; sqrt = 0.01 ; impact = 100 * 0.01 = 1 bps ; slippage 3 bps = $1.50
    assert est.participation == Decimal("0.0001")
    assert est.impact_bps == Decimal("1")
    assert est.slippage_bps == Decimal("3")
    assert est.slippage_usd == Decimal("1.5")
    assert est.total_usd == Decimal("2.66")
    assert est.total_bps == Decimal("5.32")


def test_finra_taf_is_capped_per_trade() -> None:
    est = CostModel().estimate("AAA", Side.SELL, 100_000, "10", spread_bps=0, adv_shares=10_000_000)
    # raw TAF 100000 * 0.000166 = 16.60 > cap 8.30 ; SEC 1,000,000 * 0.0000278 = 27.80 exactly
    assert est.finra_taf_usd == Decimal("8.30")
    assert est.sec_fee_usd == Decimal("27.80")
    assert "FINRA TAF capped at max per trade" in est.notes


def test_spread_bucket_by_dollar_adv_when_quote_absent() -> None:
    m = CostModel()
    # ADV 10M shares * $50 = $500M -> LARGE bucket 5 bps full -> 2.5 bps crossing
    est = m.estimate("AAA", Side.BUY, 10, "50", adv_shares=10_000_000)
    assert est.quoted_spread_bps == Decimal("5")
    assert est.spread_cost_bps == Decimal("2.5")
    assert est.spread_source.startswith("LIQUIDITY_BUCKET:LARGE")
    # ADV 100k shares * $50 = $5M -> ILLIQUID 50 bps full
    est2 = m.estimate("AAA", Side.BUY, 10, "50", adv_shares=100_000)
    assert est2.quoted_spread_bps == Decimal("50")
    # ADV 30M * $50 = $1.5B -> MEGA 2 bps
    est3 = m.estimate("AAA", Side.BUY, 10, "50", adv_shares=30_000_000)
    assert est3.quoted_spread_bps == Decimal("2")


def test_commission_configurable_with_minimum() -> None:
    fees = FeeSchedule(commission_per_share=Decimal("0.005"), commission_min_per_order=Decimal("1"))
    est = CostModel(fees=fees).estimate("AAA", Side.BUY, 100, "20", spread_bps=0, adv_shares=1_000_000)
    assert est.commission_usd == Decimal("1")  # 100 * 0.005 = 0.50 < min 1.00
    est2 = CostModel(fees=fees).estimate("AAA", Side.BUY, 1000, "20", spread_bps=0, adv_shares=1_000_000)
    assert est2.commission_usd == Decimal("5")


def test_zero_sec_rate_and_no_rounding_configurable() -> None:
    fees = FeeSchedule(sec_fee_rate=Decimal(0), round_fees_up_to_cent=False)
    est = CostModel(fees=fees).estimate("AAA", Side.SELL, 100, "50", spread_bps=0, adv_shares=1_000_000)
    assert est.sec_fee_usd == 0
    assert est.finra_taf_usd == Decimal("0.016600")


def test_round_trip_sums_both_legs_relative_to_entry_notional() -> None:
    m = CostModel()
    rt = m.estimate_round_trip("AAA", 100, "50", spread_bps="4", adv_shares=1_000_000)
    # entry: spread $1 + slippage $1.50 ; exit: + fees $0.16 -> total $5.16 on $5000 = 10.32 bps
    assert rt.total_usd == Decimal("5.16")
    assert rt.total_bps == Decimal("10.32")
    assert rt.spread_bps == Decimal("4")
    assert rt.slippage_bps == Decimal("6")
    assert rt.fees_bps == Decimal("0.32")
    assert rt.spread_bps + rt.fees_bps + rt.slippage_bps == rt.total_bps


def test_quote_spread_helper() -> None:
    # (100.02 - 99.98) / 100 * 1e4 = 4 bps
    assert spread_bps_from_quote("99.98", "100.02") == Decimal("4")
    with pytest.raises(ValueError, match="crossed"):
        spread_bps_from_quote("100.02", "99.98")


@pytest.mark.parametrize("bad", [1.5, float("nan")])
def test_floats_rejected(bad: float) -> None:
    with pytest.raises(TypeError):
        CostModel().estimate("AAA", Side.BUY, bad, "50")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        CostModel().estimate("AAA", Side.BUY, 10, bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        CostModel().estimate("AAA", Side.BUY, 10, "50", spread_bps=bad)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", Decimal("NaN"), Decimal("-Infinity")])
def test_non_finite_decimals_rejected(bad: Decimal | str) -> None:
    m = CostModel()
    with pytest.raises(ValueError, match="finite"):
        m.estimate("AAA", Side.BUY, bad, "50")
    with pytest.raises(ValueError, match="finite"):
        m.estimate("AAA", Side.BUY, 10, bad)
    with pytest.raises(ValueError, match="finite"):
        m.estimate("AAA", Side.BUY, 10, "50", spread_bps=bad)
    with pytest.raises(ValueError, match="finite"):
        m.estimate("AAA", Side.SELL, 10, "50", adv_shares=bad)


def test_invalid_inputs_rejected() -> None:
    m = CostModel()
    with pytest.raises(ValueError, match="quantity"):
        m.estimate("AAA", Side.BUY, 0, "50")
    with pytest.raises(ValueError, match="price"):
        m.estimate("AAA", Side.BUY, 1, "0")
    with pytest.raises(ValueError, match="spread"):
        m.estimate("AAA", Side.BUY, 1, "10", spread_bps="-1")
    with pytest.raises(ValueError, match="non-negative"):
        FeeSchedule(sec_fee_rate=Decimal("-0.1"))
    with pytest.raises(ValueError, match="descending"):
        SpreadModel(buckets=(LiquidityBucket(Decimal(0), Decimal(5), "a"), LiquidityBucket(Decimal(10), Decimal(2), "b")))


def test_participation_above_cap_is_flagged_and_impact_grows_with_sqrt() -> None:
    m = CostModel(slippage=SlippageModel(fixed_bps=Decimal(0)))
    small = m.estimate("AAA", Side.BUY, 100, "10", spread_bps=0, adv_shares=10_000)  # 1% -> 10 bps
    big = m.estimate("AAA", Side.BUY, 1600, "10", spread_bps=0, adv_shares=10_000)  # 16% -> 40 bps
    assert small.impact_bps == Decimal("10")
    assert big.impact_bps == Decimal("40")
    assert any("exceeds max" in n for n in big.notes)


def test_describe_documents_verify_current_rate() -> None:
    d = CostModel().describe()
    fees = d["fees"]
    assert isinstance(fees, dict)
    assert "VERIFY CURRENT RATE" in str(fees["sec_fee_asof"])
    assert "VERIFY CURRENT RATE" in str(fees["finra_taf_asof"])
