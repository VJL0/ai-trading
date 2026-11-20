"""Generate the cross-language ECDSA P-256 intent test vectors consumed by trading/lean/LatStrategies.Tests.

Signed by the REAL Python signer (packages.core.signing.IntentSigner) over the REAL contract signing string
(packages.contracts.ApprovedOrderIntent.signing_string()). Throw-away keys: only PUBLIC keys are written.

    uv run --frozen python trading/lean/tools/gen_intent_vectors.py [out.json]
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.contracts.enums import GatewayKind, OrderType, Side, TimeInForce, TradingMode  # noqa: E402
from packages.contracts.models import ApprovedOrderIntent  # noqa: E402
from packages.core.signing import IntentSigner  # noqa: E402

DEFAULT_OUT = ROOT / "trading" / "lean" / "LatStrategies.Tests" / "vectors" / "intent_vectors.json"
CREATED = datetime(2026, 11, 5, 14, 30, 0, tzinfo=UTC)
NOW = CREATED + timedelta(minutes=1)


def make_intent(
    signer: IntentSigner,
    *,
    n: int,
    side: Side,
    quantity: str,
    limit: str,
    symbol: str = "SPY",
    gateway: GatewayKind = GatewayKind.LEAN_PAPER_SIM,
    order_type: OrderType = OrderType.LIMIT,
    ttl_s: int = 300,
    created: datetime = CREATED,
    human: bool = False,
) -> ApprovedOrderIntent:
    fields: dict[str, Any] = {
        "intent_id": f"int_vec_{n:03d}",
        "client_order_id": f"lat-sim-vec-{n:03d}",
        "proposal_id": f"prop_vec_{n:03d}",
        "decision_id": f"dec_vec_{n:03d}",
        "causal_id": f"cz_vec_{n:03d}",
        "mode": TradingMode.PAPER,
        "gateway": gateway,
        "symbol": symbol,
        "side": side,
        "quantity": Decimal(quantity),
        "order_type": order_type,
        "limit_price": Decimal(limit),
        "time_in_force": TimeInForce.DAY,
        "extended_hours": False,
        "policy_version": "pol_vectors_v1",
        "created_at": created,
        "expires_at": created + timedelta(seconds=ttl_s),
        "requires_human_approval": human,
        "human_approval_id": "hap_vec" if human else None,
        "key_id": signer.key_id,
        "signature": "0" * 128,
    }
    unsigned = ApprovedOrderIntent(**fields)
    fields["signature"] = signer.sign(unsigned.signing_string())
    intent = ApprovedOrderIntent(**fields)
    assert signer.verifier.verify(intent.signing_string(), intent.signature)
    return intent


def case(name: str, intent: ApprovedOrderIntent, *, signing_string: str | None = None, signature: str | None = None,
         expect_signature_valid: bool, expect_gate: str) -> dict[str, Any]:
    return {
        "name": name,
        "signing_string": signing_string if signing_string is not None else intent.signing_string(),
        "signature": signature if signature is not None else intent.signature,
        "intent_id": intent.intent_id,
        "client_order_id": intent.client_order_id,
        "symbol": intent.symbol,
        "side": intent.side.value,
        "quantity": format(intent.quantity, "f"),
        "limit_price": format(intent.limit_price, "f"),
        "expect_signature_valid": expect_signature_valid,
        "expect_gate": expect_gate,
    }


def build() -> dict[str, Any]:
    signer = IntentSigner.generate()
    other = IntentSigner.generate()
    buy = make_intent(signer, n=1, side=Side.BUY, quantity="10", limit="450.10")
    sell = make_intent(signer, n=2, side=Side.SELL, quantity="5", limit="449.90")
    big_sell = make_intent(signer, n=3, side=Side.SELL, quantity="50", limit="449.90")
    expired = make_intent(signer, n=4, side=Side.BUY, quantity="10", limit="450.10", ttl_s=30)
    market = make_intent(signer, n=5, side=Side.BUY, quantity="10", limit="450.10", order_type=OrderType.MARKET)
    wrong_gw = make_intent(signer, n=6, side=Side.BUY, quantity="10", limit="450.10", gateway=GatewayKind.LEAN_ALPACA_PAPER)
    foreign = make_intent(other, n=7, side=Side.BUY, quantity="10", limit="450.10")
    tampered = buy.signing_string().replace("|quantity=10|", "|quantity=1000|")
    assert tampered != buy.signing_string()
    return {
        "generated_by": "packages.core.signing.IntentSigner over packages.contracts.ApprovedOrderIntent.signing_string()",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "signature_alg": "ECDSA_P256_SHA256",
        "signature_encoding": "hex(r||s) IEEE-P1363, 64 bytes",
        "public_key_pem": signer.verifier.public_pem().decode(),
        "key_id": signer.key_id,
        "other_public_key_pem": other.verifier.public_pem().decode(),
        "other_key_id": other.key_id,
        "now_utc": NOW.isoformat().replace("+00:00", "Z"),
        "cases": [
            case("valid_buy", buy, expect_signature_valid=True, expect_gate="allow"),
            case("valid_sell", sell, expect_signature_valid=True, expect_gate="allow_if_held"),
            case("sell_more_than_held", big_sell, expect_signature_valid=True, expect_gate="refuse_short"),
            case("expired", expired, expect_signature_valid=True, expect_gate="refuse_expired"),
            case("market_order_intent", market, expect_signature_valid=True, expect_gate="refuse_order_type"),
            case("wrong_gateway", wrong_gw, expect_signature_valid=True, expect_gate="refuse_gateway"),
            case("signed_by_other_key", foreign, expect_signature_valid=False, expect_gate="refuse_signature"),
            case("tampered_quantity", buy, signing_string=tampered, expect_signature_valid=False, expect_gate="refuse_signature"),
            case("bitflip_signature", buy, signature=("0" if buy.signature[0] != "0" else "1") + buy.signature[1:],
                 expect_signature_valid=False, expect_gate="refuse_signature"),
        ],
    }


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
