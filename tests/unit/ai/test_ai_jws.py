"""JWS / OIDC ID-token verification (RS256 + ES256 with ``cryptography``): good tokens pass, everything else fails."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from services.ai.chatgpt_fake import FakeChatGptServer
from services.ai.jws import JwsError, b64url_encode, validate_id_token, verify_jws

NOW = datetime(2026, 11, 5, 15, 30, tzinfo=UTC)
ISS = "https://auth.openai-fake.invalid"


def _claims(**over: object) -> dict[str, object]:
    base: dict[str, object] = {"iss": ISS, "aud": "oaiapp_x", "sub": "user-1", "nonce": "n-1",
                               "iat": int(NOW.timestamp()) - 5, "exp": int(NOW.timestamp()) + 600}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


@pytest.fixture(params=["ES256", "RS256"])
def server(request: pytest.FixtureRequest) -> FakeChatGptServer:
    return FakeChatGptServer(alg=request.param)


def _validate(server: FakeChatGptServer, token: str, **kw: object) -> dict[str, object]:
    args: dict[str, object] = {"issuer": ISS, "audience": "oaiapp_x", "nonce": "n-1", "now": NOW}
    args.update(kw)
    return validate_id_token(token, server.jwks(), **args)  # type: ignore[arg-type]


def test_valid_tokens_verify(server: FakeChatGptServer) -> None:
    claims = _validate(server, server.sign(_claims()))
    assert claims["sub"] == "user-1"
    header, payload = verify_jws(server.sign(_claims()), server.jwks())
    assert header["alg"] == server.alg and payload["aud"] == "oaiapp_x"


def test_tampered_payload_or_signature_fails(server: FakeChatGptServer) -> None:
    token = server.sign(_claims())
    head, _payload, sig = token.split(".")
    forged = f"{head}.{b64url_encode(json.dumps(_claims(sub='attacker')).encode())}.{sig}"
    with pytest.raises(JwsError, match="signature"):
        _validate(server, forged)
    other = FakeChatGptServer(alg=server.alg)  # same kid, different key
    with pytest.raises(JwsError, match="signature"):
        _validate(server, other.sign(_claims()))


@pytest.mark.parametrize("alg", ["none", "HS256", "RS512", "ES384", None])
def test_unsupported_algorithms_are_refused(server: FakeChatGptServer, alg: str | None) -> None:
    header = {"alg": alg, "kid": server.kid} if alg else {"kid": server.kid}
    body = b64url_encode(json.dumps(_claims()).encode())
    token = f"{b64url_encode(json.dumps(header).encode())}.{body}.{b64url_encode(b'x' * 64)}"
    with pytest.raises(JwsError, match="alg"):
        _validate(server, token)


def test_unsigned_or_malformed_tokens_fail(server: FakeChatGptServer) -> None:
    good = server.sign(_claims())
    for bad in ("", "a.b", good + ".x", good.rsplit(".", 1)[0] + ".", "e30.e30.@@@", "x" * 20_000):
        with pytest.raises(JwsError):
            _validate(server, bad)


def test_key_selection(server: FakeChatGptServer) -> None:
    unknown_kid = server.sign(_claims(), header={"alg": server.alg, "kid": "rotated-away"})
    with pytest.raises(JwsError, match="kid"):
        _validate(server, unknown_kid)
    no_kid = server.sign(_claims(), header={"alg": server.alg})
    assert _validate(server, no_kid)["sub"] == "user-1"  # single usable key
    jwks = server.jwks()
    jwks["keys"].append({**jwks["keys"][0], "kid": "second"})
    with pytest.raises(JwsError, match="ambiguous"):
        validate_id_token(no_kid, jwks, issuer=ISS, audience="oaiapp_x", nonce="n-1", now=NOW)
    with pytest.raises(JwsError, match="critical"):
        _validate(server, server.sign(_claims(), header={"alg": server.alg, "kid": server.kid, "crit": ["exp"]}))


@pytest.mark.parametrize(("over", "kw", "match"), [
    ({"iss": "https://evil.example"}, {}, "issuer"),
    ({"aud": "oaiapp_other"}, {}, "audience"),
    ({"aud": ["oaiapp_x", "other"]}, {}, "azp"),
    ({"exp": int(NOW.timestamp()) - 3600}, {}, "expired"),
    ({"nbf": int(NOW.timestamp()) + 3600}, {}, "not yet valid"),
    ({"iat": int(NOW.timestamp()) + 3600}, {}, "future"),
    ({"nonce": "other"}, {}, "nonce"),
    ({"nonce": None}, {}, "nonce"),
    ({"sub": None}, {}, "subject"),
    ({"exp": "soon"}, {}, "NumericDate"),
    ({"exp": None}, {}, "exp missing"),
])
def test_claim_validation(server: FakeChatGptServer, over: dict[str, object], kw: dict[str, object], match: str) -> None:
    with pytest.raises(JwsError, match=match):
        _validate(server, server.sign(_claims(**over)), **kw)


def test_multiple_audiences_need_matching_azp(server: FakeChatGptServer) -> None:
    token = server.sign(_claims(aud=["oaiapp_x", "other"], azp="oaiapp_x"))
    assert _validate(server, token)["azp"] == "oaiapp_x"


def test_refresh_id_token_without_nonce_check(server: FakeChatGptServer) -> None:
    token = server.sign(_claims(nonce=None))
    assert _validate(server, token, nonce=None)["sub"] == "user-1"


def test_small_rsa_keys_are_refused() -> None:
    srv = FakeChatGptServer(alg="RS256")
    srv._rsa = rsa.generate_private_key(public_exponent=65537, key_size=1024)  # noqa: S505  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(JwsError, match="too small"):
        _validate(srv, srv.sign(_claims()))
