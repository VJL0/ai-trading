"""Minimal, strict JWS / OIDC ID-token verification with ``cryptography`` (PyJWT is not installed).

Only the compact serialization with ``RS256`` or ``ES256`` is accepted; ``none``, HMAC and every other algorithm
are refused, as are unknown critical headers. Keys come exclusively from the JWKS document the caller fetched
from the discovery ``jwks_uri`` -- header fields such as ``jku``/``x5u``/``jwk`` are never followed.

:func:`validate_id_token` additionally checks ``iss`` (exact), ``aud`` (must contain the issued client id; with
several audiences ``azp`` must equal it), ``exp``/``nbf``/``iat`` with a small leeway, and ``nonce`` (exact, for
the pending sign-in attempt). Every failure raises :class:`JwsError` -- callers treat that as "not signed in".
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

SUPPORTED_ALGS: Final[frozenset[str]] = frozenset({"RS256", "ES256"})
_MAX_TOKEN_CHARS: Final = 16_384
_MIN_RSA_BITS: Final = 2048


class JwsError(ValueError):
    """The token is malformed, unsigned, signed with an unknown/unsupported key, or its claims are invalid."""


def b64url_decode(text: str) -> bytes:
    if not isinstance(text, str) or any(c in text for c in "=+/ \n"):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise JwsError("invalid base64url segment")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise JwsError("invalid base64url segment") from None


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _json_segment(segment: str, what: str) -> dict[str, Any]:
    try:
        value = json.loads(b64url_decode(segment))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise JwsError(f"{what} is not JSON") from None
    if not isinstance(value, dict):
        raise JwsError(f"{what} is not a JSON object")
    return cast(dict[str, Any], value)


def _int_from_b64(value: Any, what: str) -> int:
    if not isinstance(value, str) or not value:
        raise JwsError(f"JWK member {what} missing")
    return int.from_bytes(b64url_decode(value), "big")


def _select_key(header: Mapping[str, Any], jwks: Mapping[str, Any]) -> Mapping[str, Any]:
    keys_raw = jwks.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise JwsError("JWKS has no keys")
    keys = [cast(Mapping[str, Any], k) for k in cast(list[Any], keys_raw) if isinstance(k, Mapping)]
    alg = header["alg"]
    kty = "RSA" if alg == "RS256" else "EC"
    usable = [
        k for k in keys
        if k.get("kty") == kty and k.get("use", "sig") == "sig" and k.get("alg", alg) == alg
        and (kty != "EC" or k.get("crv") == "P-256")
    ]
    kid = header.get("kid")
    if kid is not None:
        if not isinstance(kid, str):
            raise JwsError("kid must be a string")
        matches = [k for k in usable if k.get("kid") == kid]
        if len(matches) != 1:
            raise JwsError(f"no unique JWKS key for kid {kid[:64]!r}")
        return matches[0]
    if len(usable) != 1:
        raise JwsError("token has no kid and the JWKS key is ambiguous")
    return usable[0]


def _verify_signature(alg: str, key: Mapping[str, Any], signing_input: bytes, signature: bytes) -> None:
    try:
        if alg == "RS256":
            pub = rsa.RSAPublicNumbers(_int_from_b64(key.get("e"), "e"), _int_from_b64(key.get("n"), "n")).public_key()
            if pub.key_size < _MIN_RSA_BITS:
                raise JwsError("RSA key is too small")
            pub.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
        elif alg == "ES256":
            if len(signature) != 64:
                raise JwsError("ES256 signature must be 64 bytes (r||s)")
            pub_ec = ec.EllipticCurvePublicNumbers(
                _int_from_b64(key.get("x"), "x"), _int_from_b64(key.get("y"), "y"), ec.SECP256R1()
            ).public_key()
            der = encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
            pub_ec.verify(der, signing_input, ec.ECDSA(hashes.SHA256()))
        else:  # pragma: no cover - guarded by the caller
            raise JwsError(f"unsupported alg {alg}")
    except InvalidSignature:
        raise JwsError("signature verification failed") from None
    except ValueError as exc:  # invalid point / key material
        if isinstance(exc, JwsError):
            raise
        raise JwsError(f"invalid JWK key material ({type(exc).__name__})") from None


def verify_jws(token: str, jwks: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a compact JWS against ``jwks``; returns ``(header, claims)``."""
    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_CHARS:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise JwsError("token missing or too long")
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        raise JwsError("token is not a signed compact JWS")
    header = _json_segment(parts[0], "JWS header")
    alg = header.get("alg")
    if alg not in SUPPORTED_ALGS:
        raise JwsError(f"unsupported or missing alg {str(alg)[:16]!r} (only RS256/ES256)")
    crit = header.get("crit")
    if crit is not None:
        raise JwsError("critical JWS header extensions are not supported")
    if header.get("typ") not in (None, "JWT", "jwt", "at+jwt"):
        raise JwsError("unexpected typ header")
    key = _select_key(header, jwks)
    _verify_signature(str(alg), key, f"{parts[0]}.{parts[1]}".encode("ascii"), b64url_decode(parts[2]))
    claims = _json_segment(parts[1], "JWS payload")
    return header, claims


def _numeric(claims: Mapping[str, Any], name: str, *, required: bool) -> float | None:
    v = claims.get(name)
    if v is None:
        if required:
            raise JwsError(f"claim {name} missing")
        return None
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise JwsError(f"claim {name} is not a NumericDate")
    return float(v)


def validate_id_token(
    token: str,
    jwks: Mapping[str, Any],
    *,
    issuer: str,
    audience: str,
    nonce: str | None,
    now: datetime,
    leeway_s: float = 60.0,
) -> dict[str, Any]:
    """Verify signature + standard OIDC ID-token claims. Returns the claims."""
    _header, claims = verify_jws(token, jwks)
    if claims.get("iss") != issuer:
        raise JwsError("id_token issuer mismatch")
    aud = claims.get("aud")
    auds: Sequence[Any] = [aud] if isinstance(aud, str) else cast(list[Any], aud) if isinstance(aud, list) else []
    if audience not in auds:
        raise JwsError("id_token audience does not contain the issued client id")
    if len(auds) > 1 and claims.get("azp") != audience:
        raise JwsError("id_token has several audiences but azp is not the issued client id")
    t = now.timestamp()
    exp = _numeric(claims, "exp", required=True)
    assert exp is not None
    if exp <= t - leeway_s:
        raise JwsError("id_token expired")
    nbf = _numeric(claims, "nbf", required=False)
    if nbf is not None and nbf > t + leeway_s:
        raise JwsError("id_token not yet valid")
    iat = _numeric(claims, "iat", required=False)
    if iat is not None and iat > t + leeway_s:
        raise JwsError("id_token issued in the future")
    if nonce is not None:
        got = claims.get("nonce")
        if not isinstance(got, str) or not hmac.compare_digest(got, nonce):
            raise JwsError("id_token nonce mismatch")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise JwsError("id_token has no subject")
    return claims


__all__ = ["SUPPORTED_ALGS", "JwsError", "b64url_decode", "b64url_encode", "validate_id_token", "verify_jws"]
