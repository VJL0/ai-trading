"""Asymmetric intent signing (ECDSA P-256 / SHA-256, IEEE-P1363 r||s encoding).

The risk engine holds the private key (``var/<ns>/keys/intent_signing.pem``, 0600). The execution
boundary — the Python gateway *and* the C# LEAN gated brokerage model — hold only the public key, so
neither a LEAN strategy nor any other component can mint an order intent.

Keys are per namespace: a paper-signed intent can never verify under the live key.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

from .canonical import sha256_hex

SIGNATURE_ALG = "ECDSA_P256_SHA256"
PRIVATE_KEY_FILE = "intent_signing.pem"
PUBLIC_KEY_FILE = "intent_signing.pub.pem"


def _key_id(public_key: ec.EllipticCurvePublicKey) -> str:
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return "k_" + sha256_hex(der)[:16]


class IntentVerifier:
    def __init__(self, public_key: ec.EllipticCurvePublicKey) -> None:
        self._pub = public_key
        self.key_id = _key_id(public_key)

    @classmethod
    def from_pem(cls, pem: bytes) -> IntentVerifier:
        key = serialization.load_pem_public_key(pem)
        if not isinstance(key, ec.EllipticCurvePublicKey):
            raise TypeError("intent verification key must be an EC public key")
        return cls(key)

    def public_pem(self) -> bytes:
        return self._pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    def verify(self, message: str, signature_hex: str) -> bool:
        try:
            raw = bytes.fromhex(signature_hex)
        except ValueError:
            return False
        if len(raw) != 64:
            return False
        der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        try:
            self._pub.verify(der, message.encode(), ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            return False
        return True


class IntentSigner:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey) -> None:
        self._key = private_key
        self.verifier = IntentVerifier(private_key.public_key())
        self.key_id = self.verifier.key_id

    @classmethod
    def generate(cls) -> IntentSigner:
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def load_or_create(cls, keys_dir: Path, *, allow_create: bool = True) -> IntentSigner:
        keys_dir.mkdir(parents=True, exist_ok=True)
        priv_path = keys_dir / PRIVATE_KEY_FILE
        if priv_path.exists():
            key = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
            if not isinstance(key, ec.EllipticCurvePrivateKey):
                raise TypeError("intent signing key must be EC")
            signer = cls(key)
        else:
            if not allow_create:
                raise FileNotFoundError(f"no intent signing key at {priv_path}")
            signer = cls.generate()
            pem = signer._key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
            fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(pem)
        (keys_dir / PUBLIC_KEY_FILE).write_bytes(signer.verifier.public_pem())
        return signer

    def sign(self, message: str) -> str:
        der = self._key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()


def load_verifier(keys_dir: Path) -> IntentVerifier:
    return IntentVerifier.from_pem((keys_dir / PUBLIC_KEY_FILE).read_bytes())
