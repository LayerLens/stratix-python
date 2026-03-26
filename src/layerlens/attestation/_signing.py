"""HMAC-SHA256 signing for attestation envelopes."""

from __future__ import annotations

import hmac as hmac_mod
import base64
import hashlib


def hmac_sign(secret: bytes, data: bytes) -> str:
    """Sign data with HMAC-SHA256, returning a base64-encoded signature."""
    sig = hmac_mod.new(secret, data, hashlib.sha256).digest()
    return base64.b64encode(sig).decode("ascii")


def hmac_verify(secret: bytes, data: bytes, signature: str) -> bool:
    """Verify a base64-encoded HMAC-SHA256 signature. Timing-safe."""
    expected = hmac_sign(secret, data)
    return hmac_mod.compare_digest(signature, expected)
