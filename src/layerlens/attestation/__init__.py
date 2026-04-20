from __future__ import annotations

from ._hash import compute_hash
from ._chain import HashChain
from ._verify import (
    TamperingResult,
    ChainVerification,
    TrialVerification,
    verify_chain,
    verify_trial,
    detect_tampering,
)
from ._signing import hmac_sign, hmac_verify
from ._envelope import HashScope, AttestationEnvelope

__all__ = [
    "AttestationEnvelope",
    "ChainVerification",
    "HashChain",
    "HashScope",
    "TamperingResult",
    "TrialVerification",
    "compute_hash",
    "detect_tampering",
    "hmac_sign",
    "hmac_verify",
    "verify_chain",
    "verify_trial",
]
