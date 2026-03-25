from __future__ import annotations

from ._hash import compute_hash
from ._chain import HashChain
from ._verify import (
    TamperingResult,
    ChainVerification,
    verify_chain,
    verify_trial,
    detect_tampering,
)
from ._envelope import HashScope, AttestationEnvelope

__all__ = [
    "AttestationEnvelope",
    "ChainVerification",
    "HashChain",
    "HashScope",
    "TamperingResult",
    "compute_hash",
    "detect_tampering",
    "verify_chain",
    "verify_trial",
]
