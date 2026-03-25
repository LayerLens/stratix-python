from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass

from ._hash import compute_hash
from ._envelope import HashScope, AttestationEnvelope


@dataclass
class ChainVerification:
    """Result of verifying a hash chain's integrity."""

    valid: bool
    break_index: Optional[int] = None
    error: Optional[str] = None


@dataclass
class TamperingResult:
    """Result of checking whether trace data was modified after hashing."""

    tampered: bool
    modified_indices: List[int] = field(default_factory=list)
    chain_broken: bool = False


def verify_chain(envelopes: List[AttestationEnvelope]) -> ChainVerification:
    """Verify that a hash chain is continuous and unbroken.

    Checks:
    - First envelope has previous_hash=None
    - Each subsequent envelope's previous_hash matches the prior envelope's hash
    """
    if not envelopes:
        return ChainVerification(valid=True)

    if envelopes[0].previous_hash is not None:
        return ChainVerification(
            valid=False,
            break_index=0,
            error="First envelope must have previous_hash=None",
        )

    for i in range(1, len(envelopes)):
        if envelopes[i].previous_hash != envelopes[i - 1].hash:
            return ChainVerification(
                valid=False,
                break_index=i,
                error=f"Chain broken at index {i}: "
                f"expected previous_hash={envelopes[i - 1].hash!r}, "
                f"got {envelopes[i].previous_hash!r}",
            )

    return ChainVerification(valid=True)


def verify_trial(
    envelopes: List[AttestationEnvelope],
    trial_envelope: AttestationEnvelope,
) -> ChainVerification:
    """Verify a trial envelope against its event chain.

    Checks chain integrity, then verifies the trial hash is correctly
    computed over all event hashes.
    """
    chain_result = verify_chain(envelopes)
    if not chain_result.valid:
        return chain_result

    if trial_envelope.scope != HashScope.TRIAL:
        return ChainVerification(
            valid=False,
            error=f"Trial envelope has wrong scope: {trial_envelope.scope}",
        )

    event_hashes = [e.hash for e in envelopes]
    expected_hash = compute_hash({"event_hashes": event_hashes})
    if trial_envelope.hash != expected_hash:
        return ChainVerification(
            valid=False,
            error="Trial hash does not match event hashes",
        )

    return ChainVerification(valid=True)


def detect_tampering(
    envelopes: List[AttestationEnvelope],
    original_data: List[Dict[str, Any]],
) -> TamperingResult:
    """Detect which events were modified after being hashed.

    Recomputes the hash for each event (using its stored previous_hash
    for chain linkage) and compares against the stored hash.
    """
    if len(envelopes) != len(original_data):
        return TamperingResult(
            tampered=True,
            chain_broken=True,
        )

    modified: List[int] = []
    for i, (envelope, data) in enumerate(zip(envelopes, original_data)):
        payload = {**data, "_previous_hash": envelope.previous_hash}
        recomputed = compute_hash(payload)
        if recomputed != envelope.hash:
            modified.append(i)

    chain_result = verify_chain(envelopes)
    return TamperingResult(
        tampered=len(modified) > 0 or not chain_result.valid,
        modified_indices=modified,
        chain_broken=not chain_result.valid,
    )
