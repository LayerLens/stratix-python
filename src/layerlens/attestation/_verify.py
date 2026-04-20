from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass

from ._hash import compute_hash
from ._signing import hmac_verify
from ._envelope import HashScope, AttestationEnvelope


@dataclass
class ChainVerification:
    """Result of verifying a hash chain's integrity."""

    valid: bool
    break_index: Optional[int] = None
    error: Optional[str] = None


@dataclass
class TrialVerification:
    """Result of verifying a full trial: chain + root hash + signatures."""

    valid: bool
    chain_valid: bool = True
    trial_hash_valid: bool = True
    signatures_valid: bool = True
    errors: List[str] = field(default_factory=list)


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
    signing_secret: Optional[bytes] = None,
) -> TrialVerification:
    """Verify a trial envelope against its event chain.

    Checks chain integrity, trial hash correctness, and (optionally) signatures.
    Pass ``signing_secret`` to verify HMAC-SHA256 signatures.
    """
    errors: List[str] = []

    # 1. Chain continuity
    chain_result = verify_chain(envelopes)
    chain_valid = chain_result.valid
    if not chain_valid:
        errors.append(f"Chain integrity failed: {chain_result.error}")

    # 2. Trial scope + hash
    trial_hash_valid = True
    if trial_envelope.scope != HashScope.TRIAL:
        trial_hash_valid = False
        errors.append(f"Trial envelope has wrong scope: {trial_envelope.scope}")
    else:
        event_hashes = [e.hash for e in envelopes]
        expected_hash = compute_hash({"event_hashes": event_hashes})
        if trial_envelope.hash != expected_hash:
            trial_hash_valid = False
            errors.append("Trial hash does not match event hashes")

    # 3. Signatures (only if a signing secret is provided)
    signatures_valid = True
    if signing_secret is not None:
        for i, envelope in enumerate(envelopes):
            if not envelope.signature:
                signatures_valid = False
                errors.append(f"Missing signature on event {i}")
            else:
                if not hmac_verify(signing_secret, envelope.hash.encode("utf-8"), envelope.signature):
                    signatures_valid = False
                    errors.append(f"Invalid signature on event {i}")

        if not trial_envelope.signature:
            signatures_valid = False
            errors.append("Missing signature on trial envelope")
        else:
            if not hmac_verify(signing_secret, trial_envelope.hash.encode("utf-8"), trial_envelope.signature):
                signatures_valid = False
                errors.append("Invalid signature on trial envelope")

    valid = chain_valid and trial_hash_valid and signatures_valid
    return TrialVerification(
        valid=valid,
        chain_valid=chain_valid,
        trial_hash_valid=trial_hash_valid,
        signatures_valid=signatures_valid,
        errors=errors,
    )


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
