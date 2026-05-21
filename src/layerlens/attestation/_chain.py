from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Optional

from ._hash import compute_hash
from ._envelope import HashScope, AttestationEnvelope


class HashChain:
    """Builds a linear hash chain over a sequence of events.

    Each event is hashed and linked to the previous hash, forming
    a tamper-evident chain. If any event is modified after the fact,
    the chain breaks at that point.

    Signing is handled server-side at trace ingestion. The SDK builds
    the hash chain for integrity; the backend signs for authenticity.
    """

    def __init__(self) -> None:
        self._chain: List[AttestationEnvelope] = []
        self._last_hash: Optional[str] = None
        self._terminated: bool = False
        self._terminate_reason: Optional[str] = None

    @property
    def envelopes(self) -> List[AttestationEnvelope]:
        return [copy(e) for e in self._chain]

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    def _check_active(self) -> None:
        if self._terminated:
            raise RuntimeError(f"Hash chain terminated: {self._terminate_reason}. No further events can be added.")

    def add_event(self, data: Dict[str, Any]) -> AttestationEnvelope:
        """Hash an event and append it to the chain."""
        self._check_active()
        # Include previous_hash in the hashed payload for chaining
        payload = {**data, "_previous_hash": self._last_hash}
        event_hash = compute_hash(payload)
        envelope = AttestationEnvelope(
            hash=event_hash,
            scope=HashScope.EVENT,
            previous_hash=self._last_hash,
        )
        self._chain.append(envelope)
        self._last_hash = event_hash
        return envelope

    def terminate(self, reason: str) -> None:
        """Permanently stop the chain. No further events or finalization allowed."""
        self._terminated = True
        self._terminate_reason = reason

    def finalize(self) -> AttestationEnvelope:
        """Compute a trial-level root hash over all event hashes and seal the chain."""
        if self._terminated:
            raise RuntimeError(
                f"Cannot finalize terminated hash chain. Trial is non-attestable due to: {self._terminate_reason}"
            )
        if not self._chain:
            raise RuntimeError("Cannot finalize empty hash chain.")
        event_hashes = [e.hash for e in self._chain]
        root_hash = compute_hash({"event_hashes": event_hashes})
        trial_envelope = AttestationEnvelope(
            hash=root_hash,
            scope=HashScope.TRIAL,
            previous_hash=self._last_hash,
        )
        # Seal — no more events after finalization
        self._terminated = True
        self._terminate_reason = "chain finalized"
        return trial_envelope

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the chain for inclusion in trace uploads."""
        result: Dict[str, Any] = {
            "events": [e.to_dict() for e in self._chain],
        }
        # Only include termination details when the chain was stopped
        # due to a policy violation (not normal finalization).
        if self._terminated and self._terminate_reason != "chain finalized":
            result["terminated_reason"] = self._terminate_reason
        return result
