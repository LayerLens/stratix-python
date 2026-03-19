"""
STRATIX Attestation Model

Defines the attestation envelope required for hash chains and cryptographic
signing as specified in Step 1: Canonical Event & Trace Schema.

NORMATIVE:
- Every event and integration boundary MUST include hash metadata
- Hash chains MUST be continuous within a trial
- On policy violation, hashing MUST stop immediately
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HashScope(str, Enum):
    """
    Scope levels for hashing operations.

    From Step 3 architecture:
    | Boundary    | Hash Scope    | Notes                           |
    |-------------|---------------|--------------------------------|
    | Event       | event         | Every emitted event            |
    | Integration | integration   | Tool call boundary; per action |
    | Task        | task          | Optional grouping of steps     |
    | Trial       | trial         | Composite for entire execution |
    """

    EVENT = "event"
    INTEGRATION = "integration"
    TASK = "task"
    TRIAL = "trial"


class AttestationEnvelope(BaseModel):
    """
    Attestation envelope for hash chains and cryptographic signing.

    NORMATIVE:
    - Hash chains MUST be continuous within a trial
    - On policy violation, hashing MUST stop immediately
    - No further hashes are generated after violation

    From Step 1 specification:
    {
        "attestation": {
            "hash": "sha256:...",
            "hash_scope": "event | integration | task | trial",
            "signing_key_id": "string",
            "signature": "base64",
            "previous_hash": "sha256 | null"
        }
    }
    """

    hash: str = Field(
        description="SHA-256 hash of the event content"
    )
    hash_scope: HashScope = Field(
        description="Scope of this hash (event, integration, task, trial)"
    )
    signing_key_id: str | None = Field(
        default=None,
        description="Identifier for the signing key (platform or BYOK)"
    )
    signature: str | None = Field(
        default=None,
        description="Base64 encoded signature over the hash"
    )
    previous_hash: str | None = Field(
        default=None,
        description="Hash of the previous event in the chain (null for first)"
    )

    @field_validator("hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate that hash has correct format."""
        if not v.startswith("sha256:"):
            raise ValueError("hash must start with 'sha256:'")
        hex_part = v[7:]
        if len(hex_part) != 64:
            raise ValueError("hash must be sha256: followed by 64 hex characters")
        try:
            int(hex_part, 16)
        except ValueError as e:
            raise ValueError(f"hash hex portion is invalid: {e}") from e
        return v

    @field_validator("previous_hash")
    @classmethod
    def validate_previous_hash(cls, v: str | None) -> str | None:
        """Validate that previous_hash has correct format if provided."""
        if v is not None:
            if not v.startswith("sha256:"):
                raise ValueError("previous_hash must start with 'sha256:'")
            hex_part = v[7:]
            if len(hex_part) != 64:
                raise ValueError("previous_hash must be sha256: followed by 64 hex characters")
            try:
                int(hex_part, 16)
            except ValueError as e:
                raise ValueError(f"previous_hash hex portion is invalid: {e}") from e
        return v

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON encoder for non-standard types."""
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    @classmethod
    def compute_hash(cls, data: Any) -> str:
        """
        Compute the SHA-256 hash of data.

        Args:
            data: The data to hash (will be JSON serialized canonically)

        Returns:
            Hash string in format "sha256:<hex>"
        """
        if isinstance(data, bytes):
            serialized = data
        elif isinstance(data, str):
            serialized = data.encode("utf-8")
        else:
            # Canonical JSON serialization (sorted keys, no extra whitespace)
            serialized = json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=cls._json_default,
            ).encode("utf-8")

        hash_hex = hashlib.sha256(serialized).hexdigest()
        return f"sha256:{hash_hex}"

    @classmethod
    def create_event_hash(
        cls,
        event_data: dict[str, Any],
        previous_hash: str | None = None,
        signing_key_id: str | None = None,
    ) -> AttestationEnvelope:
        """
        Create an attestation envelope for an event.

        Args:
            event_data: The event data to hash
            previous_hash: Hash of the previous event in the chain
            signing_key_id: Optional signing key identifier

        Returns:
            AttestationEnvelope for the event
        """
        return cls(
            hash=cls.compute_hash(event_data),
            hash_scope=HashScope.EVENT,
            signing_key_id=signing_key_id,
            previous_hash=previous_hash,
        )

    @classmethod
    def create_integration_hash(
        cls,
        integration_data: dict[str, Any],
        previous_hash: str | None = None,
        signing_key_id: str | None = None,
    ) -> AttestationEnvelope:
        """
        Create an attestation envelope for an integration boundary.

        Args:
            integration_data: The integration data to hash
            previous_hash: Hash of the previous item in the chain
            signing_key_id: Optional signing key identifier

        Returns:
            AttestationEnvelope for the integration
        """
        return cls(
            hash=cls.compute_hash(integration_data),
            hash_scope=HashScope.INTEGRATION,
            signing_key_id=signing_key_id,
            previous_hash=previous_hash,
        )

    @classmethod
    def create_task_hash(
        cls,
        task_data: dict[str, Any],
        previous_hash: str | None = None,
        signing_key_id: str | None = None,
    ) -> AttestationEnvelope:
        """
        Create an attestation envelope for a task grouping.

        Args:
            task_data: The task data to hash
            previous_hash: Hash of the previous item in the chain
            signing_key_id: Optional signing key identifier

        Returns:
            AttestationEnvelope for the task
        """
        return cls(
            hash=cls.compute_hash(task_data),
            hash_scope=HashScope.TASK,
            signing_key_id=signing_key_id,
            previous_hash=previous_hash,
        )

    @classmethod
    def create_trial_hash(
        cls,
        event_hashes: list[str],
        signing_key_id: str | None = None,
    ) -> AttestationEnvelope:
        """
        Create an attestation envelope for an entire trial.

        The trial hash is computed over all event hashes in order.

        Args:
            event_hashes: List of all event hashes in the trial
            signing_key_id: Optional signing key identifier

        Returns:
            AttestationEnvelope for the trial
        """
        # Compute composite hash over all event hashes
        composite = {"event_hashes": event_hashes}
        return cls(
            hash=cls.compute_hash(composite),
            hash_scope=HashScope.TRIAL,
            signing_key_id=signing_key_id,
            previous_hash=event_hashes[-1] if event_hashes else None,
        )

    def verify_chain(self, expected_previous: str | None) -> bool:
        """
        Verify that this attestation links to the expected previous hash.

        Args:
            expected_previous: The expected previous hash

        Returns:
            True if the chain is valid
        """
        return self.previous_hash == expected_previous

    def is_signed(self) -> bool:
        """Check if this attestation includes a signature."""
        return self.signature is not None and self.signing_key_id is not None


class HashChainBuilder:
    """
    Builder for maintaining hash chain continuity within a trial.

    NORMATIVE: Hash chains MUST be continuous within a trial.
    """

    def __init__(self, signing_key_id: str | None = None):
        self._chain: list[AttestationEnvelope] = []
        self._last_hash: str | None = None
        self._signing_key_id = signing_key_id
        self._terminated = False

    @property
    def is_terminated(self) -> bool:
        """Check if the hash chain has been terminated (e.g., due to violation)."""
        return self._terminated

    @property
    def last_hash(self) -> str | None:
        """Get the hash of the last item in the chain."""
        return self._last_hash

    @property
    def chain_length(self) -> int:
        """Get the number of items in the chain."""
        return len(self._chain)

    def add_event(self, event_data: dict[str, Any]) -> AttestationEnvelope:
        """
        Add an event to the hash chain.

        Args:
            event_data: The event data to hash

        Returns:
            AttestationEnvelope for the event

        Raises:
            RuntimeError: If the chain has been terminated
        """
        if self._terminated:
            raise RuntimeError(
                "Cannot add to terminated hash chain. "
                "Chain was terminated due to policy violation."
            )

        envelope = AttestationEnvelope.create_event_hash(
            event_data=event_data,
            previous_hash=self._last_hash,
            signing_key_id=self._signing_key_id,
        )
        self._chain.append(envelope)
        self._last_hash = envelope.hash
        return envelope

    def add_integration(self, integration_data: dict[str, Any]) -> AttestationEnvelope:
        """
        Add an integration boundary to the hash chain.

        Args:
            integration_data: The integration data to hash

        Returns:
            AttestationEnvelope for the integration

        Raises:
            RuntimeError: If the chain has been terminated
        """
        if self._terminated:
            raise RuntimeError(
                "Cannot add to terminated hash chain. "
                "Chain was terminated due to policy violation."
            )

        envelope = AttestationEnvelope.create_integration_hash(
            integration_data=integration_data,
            previous_hash=self._last_hash,
            signing_key_id=self._signing_key_id,
        )
        self._chain.append(envelope)
        self._last_hash = envelope.hash
        return envelope

    def terminate(self, reason: str = "policy_violation") -> None:
        """
        Terminate the hash chain.

        NORMATIVE: On policy violation, hashing MUST stop immediately.

        Args:
            reason: Reason for termination
        """
        self._terminated = True
        self._termination_reason = reason

    def finalize_trial(self) -> AttestationEnvelope:
        """
        Create a trial-level hash over all events.

        Returns:
            AttestationEnvelope for the trial

        Raises:
            RuntimeError: If the chain has been terminated
        """
        if self._terminated:
            raise RuntimeError(
                "Cannot finalize terminated hash chain. "
                "Trial is non-attestable due to policy violation."
            )

        event_hashes = [e.hash for e in self._chain]
        return AttestationEnvelope.create_trial_hash(
            event_hashes=event_hashes,
            signing_key_id=self._signing_key_id,
        )

    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire hash chain.

        Returns:
            True if all chain links are valid
        """
        if len(self._chain) == 0:
            return True

        # First item should have no previous hash
        if self._chain[0].previous_hash is not None:
            return False

        # Each subsequent item should link to the previous
        for i in range(1, len(self._chain)):
            if self._chain[i].previous_hash != self._chain[i - 1].hash:
                return False

        return True

    def get_chain(self) -> list[AttestationEnvelope]:
        """Get a copy of the hash chain."""
        return list(self._chain)
