"""
STRATIX Privacy Model

Defines the privacy envelope required by all payload-bearing events
as specified in Step 1: Canonical Event & Trace Schema.

NORMATIVE:
- Every payload-bearing event MUST include a privacy envelope
- Payload omission MUST still include payload_hash
- not_provided is explicit and attestable
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PrivacyLevel(str, Enum):
    """
    Privacy classification levels for event payloads.

    From Step 1 specification:
    - cleartext: Full payload stored and accessible
    - redacted: Sanitized payload with PII/sensitive data removed
    - hashed: Only hash of payload stored
    - external_reference: Payload stored externally, URI + hash stored
    - not_provided: Payload explicitly not captured (but hashed as marker)
    """

    CLEARTEXT = "cleartext"
    REDACTED = "redacted"
    HASHED = "hashed"
    EXTERNAL_REFERENCE = "external_reference"
    NOT_PROVIDED = "not_provided"


class RedactionMethod(str, Enum):
    """
    Methods for redacting sensitive data from payloads.

    Used when privacy level is REDACTED.
    """

    PII = "pii"  # Remove personally identifiable information
    CUSTOM = "custom"  # Custom redaction rules
    PATTERN = "pattern"  # Regex-based pattern redaction
    FIELD = "field"  # Specific field redaction


class PrivacyEnvelope(BaseModel):
    """
    Privacy envelope required by all payload-bearing events.

    NORMATIVE:
    - Payload omission MUST still include payload_hash
    - not_provided is explicit and attestable

    From Step 2 DSL translation:
    | Privacy Level      | Storage         | Graders       | Attestation            |
    |--------------------|-----------------|---------------|------------------------|
    | cleartext          | Store payload   | Full access   | Hash payload           |
    | redacted           | Store redacted  | Redacted view | Hash redacted payload  |
    | hashed             | Store hash only | Hash-only     | Hash-of-hash permitted |
    | external_reference | Store URI+hash  | URI + hash    | Hash over URI+hash     |
    | not_provided       | No payload      | None          | Hash "not_provided"    |
    """

    level: PrivacyLevel = Field(
        description="Privacy level for this payload"
    )
    payload_hash: str = Field(
        description="SHA-256 hash of the payload (always required)"
    )
    external_ref: str | None = Field(
        default=None,
        description="URI for externally stored payload (when level=external_reference)"
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable reason for privacy level selection"
    )
    redaction_method: RedactionMethod | None = Field(
        default=None,
        description="Method used for redaction (when level=redacted)"
    )
    redacted_fields: list[str] | None = Field(
        default=None,
        description="List of fields that were redacted"
    )

    @field_validator("payload_hash")
    @classmethod
    def validate_payload_hash(cls, v: str) -> str:
        """Validate that payload_hash has correct format."""
        if not v.startswith("sha256:"):
            raise ValueError("payload_hash must start with 'sha256:'")
        hex_part = v[7:]
        if len(hex_part) != 64:
            raise ValueError("payload_hash must be sha256: followed by 64 hex characters")
        try:
            int(hex_part, 16)
        except ValueError as e:
            raise ValueError(f"payload_hash hex portion is invalid: {e}") from e
        return v

    @field_validator("external_ref")
    @classmethod
    def validate_external_ref(cls, v: str | None, info) -> str | None:
        """Validate external_ref is provided when level is external_reference."""
        # Access level from the data being validated
        if v is not None and not v.startswith(("http://", "https://", "s3://", "gs://", "file://")):
            raise ValueError("external_ref must be a valid URI")
        return v

    @classmethod
    def compute_hash(cls, payload: Any) -> str:
        """
        Compute the SHA-256 hash of a payload.

        Args:
            payload: The payload to hash (will be JSON serialized)

        Returns:
            Hash string in format "sha256:<hex>"
        """
        if payload is None:
            # Hash the string "not_provided" for missing payloads
            data = b"not_provided"
        elif isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            # Serialize to canonical JSON (sorted keys, no whitespace)
            data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        hash_hex = hashlib.sha256(data).hexdigest()
        return f"sha256:{hash_hex}"

    @classmethod
    def for_cleartext(cls, payload: Any, reason: str | None = None) -> PrivacyEnvelope:
        """
        Create a privacy envelope for cleartext payload.

        Args:
            payload: The full payload
            reason: Optional reason for cleartext storage

        Returns:
            PrivacyEnvelope configured for cleartext
        """
        return cls(
            level=PrivacyLevel.CLEARTEXT,
            payload_hash=cls.compute_hash(payload),
            reason=reason,
        )

    @classmethod
    def for_redacted(
        cls,
        original_payload: Any,
        redacted_payload: Any,
        method: RedactionMethod,
        redacted_fields: list[str] | None = None,
        reason: str | None = None,
    ) -> PrivacyEnvelope:
        """
        Create a privacy envelope for redacted payload.

        Note: The hash is computed on the REDACTED payload, not the original.

        Args:
            original_payload: The original unredacted payload (not stored)
            redacted_payload: The redacted payload that will be stored
            method: The redaction method used
            redacted_fields: List of fields that were redacted
            reason: Optional reason for redaction

        Returns:
            PrivacyEnvelope configured for redacted storage
        """
        return cls(
            level=PrivacyLevel.REDACTED,
            payload_hash=cls.compute_hash(redacted_payload),
            reason=reason,
            redaction_method=method,
            redacted_fields=redacted_fields,
        )

    @classmethod
    def for_hashed(cls, payload: Any, reason: str | None = None) -> PrivacyEnvelope:
        """
        Create a privacy envelope for hash-only storage.

        Args:
            payload: The payload to hash (payload itself is not stored)
            reason: Optional reason for hash-only storage

        Returns:
            PrivacyEnvelope configured for hash-only storage
        """
        return cls(
            level=PrivacyLevel.HASHED,
            payload_hash=cls.compute_hash(payload),
            reason=reason,
        )

    @classmethod
    def for_external_reference(
        cls,
        payload: Any,
        external_uri: str,
        reason: str | None = None,
    ) -> PrivacyEnvelope:
        """
        Create a privacy envelope for externally stored payload.

        Args:
            payload: The payload (stored externally)
            external_uri: URI where the payload is stored
            reason: Optional reason for external storage

        Returns:
            PrivacyEnvelope configured for external reference
        """
        return cls(
            level=PrivacyLevel.EXTERNAL_REFERENCE,
            payload_hash=cls.compute_hash(payload),
            external_ref=external_uri,
            reason=reason,
        )

    @classmethod
    def for_not_provided(cls, reason: str | None = None) -> PrivacyEnvelope:
        """
        Create a privacy envelope for explicitly not provided payload.

        NORMATIVE: not_provided markers are hashed (never silent).

        Args:
            reason: Optional reason why payload is not provided

        Returns:
            PrivacyEnvelope configured for not_provided
        """
        return cls(
            level=PrivacyLevel.NOT_PROVIDED,
            payload_hash=cls.compute_hash(None),  # Hashes "not_provided" string
            reason=reason or "Payload explicitly not captured per policy",
        )

    def is_accessible_to_graders(self) -> bool:
        """Check if the payload is accessible to graders."""
        return self.level in (PrivacyLevel.CLEARTEXT, PrivacyLevel.REDACTED)

    def is_hashable(self) -> bool:
        """Check if this envelope contains valid hash data."""
        return self.payload_hash is not None and len(self.payload_hash) > 0

    def get_attestation_data(self) -> str:
        """
        Get the data to be included in attestation hash.

        For different privacy levels:
        - cleartext: hash of payload
        - redacted: hash of redacted payload
        - hashed: the hash itself (hash-of-hash permitted)
        - external_reference: hash over URI+hash
        - not_provided: hash of "not_provided" marker
        """
        if self.level == PrivacyLevel.EXTERNAL_REFERENCE:
            # Hash over the combination of URI and payload hash
            combined = f"{self.external_ref}|{self.payload_hash}"
            return PrivacyEnvelope.compute_hash(combined)
        return self.payload_hash
