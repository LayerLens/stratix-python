"""Tests for STRATIX privacy types."""

import pytest

from layerlens.instrument.schema.privacy import (
    PrivacyLevel,
    RedactionMethod,
    PrivacyEnvelope,
)


class TestPrivacyLevel:
    """Tests for PrivacyLevel enum."""

    def test_all_levels_defined(self):
        """Test that all privacy levels from spec are defined."""
        assert PrivacyLevel.CLEARTEXT.value == "cleartext"
        assert PrivacyLevel.REDACTED.value == "redacted"
        assert PrivacyLevel.HASHED.value == "hashed"
        assert PrivacyLevel.EXTERNAL_REFERENCE.value == "external_reference"
        assert PrivacyLevel.NOT_PROVIDED.value == "not_provided"


class TestPrivacyEnvelope:
    """Tests for PrivacyEnvelope."""

    def test_compute_hash_deterministic(self):
        """Test that compute_hash is deterministic."""
        payload = {"key": "value", "number": 42}
        hash1 = PrivacyEnvelope.compute_hash(payload)
        hash2 = PrivacyEnvelope.compute_hash(payload)
        assert hash1 == hash2

    def test_compute_hash_format(self):
        """Test that hash has correct format."""
        hash_val = PrivacyEnvelope.compute_hash("test")
        assert hash_val.startswith("sha256:")
        assert len(hash_val) == 71  # "sha256:" + 64 hex chars

    def test_for_cleartext(self):
        """Test creating cleartext privacy envelope."""
        envelope = PrivacyEnvelope.for_cleartext({"data": "test"})
        assert envelope.level == PrivacyLevel.CLEARTEXT
        assert envelope.payload_hash.startswith("sha256:")

    def test_for_hashed(self):
        """Test creating hash-only privacy envelope."""
        envelope = PrivacyEnvelope.for_hashed({"data": "test"})
        assert envelope.level == PrivacyLevel.HASHED
        assert envelope.payload_hash.startswith("sha256:")

    def test_for_not_provided(self):
        """Test creating not_provided privacy envelope.

        NORMATIVE: not_provided markers are hashed (never silent).
        """
        envelope = PrivacyEnvelope.for_not_provided()
        assert envelope.level == PrivacyLevel.NOT_PROVIDED
        assert envelope.payload_hash.startswith("sha256:")
        # Should hash "not_provided" string
        expected_hash = PrivacyEnvelope.compute_hash(None)
        assert envelope.payload_hash == expected_hash

    def test_for_redacted(self):
        """Test creating redacted privacy envelope."""
        original = {"name": "John Doe", "email": "john@example.com"}
        redacted = {"name": "[REDACTED]", "email": "[REDACTED]"}
        envelope = PrivacyEnvelope.for_redacted(
            original_payload=original,
            redacted_payload=redacted,
            method=RedactionMethod.PII,
            redacted_fields=["name", "email"],
        )
        assert envelope.level == PrivacyLevel.REDACTED
        assert envelope.redaction_method == RedactionMethod.PII
        assert envelope.redacted_fields == ["name", "email"]
        # Hash should be of redacted payload, not original
        assert envelope.payload_hash == PrivacyEnvelope.compute_hash(redacted)

    def test_for_external_reference(self):
        """Test creating external reference privacy envelope."""
        payload = {"large": "data"}
        envelope = PrivacyEnvelope.for_external_reference(
            payload=payload,
            external_uri="s3://bucket/key",
        )
        assert envelope.level == PrivacyLevel.EXTERNAL_REFERENCE
        assert envelope.external_ref == "s3://bucket/key"

    def test_external_ref_validation(self):
        """Test that external_ref must be a valid URI."""
        with pytest.raises(ValueError):
            PrivacyEnvelope(
                level=PrivacyLevel.EXTERNAL_REFERENCE,
                payload_hash="sha256:" + "a" * 64,
                external_ref="not-a-uri",
            )

    def test_is_accessible_to_graders(self):
        """Test grader accessibility check."""
        cleartext = PrivacyEnvelope.for_cleartext({"data": "test"})
        assert cleartext.is_accessible_to_graders() is True

        redacted = PrivacyEnvelope.for_redacted(
            {"data": "original"},
            {"data": "redacted"},
            RedactionMethod.PII,
        )
        assert redacted.is_accessible_to_graders() is True

        hashed = PrivacyEnvelope.for_hashed({"data": "test"})
        assert hashed.is_accessible_to_graders() is False

        not_provided = PrivacyEnvelope.for_not_provided()
        assert not_provided.is_accessible_to_graders() is False

    def test_payload_hash_validation(self):
        """Test that payload_hash must have correct format."""
        with pytest.raises(ValueError):
            PrivacyEnvelope(
                level=PrivacyLevel.CLEARTEXT,
                payload_hash="invalid-hash",
            )

        with pytest.raises(ValueError):
            PrivacyEnvelope(
                level=PrivacyLevel.CLEARTEXT,
                payload_hash="sha256:tooshort",
            )
