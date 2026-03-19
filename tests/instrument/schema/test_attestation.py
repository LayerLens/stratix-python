"""Tests for STRATIX attestation types."""

import pytest

from layerlens.instrument.schema.attestation import (
    HashScope,
    AttestationEnvelope,
    HashChainBuilder,
)


class TestHashScope:
    """Tests for HashScope enum."""

    def test_all_scopes_defined(self):
        """Test that all hash scopes from spec are defined."""
        assert HashScope.EVENT.value == "event"
        assert HashScope.INTEGRATION.value == "integration"
        assert HashScope.TASK.value == "task"
        assert HashScope.TRIAL.value == "trial"


class TestAttestationEnvelope:
    """Tests for AttestationEnvelope."""

    def test_compute_hash_deterministic(self):
        """Test that compute_hash is deterministic."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        hash1 = AttestationEnvelope.compute_hash(data)
        hash2 = AttestationEnvelope.compute_hash(data)
        assert hash1 == hash2

    def test_compute_hash_canonical_json(self):
        """Test that compute_hash produces same result regardless of key order."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        assert AttestationEnvelope.compute_hash(data1) == AttestationEnvelope.compute_hash(data2)

    def test_compute_hash_format(self):
        """Test that hash has correct format."""
        hash_val = AttestationEnvelope.compute_hash("test")
        assert hash_val.startswith("sha256:")
        assert len(hash_val) == 71  # "sha256:" + 64 hex chars

    def test_create_event_hash(self):
        """Test creating an event hash."""
        event_data = {"event_type": "agent.input", "content": "test"}
        envelope = AttestationEnvelope.create_event_hash(event_data)
        assert envelope.hash_scope == HashScope.EVENT
        assert envelope.hash.startswith("sha256:")
        assert envelope.previous_hash is None

    def test_create_event_hash_with_previous(self):
        """Test creating an event hash with previous hash."""
        event_data = {"event_type": "agent.input", "content": "test"}
        prev_hash = "sha256:" + "a" * 64
        envelope = AttestationEnvelope.create_event_hash(
            event_data=event_data,
            previous_hash=prev_hash,
        )
        assert envelope.previous_hash == prev_hash

    def test_create_integration_hash(self):
        """Test creating an integration hash."""
        integration_data = {"tool": "lookup", "input": {}, "output": {}}
        envelope = AttestationEnvelope.create_integration_hash(integration_data)
        assert envelope.hash_scope == HashScope.INTEGRATION

    def test_create_trial_hash(self):
        """Test creating a trial hash from event hashes."""
        event_hashes = [
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
        ]
        envelope = AttestationEnvelope.create_trial_hash(event_hashes)
        assert envelope.hash_scope == HashScope.TRIAL
        assert envelope.previous_hash == event_hashes[-1]

    def test_verify_chain(self):
        """Test chain verification."""
        prev_hash = "sha256:" + "a" * 64
        envelope = AttestationEnvelope(
            hash="sha256:" + "b" * 64,
            hash_scope=HashScope.EVENT,
            previous_hash=prev_hash,
        )
        assert envelope.verify_chain(prev_hash) is True
        assert envelope.verify_chain("sha256:" + "c" * 64) is False

    def test_is_signed(self):
        """Test signature detection."""
        unsigned = AttestationEnvelope(
            hash="sha256:" + "a" * 64,
            hash_scope=HashScope.EVENT,
        )
        assert unsigned.is_signed() is False

        signed = AttestationEnvelope(
            hash="sha256:" + "a" * 64,
            hash_scope=HashScope.EVENT,
            signing_key_id="key-1",
            signature="dGVzdA==",  # base64 "test"
        )
        assert signed.is_signed() is True

    def test_hash_validation(self):
        """Test that hash must have correct format."""
        with pytest.raises(ValueError):
            AttestationEnvelope(
                hash="invalid",
                hash_scope=HashScope.EVENT,
            )


class TestHashChainBuilder:
    """Tests for HashChainBuilder."""

    def test_empty_chain(self):
        """Test empty chain state."""
        builder = HashChainBuilder()
        assert builder.chain_length == 0
        assert builder.last_hash is None
        assert builder.is_terminated is False

    def test_add_event_builds_chain(self):
        """Test adding events builds a continuous chain."""
        builder = HashChainBuilder()
        event1 = builder.add_event({"event": "first"})
        assert builder.chain_length == 1
        assert builder.last_hash == event1.hash
        assert event1.previous_hash is None

        event2 = builder.add_event({"event": "second"})
        assert builder.chain_length == 2
        assert builder.last_hash == event2.hash
        assert event2.previous_hash == event1.hash

    def test_chain_integrity(self):
        """Test chain integrity verification."""
        builder = HashChainBuilder()
        builder.add_event({"event": "first"})
        builder.add_event({"event": "second"})
        builder.add_event({"event": "third"})
        assert builder.verify_chain_integrity() is True

    def test_terminate_stops_hashing(self):
        """Test that termination stops further hashing.

        NORMATIVE: On policy violation, hashing MUST stop immediately.
        """
        builder = HashChainBuilder()
        builder.add_event({"event": "first"})
        builder.terminate("policy_violation")
        assert builder.is_terminated is True

        with pytest.raises(RuntimeError, match="terminated"):
            builder.add_event({"event": "second"})

    def test_finalize_trial(self):
        """Test trial finalization."""
        builder = HashChainBuilder()
        builder.add_event({"event": "first"})
        builder.add_event({"event": "second"})
        trial_envelope = builder.finalize_trial()
        assert trial_envelope.hash_scope == HashScope.TRIAL

    def test_finalize_terminated_fails(self):
        """Test that finalization fails for terminated chain."""
        builder = HashChainBuilder()
        builder.add_event({"event": "first"})
        builder.terminate("policy_violation")

        with pytest.raises(RuntimeError, match="non-attestable"):
            builder.finalize_trial()
