"""Tests for STRATIX Python SDK Core."""

import pytest

from layerlens.instrument import STRATIX, STRATIXContext, get_current_context
from layerlens.instrument.schema.privacy import PrivacyLevel


class TestSTRATIXInitialization:
    """Tests for STRATIX SDK initialization."""

    def test_one_liner_initialization(self):
        """Test that SDK can be initialized with a one-liner."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
        )

        assert stratix.policy_ref == "test-policy@1.0.0"
        assert stratix.policy_id == "test-policy"
        assert stratix.policy_version == "1.0.0"
        assert stratix.agent_id == "test_agent"

    def test_initialization_with_all_options(self):
        """Test initialization with all configuration options."""
        stratix = STRATIX(
            policy_ref="stratix-policy-cs-v1@1.0.0",
            agent_id="support_agent",
            framework="langgraph",
            exporter="otel",
            endpoint="otel-collector:4317",
            signing_key_id="key-123",
            privacy_default=PrivacyLevel.HASHED,
        )

        assert stratix.framework == "langgraph"
        assert stratix.policy_id == "stratix-policy-cs-v1"
        assert stratix.policy_version == "1.0.0"

    def test_policy_ref_without_version(self):
        """Test parsing policy reference without explicit version."""
        stratix = STRATIX(
            policy_ref="my-policy",
            agent_id="test_agent",
        )

        assert stratix.policy_id == "my-policy"
        assert stratix.policy_version == "latest"


class TestTrialLifecycle:
    """Tests for trial lifecycle management."""

    def test_start_trial(self):
        """Test starting a new trial."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        assert ctx is not None
        assert isinstance(ctx, STRATIXContext)
        assert ctx.evaluation_id is not None
        assert ctx.trial_id is not None
        assert ctx.trace_id is not None

    def test_start_trial_with_ids(self):
        """Test starting a trial with explicit IDs."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial(
            evaluation_id="11111111-1111-1111-1111-111111111111",
            trial_id="22222222-2222-2222-2222-222222222222",
            trace_id="33333333-3333-3333-3333-333333333333",
        )

        assert ctx.evaluation_id == "11111111-1111-1111-1111-111111111111"
        assert ctx.trial_id == "22222222-2222-2222-2222-222222222222"
        assert ctx.trace_id == "33333333-3333-3333-3333-333333333333"

    def test_end_trial_attestable(self):
        """Test ending a trial that is attestable."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # End trial
        result = stratix.end_trial()

        assert result is not None
        assert result["status"] == "attestable"
        assert "trial_hash" in result
        assert result["chain_verified"] is True

    def test_context_propagation(self):
        """Test that context is propagated correctly."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Context should be set as current
        current = get_current_context()
        assert current is ctx


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_input(self):
        """Test emitting an input event."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Emit input
        event = stratix.emit_input("Hello, agent!")

        assert event is not None
        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "agent.input"

    def test_emit_output(self):
        """Test emitting an output event."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Emit output
        event = stratix.emit_output("Hello, human!")

        assert event is not None
        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "agent.output"

    def test_sequence_id_monotonicity(self):
        """Test that sequence IDs are monotonically increasing."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Emit multiple events
        stratix.emit_input("Message 1")
        stratix.emit_input("Message 2")
        stratix.emit_input("Message 3")

        events = stratix.get_events()
        seq_ids = [e.identity.sequence_id for e in events]

        assert seq_ids == sorted(seq_ids)
        assert len(set(seq_ids)) == len(seq_ids)  # All unique

    def test_hash_chain_continuity(self):
        """Test that hash chain maintains continuity."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Emit multiple events
        stratix.emit_input("Message 1")
        stratix.emit_input("Message 2")
        stratix.emit_input("Message 3")

        events = stratix.get_events()

        # First event has no previous hash
        assert events[0].attestation.previous_hash is None

        # Subsequent events have a previous hash (chain linkage)
        for i in range(1, len(events)):
            assert events[i].attestation.previous_hash is not None
            assert events[i].attestation.previous_hash.startswith("sha256:")

        # Each event has a unique hash
        hashes = [e.attestation.hash for e in events]
        assert len(set(hashes)) == len(hashes)


class TestPolicyViolation:
    """Tests for policy violation handling."""

    def test_policy_violation_terminates_hashing(self):
        """Test that policy violation terminates the hash chain."""
        from layerlens.instrument.schema.events import ViolationType

        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Emit some events
        stratix.emit_input("Message 1")

        # Trigger violation
        stratix.emit_policy_violation(
            violation_type=ViolationType.PRIVACY,
            root_cause="Sensitive data leaked",
            remediation="Redact PII before output",
        )

        assert stratix.is_policy_violated is True

        # Further events should not be recorded
        event = stratix.emit_input("Message after violation")
        assert event is None

    def test_end_trial_after_violation(self):
        """Test ending a trial after policy violation."""
        from layerlens.instrument.schema.events import ViolationType

        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        # Trigger violation
        stratix.emit_policy_violation(
            violation_type=ViolationType.SAFETY,
            root_cause="Unsafe action attempted",
            remediation="Block action and notify user",
        )

        result = stratix.end_trial()

        assert result["status"] == "non-attestable"
        assert result["reason"] == "policy_violation"
