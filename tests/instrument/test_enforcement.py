"""Tests for STRATIX Python SDK Policy Enforcement."""

import pytest

from layerlens.instrument import (
    STRATIX,
    PolicyEnforcer,
    PolicyViolationError,
    check_tool_allowed,
    check_model_allowed,
    check_max_tokens,
)
from layerlens.instrument._enforcement import enforce_or_fail
from layerlens.instrument.schema.events import ViolationType


class TestPolicyEnforcer:
    """Tests for the PolicyEnforcer class."""

    def test_create_enforcer(self):
        """Test creating a policy enforcer."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        assert enforcer._stratix is stratix
        assert enforcer.has_violations is False

    def test_register_pre_check(self):
        """Test registering a pre-check."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        def my_check(action_type: str, params: dict) -> str | None:
            if params.get("blocked"):
                return "This action is blocked"
            return None

        enforcer.register_pre_check(my_check)

        # Check passes
        result = enforcer.run_pre_checks("test", {"blocked": False})
        assert result is None

        # Check fails
        result = enforcer.run_pre_checks("test", {"blocked": True})
        assert result == "This action is blocked"


class TestPreCheckFunctions:
    """Tests for built-in pre-check functions."""

    def test_check_tool_allowed_blocked(self):
        """Test blocking specific tools."""
        check = check_tool_allowed(blocked_tools=["dangerous_tool"])

        # Allowed tool
        result = check("tool_call", {"name": "safe_tool"})
        assert result is None

        # Blocked tool
        result = check("tool_call", {"name": "dangerous_tool"})
        assert result is not None
        assert "blocked" in result

    def test_check_tool_allowed_allowlist(self):
        """Test allowlist-only tools."""
        check = check_tool_allowed(allowed_tools=["safe_tool", "another_safe"])

        # Allowed tool
        result = check("tool_call", {"name": "safe_tool"})
        assert result is None

        # Not in allowlist
        result = check("tool_call", {"name": "unknown_tool"})
        assert result is not None
        assert "allowed list" in result

    def test_check_tool_allowed_ignores_other_actions(self):
        """Test that tool check ignores non-tool actions."""
        check = check_tool_allowed(blocked_tools=["dangerous_tool"])

        # Non-tool action
        result = check("model_invoke", {"name": "dangerous_tool"})
        assert result is None

    def test_check_model_allowed_blocked(self):
        """Test blocking specific models."""
        check = check_model_allowed(blocked_models=["gpt-3"])

        # Allowed model
        result = check("model_invoke", {"name": "gpt-4"})
        assert result is None

        # Blocked model
        result = check("model_invoke", {"name": "gpt-3"})
        assert result is not None
        assert "blocked" in result

    def test_check_model_allowed_allowlist(self):
        """Test allowlist-only models."""
        check = check_model_allowed(allowed_models=["gpt-4", "claude-3"])

        # Allowed model
        result = check("model_invoke", {"name": "gpt-4"})
        assert result is None

        # Not in allowlist
        result = check("model_invoke", {"name": "unknown-model"})
        assert result is not None
        assert "allowed list" in result

    def test_check_max_tokens(self):
        """Test max tokens enforcement."""
        check = check_max_tokens(max_tokens=1000)

        # Within limit
        result = check("model_invoke", {"parameters": {"max_tokens": 500}})
        assert result is None

        # At limit
        result = check("model_invoke", {"parameters": {"max_tokens": 1000}})
        assert result is None

        # Over limit
        result = check("model_invoke", {"parameters": {"max_tokens": 1500}})
        assert result is not None
        assert "exceeds limit" in result

    def test_check_max_tokens_no_parameter(self):
        """Test max tokens when parameter not set."""
        check = check_max_tokens(max_tokens=1000)

        # No max_tokens parameter
        result = check("model_invoke", {"parameters": {}})
        assert result is None

        # No parameters at all
        result = check("model_invoke", {})
        assert result is None


class TestRequiredLayersCheck:
    """Tests for required layers checking."""

    def test_check_required_layers_pass(self):
        """Test when all required layers are present."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        result = enforcer.check_required_layers(
            required=["L1", "L3"],
            present=["L1", "L2", "L3", "L5"],
        )

        assert result is None

    def test_check_required_layers_fail(self):
        """Test when required layers are missing."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        result = enforcer.check_required_layers(
            required=["L1", "L3", "L5"],
            present=["L1", "L2"],
        )

        assert result is not None
        assert "L3" in result
        assert "L5" in result


class TestRequiredEventTypesCheck:
    """Tests for required event types checking."""

    def test_check_required_event_types_pass(self):
        """Test when all required event types are present."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        result = enforcer.check_required_event_types(
            required=["agent.input", "model.invoke"],
            emitted=["agent.input", "model.invoke", "tool.call"],
        )

        assert result is None

    def test_check_required_event_types_fail(self):
        """Test when required event types are missing."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        result = enforcer.check_required_event_types(
            required=["agent.input", "model.invoke", "agent.output"],
            emitted=["agent.input"],
        )

        assert result is not None
        assert "model.invoke" in result
        assert "agent.output" in result


class TestEnforceOrFail:
    """Tests for fail-fast enforcement."""

    def test_enforce_or_fail_pass(self):
        """Test that enforce_or_fail passes when checks pass."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)
        enforcer.register_pre_check(check_tool_allowed(blocked_tools=["bad_tool"]))

        ctx = stratix.start_trial()

        # Should not raise
        enforce_or_fail(enforcer, "tool_call", {"name": "good_tool"})

    def test_enforce_or_fail_raises(self):
        """Test that enforce_or_fail raises on violation."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)
        enforcer.register_pre_check(check_tool_allowed(blocked_tools=["bad_tool"]))

        ctx = stratix.start_trial()

        with pytest.raises(PolicyViolationError):
            enforce_or_fail(enforcer, "tool_call", {"name": "bad_tool"})


class TestViolationEmission:
    """Tests for violation emission."""

    def test_emit_violation(self):
        """Test emitting a policy violation."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        ctx = stratix.start_trial()

        enforcer.emit_violation(
            violation_type=ViolationType.PRIVACY,
            root_cause="PII detected in output",
            remediation="Enable PII redaction",
        )

        assert enforcer.has_violations is True
        assert stratix.is_policy_violated is True

        # Check violation event was emitted
        events = stratix.get_events()
        violation_events = [e for e in events if e.payload.event_type == "policy.violation"]
        assert len(violation_events) == 1

    def test_emit_violation_terminates_chain(self):
        """Test that violation emission terminates hash chain."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        enforcer = PolicyEnforcer(stratix)

        ctx = stratix.start_trial()

        # Emit some events first
        stratix.emit_input("Hello")

        enforcer.emit_violation(
            violation_type=ViolationType.SAFETY,
            root_cause="Unsafe action",
            remediation="Block action",
        )

        # Further events should not be emitted
        event = stratix.emit_input("After violation")
        assert event is None
