"""
STRATIX Local Policy Enforcement

From Step 4 specification:
- Local pre-checks for policy enforcement
- Emits policy.violation events
- Stops hashing on violation

This module provides local enforcement that runs within the SDK
before events are exported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from layerlens.instrument.schema.events import PolicyViolationEvent, ViolationType
from layerlens.instrument._context import get_current_context

if TYPE_CHECKING:
    from layerlens.instrument._core import STRATIX


class PolicyEnforcer:
    """
    Local policy enforcement within the SDK.

    Provides pre-checks before actions and post-checks after events.
    On violation, emits policy.violation and terminates hash chain.
    """

    def __init__(self, stratix: "STRATIX"):
        """
        Initialize the policy enforcer.

        Args:
            stratix: The STRATIX instance
        """
        self._stratix = stratix
        self._pre_checks: list[Callable[[str, dict[str, Any]], str | None]] = []
        self._post_checks: list[Callable[[Any], str | None]] = []
        self._violations: list[PolicyViolationEvent] = []

    def register_pre_check(
        self, check: Callable[[str, dict[str, Any]], str | None]
    ) -> None:
        """
        Register a pre-check function.

        Pre-checks run before an action is executed.

        Args:
            check: Function that takes (action_type, params) and returns
                   an error message if the check fails, None otherwise
        """
        self._pre_checks.append(check)

    def register_post_check(
        self, check: Callable[[Any], str | None]
    ) -> None:
        """
        Register a post-check function.

        Post-checks run after an event is created.

        Args:
            check: Function that takes an event payload and returns
                   an error message if the check fails, None otherwise
        """
        self._post_checks.append(check)

    def run_pre_checks(
        self, action_type: str, params: dict[str, Any]
    ) -> str | None:
        """
        Run all pre-checks for an action.

        Args:
            action_type: Type of action (e.g., "tool_call", "model_invoke")
            params: Action parameters

        Returns:
            Error message if any check fails, None otherwise
        """
        for check in self._pre_checks:
            try:
                result = check(action_type, params)
                if result is not None:
                    return result
            except Exception as e:
                return f"Pre-check error: {e}"
        return None

    def run_post_checks(self, event: Any) -> str | None:
        """
        Run all post-checks for an event.

        Args:
            event: The event payload

        Returns:
            Error message if any check fails, None otherwise
        """
        for check in self._post_checks:
            try:
                result = check(event)
                if result is not None:
                    return result
            except Exception as e:
                return f"Post-check error: {e}"
        return None

    def check_required_layers(self, required: list[str], present: list[str]) -> str | None:
        """
        Check that all required layers are present.

        Args:
            required: List of required layer names
            present: List of layers that are present

        Returns:
            Error message if check fails, None otherwise
        """
        missing = set(required) - set(present)
        if missing:
            return f"Missing required layers: {', '.join(sorted(missing))}"
        return None

    def check_required_event_types(
        self, required: list[str], emitted: list[str]
    ) -> str | None:
        """
        Check that all required event types have been emitted.

        Args:
            required: List of required event type names
            emitted: List of event types that have been emitted

        Returns:
            Error message if check fails, None otherwise
        """
        missing = set(required) - set(emitted)
        if missing:
            return f"Missing required event types: {', '.join(sorted(missing))}"
        return None

    def emit_violation(
        self,
        violation_type: ViolationType | str,
        root_cause: str,
        remediation: str,
        failed_layer: str | None = None,
    ) -> None:
        """
        Emit a policy violation and terminate the hash chain.

        Args:
            violation_type: Type of violation
            root_cause: Description of what caused the violation
            remediation: Suggested remediation
            failed_layer: Layer where the violation occurred
        """
        if isinstance(violation_type, str):
            violation_type = ViolationType(violation_type)

        self._stratix.emit_policy_violation(
            violation_type=violation_type,
            root_cause=root_cause,
            remediation=remediation,
            failed_layer=failed_layer,
        )

    @property
    def has_violations(self) -> bool:
        """Check if any violations have occurred."""
        return self._stratix.is_policy_violated


# Built-in pre-check functions


def check_tool_allowed(
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
) -> Callable[[str, dict[str, Any]], str | None]:
    """
    Create a pre-check that validates tool names.

    Args:
        allowed_tools: List of allowed tool names (if set, only these are allowed)
        blocked_tools: List of blocked tool names

    Returns:
        Pre-check function
    """
    def check(action_type: str, params: dict[str, Any]) -> str | None:
        if action_type != "tool_call":
            return None

        tool_name = params.get("name", "")

        if blocked_tools and tool_name in blocked_tools:
            return f"Tool '{tool_name}' is blocked by policy"

        if allowed_tools and tool_name not in allowed_tools:
            return f"Tool '{tool_name}' is not in the allowed list"

        return None

    return check


def check_model_allowed(
    allowed_models: list[str] | None = None,
    blocked_models: list[str] | None = None,
) -> Callable[[str, dict[str, Any]], str | None]:
    """
    Create a pre-check that validates model names.

    Args:
        allowed_models: List of allowed model names
        blocked_models: List of blocked model names

    Returns:
        Pre-check function
    """
    def check(action_type: str, params: dict[str, Any]) -> str | None:
        if action_type != "model_invoke":
            return None

        model_name = params.get("name", "")

        if blocked_models and model_name in blocked_models:
            return f"Model '{model_name}' is blocked by policy"

        if allowed_models and model_name not in allowed_models:
            return f"Model '{model_name}' is not in the allowed list"

        return None

    return check


def check_max_tokens(max_tokens: int) -> Callable[[str, dict[str, Any]], str | None]:
    """
    Create a pre-check that validates max token parameter.

    Args:
        max_tokens: Maximum allowed max_tokens value

    Returns:
        Pre-check function
    """
    def check(action_type: str, params: dict[str, Any]) -> str | None:
        if action_type != "model_invoke":
            return None

        requested = params.get("parameters", {}).get("max_tokens")
        if requested is not None and requested > max_tokens:
            return f"Requested max_tokens ({requested}) exceeds limit ({max_tokens})"

        return None

    return check


# Convenience function for fail-fast enforcement


def enforce_or_fail(
    enforcer: PolicyEnforcer,
    action_type: str,
    params: dict[str, Any],
    violation_type: ViolationType = ViolationType.POLICY_CONSTRAINT,
) -> None:
    """
    Run pre-checks and raise on violation (fail-fast mode).

    Args:
        enforcer: The policy enforcer
        action_type: Type of action being performed
        params: Action parameters
        violation_type: Type of violation to emit if check fails

    Raises:
        PolicyViolationError: If any pre-check fails
    """
    error = enforcer.run_pre_checks(action_type, params)
    if error is not None:
        enforcer.emit_violation(
            violation_type=violation_type,
            root_cause=error,
            remediation="Review policy constraints and adjust action parameters",
        )
        raise PolicyViolationError(error)


class PolicyViolationError(Exception):
    """Exception raised when a policy violation occurs in fail-fast mode."""
    pass
