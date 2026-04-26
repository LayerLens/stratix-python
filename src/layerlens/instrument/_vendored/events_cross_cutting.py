"""Vendored snapshot of ``stratix.core.events.cross_cutting``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/cross_cutting.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- ``enum.StrEnum`` (added in Python 3.11) replaced with
  ``(str, Enum)`` mixin so the vendored enums behave identically on
  Python 3.9.
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]`` and ``Union[...]`` (Pydantic 2 evaluates
  field type hints via ``typing.get_type_hints``, which fails on
  Python 3.9 even with ``from __future__ import annotations``).

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Cross-Cutting Events
#
# From Step 1 specification:
#
# State Change Event:
# {
#     "event_type": "agent.state.change",
#     "state": {
#         "type": "internal | ephemeral",
#         "before_hash": "sha256",
#         "after_hash": "sha256"
#     }
# }
#
# Cost Event:
# {
#     "event_type": "cost.record",
#     "cost": {
#         "tokens": 1423,
#         "api_cost_usd": 0.031,
#         "infra_cost_usd": "unavailable"
#     }
# }
#
# Policy Violation Event:
# {
#     "event_type": "policy.violation",
#     "violation": {
#         "type": "privacy | compliance | safety",
#         "root_cause": "string",
#         "remediation": "string",
#         "failed_layer": "L3",
#         "failed_sequence_id": 17
#     }
# }
#
# Multi-Agent Handoff Event:
# {
#     "event_type": "agent.handoff",
#     "from_agent": "agent_A",
#     "to_agent": "agent_B",
#     "handoff_context_hash": "sha256"
# }

from __future__ import annotations

from enum import Enum
from typing import Any, Union, Optional

from pydantic import Field, BaseModel, field_validator


class StateType(str, Enum):
    """Type of agent state."""

    INTERNAL = "internal"
    EPHEMERAL = "ephemeral"


class StateInfo(BaseModel):
    """State information for state change events."""

    type: StateType = Field(description="Type of state (internal or ephemeral)")
    before_hash: str = Field(description="SHA-256 hash of state before change")
    after_hash: str = Field(description="SHA-256 hash of state after change")

    @field_validator("before_hash", "after_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate hash format."""
        if not v.startswith("sha256:"):
            raise ValueError("Hash must start with 'sha256:'")
        hex_part = v[7:]
        if len(hex_part) != 64:
            raise ValueError("Hash must be sha256: followed by 64 hex characters")
        return v


class AgentStateChangeEvent(BaseModel):
    """Cross-Cutting Event: Agent State Change.

    Represents a mutation to agent state.

    NORMATIVE:
    - State changes must hash before/after (even if state is redacted)
    - Emit on state mutation boundaries
    """

    event_type: str = Field(default="agent.state.change", description="Event type identifier")
    state: StateInfo = Field(description="State change information")

    @classmethod
    def create(
        cls,
        state_type: StateType,
        before_hash: str,
        after_hash: str,
    ) -> AgentStateChangeEvent:
        """Create a state change event.

        Args:
            state_type: Type of state.
            before_hash: Hash of state before change.
            after_hash: Hash of state after change.

        Returns:
            AgentStateChangeEvent instance.
        """
        return cls(
            state=StateInfo(
                type=state_type,
                before_hash=before_hash,
                after_hash=after_hash,
            )
        )


class CostInfo(BaseModel):
    """Cost information for cost record events."""

    tokens: Optional[int] = Field(default=None, ge=0, description="Number of tokens consumed")
    prompt_tokens: Optional[int] = Field(
        default=None, ge=0, description="Number of prompt tokens"
    )
    completion_tokens: Optional[int] = Field(
        default=None, ge=0, description="Number of completion tokens"
    )
    api_cost_usd: Optional[Union[float, str]] = Field(
        default=None, description="API cost in USD (or 'unavailable')"
    )
    infra_cost_usd: Optional[Union[float, str]] = Field(
        default=None, description="Infrastructure cost in USD (or 'unavailable')"
    )
    tool_calls: Optional[int] = Field(default=None, ge=0, description="Number of tool calls")


class CostRecordEvent(BaseModel):
    """Cross-Cutting Event: Cost Record.

    Represents cost/usage tracking data.

    NORMATIVE:
    - Costs must mark unavailable (never omit silently)
    - Emit on known cost/usage updates
    """

    event_type: str = Field(default="cost.record", description="Event type identifier")
    cost: CostInfo = Field(description="Cost information")

    @classmethod
    def create(
        cls,
        tokens: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        api_cost_usd: Optional[Union[float, str]] = None,
        infra_cost_usd: Optional[Union[float, str]] = None,
        tool_calls: Optional[int] = None,
    ) -> CostRecordEvent:
        """Create a cost record event."""
        return cls(
            cost=CostInfo(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                api_cost_usd=api_cost_usd,
                infra_cost_usd=infra_cost_usd,
                tool_calls=tool_calls,
            )
        )


class ViolationType(str, Enum):
    """Type of policy violation."""

    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    CAPTURE = "capture"  # Missing required layer/event
    POLICY_CONSTRAINT = "policy_constraint"  # Pre-check/policy constraint violation


class ViolationInfo(BaseModel):
    """Violation information for policy violation events."""

    type: ViolationType = Field(description="Type of violation")
    root_cause: str = Field(description="Root cause of the violation")
    remediation: str = Field(description="Suggested remediation action")
    failed_layer: Optional[str] = Field(default=None, description="Layer where violation occurred")
    failed_sequence_id: Optional[int] = Field(
        default=None, description="Sequence ID where violation occurred"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional violation details"
    )


class PolicyViolationEvent(BaseModel):
    """Cross-Cutting Event: Policy Violation.

    Represents a policy violation that terminates evaluation.

    NORMATIVE:
    - Evaluation terminates immediately
    - No further hashing occurs after violation
    - Must include root_cause, remediation, failed_layer, failed_sequence_id
    """

    event_type: str = Field(default="policy.violation", description="Event type identifier")
    violation: ViolationInfo = Field(description="Violation information")

    @classmethod
    def create(
        cls,
        violation_type: ViolationType,
        root_cause: str,
        remediation: str,
        failed_layer: Optional[str] = None,
        failed_sequence_id: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> PolicyViolationEvent:
        """Create a policy violation event."""
        return cls(
            violation=ViolationInfo(
                type=violation_type,
                root_cause=root_cause,
                remediation=remediation,
                failed_layer=failed_layer,
                failed_sequence_id=failed_sequence_id,
                details=details or {},
            )
        )


class AgentHandoffEvent(BaseModel):
    """Cross-Cutting Event: Agent Handoff.

    Represents delegation from one agent to another.

    NORMATIVE:
    - Emit when delegating to another agent
    - Include context hash/external reference
    - Propagate trace context to receiving agent
    """

    event_type: str = Field(default="agent.handoff", description="Event type identifier")
    from_agent: str = Field(description="Agent initiating the handoff")
    to_agent: str = Field(description="Agent receiving the handoff")
    handoff_context_hash: str = Field(description="SHA-256 hash of the handoff context")
    context_privacy_level: str = Field(
        default="cleartext", description="Privacy level of the handoff context"
    )

    @field_validator("handoff_context_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate hash format."""
        if not v.startswith("sha256:"):
            raise ValueError("Hash must start with 'sha256:'")
        hex_part = v[7:]
        if len(hex_part) != 64:
            raise ValueError("Hash must be sha256: followed by 64 hex characters")
        return v

    @classmethod
    def create(
        cls,
        from_agent: str,
        to_agent: str,
        handoff_context_hash: str,
        context_privacy_level: str = "cleartext",
    ) -> AgentHandoffEvent:
        """Create an agent handoff event."""
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            handoff_context_hash=handoff_context_hash,
            context_privacy_level=context_privacy_level,
        )


__all__ = [
    "StateType",
    "StateInfo",
    "AgentStateChangeEvent",
    "CostInfo",
    "CostRecordEvent",
    "ViolationType",
    "ViolationInfo",
    "PolicyViolationEvent",
    "AgentHandoffEvent",
]
