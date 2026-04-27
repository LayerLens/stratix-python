"""Tests for the typed-event foundation.

Pins the dual-path emission contract introduced by the
``feat/instrument-typed-events-foundation`` PR:

* :meth:`BaseAdapter.emit_event` — preferred path. Validates payloads
  through :func:`validate_typed_event` and REJECTS malformed inputs.
* :meth:`BaseAdapter.emit_dict_event` — legacy path. Emits a
  :class:`DeprecationWarning` on every call. Forwards the dict
  unchanged (no schema validation, because the 16 unmigrated
  framework adapters use adapter-specific dict shapes that
  intentionally diverge from the canonical schema).

The :data:`ALL_TYPED_EVENTS` registry is exercised end-to-end:
construction → validation → emission → dict serialisation.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from layerlens.instrument._compat.events import (
    ALL_TYPED_EVENTS,
    MessageRole,
    ToolCallEvent,
    ViolationType,
    AgentInputEvent,
    CostRecordEvent,
    EnvironmentType,
    AgentOutputEvent,
    ModelInvokeEvent,
    PolicyViolationEvent,
    EnvironmentConfigEvent,
    TypedEventValidationError,
    coerce_to_dict,
    validate_typed_event,
)
from layerlens.instrument.adapters._base import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Captures every emit call into ``self.events``.

    Records both shapes:

    * Two-arg legacy path: ``emit(event_type, dict)`` →
      ``{"shape": "dict", "event_type": ..., "payload": ...}``.
    * Single-arg typed path: ``emit(payload_model)`` →
      ``{"shape": "typed", "event_type": ..., "payload": ...}``.

    The shape tag is used by the dual-path tests below to assert that
    typed and dict events take different code paths under the hood.
    """

    org_id: str = "org-typed-events"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.raw_args: List[Tuple[Any, ...]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.raw_args.append(args)
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append(
                {"shape": "dict", "event_type": args[0], "payload": args[1]}
            )
            return
        if args:
            payload = args[0]
            self.events.append(
                {
                    "shape": "typed",
                    "event_type": getattr(payload, "event_type", "<unknown>"),
                    "payload": payload,
                }
            )


class _MinimalAdapter(BaseAdapter):
    """Concrete adapter exercising the base emission paths only."""

    FRAMEWORK = "test"
    VERSION = "0.0.1"

    def connect(self) -> None:
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(name="MinimalAdapter", version=self.VERSION, framework=self.FRAMEWORK)

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="MinimalAdapter",
            framework=self.FRAMEWORK,
            trace_id="test-trace",
            events=list(self._trace_events),
        )


class _OpenAdapter(_MinimalAdapter):
    """Adapter that opts into ``ALLOW_UNREGISTERED_EVENTS=True``."""

    FRAMEWORK = "test-open"
    ALLOW_UNREGISTERED_EVENTS = True


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_all_typed_events_registry_keys_match_event_type_default() -> None:
    """Every registered model's ``event_type`` default matches its key.

    Pins the canonical schema invariant: the registry key IS the event
    type string the model carries on the wire. A mismatch between the
    two surfaces would silently route events through the wrong
    validator.
    """
    for event_type, model_cls in ALL_TYPED_EVENTS.items():
        instance: Any = model_cls.model_construct() if hasattr(model_cls, "model_construct") else model_cls.construct()
        assert (
            getattr(instance, "event_type", None) == event_type
        ), f"{model_cls.__name__} default event_type does not match registry key {event_type!r}"


def test_all_typed_events_registry_covers_canonical_types() -> None:
    """The 12 canonical event payload types are all registered.

    Mirrors the classes in ``ateam/stratix/core/events/`` (L1, L3, L4,
    L5, cross-cutting). Adding a new canonical type without
    registering it here is what this test catches.
    """
    expected = {
        "agent.input",
        "agent.output",
        "model.invoke",
        "environment.config",
        "environment.metrics",
        "tool.call",
        "tool.logic",
        "tool.environment",
        "agent.state.change",
        "agent.handoff",
        "cost.record",
        "policy.violation",
    }
    assert set(ALL_TYPED_EVENTS) == expected


# ---------------------------------------------------------------------------
# validate_typed_event behaviour
# ---------------------------------------------------------------------------


def test_validate_typed_event_passes_through_typed_payload() -> None:
    """Already-typed payloads are returned unchanged (fast path)."""
    payload = ToolCallEvent.create(name="calc", input_data={"x": 1})
    result = validate_typed_event("tool.call", payload)
    assert result is payload


def test_validate_typed_event_parses_valid_dict() -> None:
    """Well-formed dicts are parsed into the registered model."""
    result = validate_typed_event(
        "tool.call",
        {
            "tool": {"name": "calc", "version": "1.0", "integration": "library"},
            "input": {"x": 1},
        },
    )
    assert isinstance(result, ToolCallEvent)
    assert result.tool.name == "calc"


def test_validate_typed_event_rejects_invalid_dict() -> None:
    """Dicts missing required fields raise TypedEventValidationError."""
    with pytest.raises(TypedEventValidationError) as excinfo:
        validate_typed_event(
            "tool.call",
            {"tool": {"name": "calc"}},  # missing version + integration
        )
    assert excinfo.value.event_type == "tool.call"


def test_validate_typed_event_rejects_unregistered_event_type() -> None:
    """Unknown event_type raises unless ``allow_unregistered=True``."""
    with pytest.raises(TypedEventValidationError):
        validate_typed_event("custom.frobnicate", {"x": 1})


def test_validate_typed_event_allows_unregistered_with_opt_in() -> None:
    """``allow_unregistered=True`` wraps unknown dicts in an open model."""
    result = validate_typed_event(
        "custom.frobnicate", {"x": 1}, allow_unregistered=True
    )
    assert getattr(result, "event_type", None) == "custom.frobnicate"


def test_validate_typed_event_falls_back_to_payload_event_type() -> None:
    """Missing ``event_type`` arg falls back to ``payload['event_type']``."""
    result = validate_typed_event(
        None,
        {
            "event_type": "model.invoke",
            "model": {"provider": "openai", "name": "gpt-5", "version": "2026-04"},
        },
    )
    assert isinstance(result, ModelInvokeEvent)
    assert result.model.name == "gpt-5"


def test_validate_typed_event_rejects_payload_without_event_type() -> None:
    """Payloads missing event_type entirely are rejected."""
    with pytest.raises(TypedEventValidationError):
        validate_typed_event(None, {"some": "data"})


# ---------------------------------------------------------------------------
# coerce_to_dict
# ---------------------------------------------------------------------------


def test_coerce_to_dict_handles_typed_model() -> None:
    payload = AgentInputEvent.create(message="hi", role=MessageRole.HUMAN)
    out = coerce_to_dict(payload)
    assert out["event_type"] == "agent.input"
    assert out["content"]["message"] == "hi"


def test_coerce_to_dict_passes_through_dict() -> None:
    out = coerce_to_dict({"event_type": "x", "k": 1})
    assert out == {"event_type": "x", "k": 1}


# ---------------------------------------------------------------------------
# Dual-path emission via BaseAdapter
# ---------------------------------------------------------------------------


def test_emit_event_typed_path_invokes_stratix_with_model_only() -> None:
    """``emit_event(model)`` calls ``stratix.emit(payload)`` (single arg)."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    adapter.emit_event(
        EnvironmentConfigEvent.create(
            env_type=EnvironmentType.SIMULATED, attributes={"k": "v"}
        )
    )

    assert len(stratix.events) == 1
    assert stratix.events[0]["shape"] == "typed"
    assert stratix.events[0]["event_type"] == "environment.config"


def test_emit_event_rejects_invalid_typed_payload_dict() -> None:
    """``emit_event`` rejects dicts that fail canonical validation."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with pytest.raises(TypedEventValidationError):
        adapter.emit_event({"event_type": "tool.call", "tool": {"name": "calc"}})

    # Nothing emitted to the client when validation fails.
    assert stratix.events == []


def test_emit_dict_event_emits_deprecation_warning() -> None:
    """Every ``emit_dict_event`` call raises a DeprecationWarning."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        adapter.emit_dict_event("agent.input", {"input": "hello"})

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert "emit_dict_event" in str(deprecation_warnings[0].message)
    assert "agent.input" in str(deprecation_warnings[0].message)


def test_emit_dict_event_forwards_dict_to_stratix_unchanged() -> None:
    """Legacy dict path forwards (event_type, dict) to ``stratix.emit``.

    The legacy path does NOT run canonical schema validation — adapter
    tests for the 16 unmigrated adapters rely on this so their
    custom dict shapes still flow through.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter.emit_dict_event(
            "tool.call",
            {"framework": "agno", "tool_name": "calc"},  # adapter-specific shape
        )

    assert len(stratix.events) == 1
    assert stratix.events[0]["shape"] == "dict"
    # Non-canonical fields still present (no schema enforcement on
    # this path).
    assert stratix.events[0]["payload"]["tool_name"] == "calc"
    # And org_id was stamped by _stamp_org_id.
    assert stratix.events[0]["payload"]["org_id"] == "org-typed-events"


def test_emit_event_typed_path_records_in_replay_trace() -> None:
    """Successful typed emits are persisted into the replay buffer.

    The :class:`ReplayableTrace` events list carries one entry per
    successful emission; each entry has ``payload`` (dict),
    ``event_type``, ``timestamp_ns``, and ``org_id`` at the envelope
    level.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    adapter.emit_event(CostRecordEvent.create(prompt_tokens=10, completion_tokens=5))
    rt = adapter.serialize_for_replay()

    assert len(rt.events) == 1
    evt = rt.events[0]
    assert evt["event_type"] == "cost.record"
    assert evt["org_id"] == "org-typed-events"
    assert evt["payload"]["org_id"] == "org-typed-events"


def test_emit_event_with_open_adapter_accepts_unregistered_event_type() -> None:
    """``ALLOW_UNREGISTERED_EVENTS=True`` lets through arbitrary dicts."""
    stratix = _RecordingStratix()
    adapter = _OpenAdapter(stratix=stratix)
    adapter.connect()

    # No registered model for "custom.thing" — strict adapter would reject.
    adapter.emit_event({"event_type": "custom.thing", "data": 42})
    assert len(stratix.events) == 1
    assert stratix.events[0]["event_type"] == "custom.thing"


def test_strict_adapter_rejects_unregistered_event_type() -> None:
    """The default (strict) adapter rejects unregistered event types."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with pytest.raises(TypedEventValidationError):
        adapter.emit_event({"event_type": "custom.thing", "data": 42})

    assert stratix.events == []


def test_emit_event_validation_failure_increments_circuit_breaker_errors() -> None:
    """Schema validation failures count toward the circuit breaker.

    Pins the CLAUDE.md 'never silently skip' rule: even though the
    error is raised back to the caller, the failure is also recorded
    in the adapter's error counter so persistent validation failures
    eventually trip the circuit breaker.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    initial_errors = adapter._error_count
    with pytest.raises(TypedEventValidationError):
        adapter.emit_event({"event_type": "tool.call", "tool": {"name": "x"}})
    assert adapter._error_count == initial_errors + 1


def test_emit_event_typed_payload_preserves_schema_round_trip() -> None:
    """A typed payload survives emit + replay + model_dump unchanged."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    payload = AgentOutputEvent.create(message="done", metadata={"k": "v"})
    adapter.emit_event(payload)

    rt = adapter.serialize_for_replay()
    assert rt.events[0]["payload"]["content"]["message"] == "done"
    assert rt.events[0]["payload"]["content"]["metadata"]["k"] == "v"


def test_emit_event_typed_payload_with_validator_constraint() -> None:
    """Validator-bearing typed payloads enforce their constraints.

    Pins that :class:`PolicyViolationEvent` instances constructed
    through ``emit_event`` retain their canonical validation — e.g.
    the ``violation.type`` enum is enforced by Pydantic itself.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    payload = PolicyViolationEvent.create(
        violation_type=ViolationType.SAFETY,
        root_cause="prompt injection detected",
        remediation="block + alert",
        failed_layer="L1",
    )
    adapter.emit_event(payload)

    assert stratix.events[0]["event_type"] == "policy.violation"
    assert stratix.events[0]["payload"].violation.type == ViolationType.SAFETY


def test_emit_event_circuit_breaker_open_drops_event() -> None:
    """Open circuit breaker silently drops typed events (existing contract)."""
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()
    adapter._circuit_open = True

    # Avoid recovery firing.
    with patch("time.monotonic", return_value=0.0):
        adapter._circuit_opened_at = 0.0
        adapter.emit_event(ToolCallEvent.create(name="x", input_data={}))

    # Circuit breaker dropped the event before it reached stratix.
    assert stratix.events == []


def test_dict_event_path_does_not_validate_canonical_schema() -> None:
    """Legacy dict path tolerates non-canonical adapter-specific shapes.

    Documents the *intentional* gap: until the 16 unmigrated adapters
    move to typed events (see typed-events-followups.md), their
    adapter-specific dicts must still flow through. The
    DeprecationWarning is what keeps the gap visible.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # This dict would FAIL canonical validation — wrong shape for
        # tool.call. The legacy path forwards it anyway.
        adapter.emit_dict_event(
            "tool.call",
            {"framework": "x", "tool_name": "y", "tool_input": {}},
        )

    assert len(stratix.events) == 1
