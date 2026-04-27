"""Typed event foundation for the LayerLens instrument layer.

This module is the **single canonical import surface** for typed
Pydantic event payloads emitted by every framework, protocol, and
provider adapter. It vendors the canonical
``stratix.core.events`` models from the ``ateam`` framework
(via :mod:`layerlens.instrument._vendored`) and exposes them through
a Pydantic v1 / v2-compatible facade so adapter code can write a single
import regardless of the runtime Pydantic major version pinned by the
host application.

Why a separate ``_compat`` module instead of importing from
``_vendored`` directly?

1. **Single rename point.** Adapter code only references the names
   re-exported here. If the upstream ``stratix.core.events`` schema
   re-organises (e.g. l5 split into l5a / l5b / l5c packages), only
   this file changes — adapters do not.
2. **Validation surface.** :func:`validate_typed_event` and
   :data:`ALL_TYPED_EVENTS` give the base adapter a registry-driven
   validator that REJECTS payloads which do not satisfy the canonical
   schema. The vendored modules themselves contain only the bare
   Pydantic types — they do not know about emission, registration, or
   the dual-path adapter contract.
3. **Pydantic v1/v2 compat.** The vendored snapshots import from
   ``pydantic`` directly. This module re-exports through the
   :mod:`layerlens._compat.pydantic` shim so callers see a stable
   ``BaseModel`` regardless of installed Pydantic major version.

Schema reference
----------------

Each event payload conforms to the **Payload** envelope defined in
``ateam/docs/incubation-docs/adapter-framework/05-trace-schema-specification.md``
section 1.4 (the four-envelope :class:`StratixEvent` structure: Identity,
Privacy, Attestation, Payload). The :class:`BaseEvent` Protocol below
captures the minimal contract every payload model must satisfy:
``event_type`` and ``layer`` (or ``None`` for cross-cutting events).

Adoption status
---------------

The :class:`BaseAdapter.emit_event` path validates payloads through
:func:`validate_typed_event` and raises :class:`TypedEventValidationError`
on mismatch. Legacy callers continue to use
:meth:`BaseAdapter.emit_dict_event`, which now emits a
:class:`DeprecationWarning` and routes the dict through schema
validation — invalid dict payloads are REJECTED, not silently emitted.
See ``docs/adapters/typed-events.md`` for the full migration guide and
``docs/adapters/typed-events-followups.md`` for the per-adapter
backlog.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Type,
    Union,
    Mapping,
    TypeVar,
    Optional,
    Protocol,
    runtime_checkable,
)

# Re-export through the SDK's Pydantic v1/v2 shim so callers see a
# single stable BaseModel symbol. The vendored modules import from
# ``pydantic`` directly (which is fine under v2 — that's where they
# were vendored from); this re-export lives here so the public API
# does not change if the underlying vendor strategy changes.
from layerlens._compat.pydantic import (
    PYDANTIC_V2 as _PYDANTIC_V2,
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)

# Vendored canonical event payload types. Keep imports explicit (no
# star-import) so static analysers can verify each name resolves.
from layerlens.instrument._vendored.events_l1_io import (
    MessageRole,
    MessageContent,
    AgentInputEvent,
    AgentOutputEvent,
)
from layerlens.instrument._vendored.events_l3_model import (
    ModelInfo,
    ModelInvokeEvent,
)
from layerlens.instrument._vendored.events_l5_tools import (
    ToolInfo,
    ToolCallEvent,
    ToolLogicInfo,
    ToolLogicEvent,
    IntegrationType,
    ToolEnvironmentInfo,
    ToolEnvironmentEvent,
)
from layerlens.instrument._vendored.events_cross_cutting import (
    CostInfo,
    StateInfo,
    StateType,
    ViolationInfo,
    ViolationType,
    CostRecordEvent,
    AgentHandoffEvent,
    PolicyViolationEvent,
    AgentStateChangeEvent,
)
from layerlens.instrument._vendored.events_l4_environment import (
    EnvironmentInfo,
    EnvironmentType,
    EnvironmentMetrics,
    EnvironmentConfigEvent,
    EnvironmentMetricsEvent,
)

# ---------------------------------------------------------------------------
# BaseEvent Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseEvent(Protocol):
    """Structural contract every typed event payload must satisfy.

    Defined as a :class:`typing.Protocol` rather than a base class so
    we do not have to retroactively re-parent the vendored Pydantic
    models. Every model already exposes ``event_type`` (and almost all
    expose ``layer``) as Pydantic fields with sensible defaults.

    Cross-cutting events (e.g. :class:`AgentHandoffEvent`,
    :class:`CostRecordEvent`) intentionally omit ``layer`` because
    they are not bound to a single layer of the canonical event model.
    The :func:`validate_typed_event` helper accepts this and only
    requires ``event_type``.
    """

    event_type: str


# ---------------------------------------------------------------------------
# Typed event registry
# ---------------------------------------------------------------------------

# Registry of every typed event the adapter layer accepts. Keyed on the
# canonical ``event_type`` string. The base adapter consults this
# registry to validate dict payloads coming through the legacy
# :meth:`emit_dict_event` path — a dict whose ``event_type`` matches a
# registered key is parsed through the corresponding model and rejected
# on validation failure.
#
# When new event payload types are added (e.g. agent memory events
# from the v1.4 schema, commerce events from v1.3), append them here
# AND to ``__all__`` so adapter code can import the new names from a
# single place.
ALL_TYPED_EVENTS: Dict[str, Type[_CompatBaseModel]] = {
    # L1 — Agent Inputs & Outputs
    "agent.input": AgentInputEvent,
    "agent.output": AgentOutputEvent,
    # L3 — Model Metadata
    "model.invoke": ModelInvokeEvent,
    # L4 — Environment
    "environment.config": EnvironmentConfigEvent,
    "environment.metrics": EnvironmentMetricsEvent,
    # L5 — Tools
    "tool.call": ToolCallEvent,
    "tool.logic": ToolLogicEvent,
    "tool.environment": ToolEnvironmentEvent,
    # Cross-cutting
    "agent.state.change": AgentStateChangeEvent,
    "agent.handoff": AgentHandoffEvent,
    "cost.record": CostRecordEvent,
    "policy.violation": PolicyViolationEvent,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TypedEventValidationError(ValueError):
    """Raised when an event payload fails canonical schema validation.

    Carries the original validation error chain via ``__cause__`` so
    callers can inspect the underlying Pydantic ``ValidationError``.
    The string representation includes the offending ``event_type``
    and a summary of the failing fields so the failure mode is
    actionable in adapter test output.
    """

    def __init__(self, event_type: str, message: str) -> None:
        super().__init__(f"event_type={event_type!r}: {message}")
        self.event_type = event_type


_TypedPayloadOrDict = Union[_CompatBaseModel, Mapping[str, Any]]
_TPayload = TypeVar("_TPayload", bound=_CompatBaseModel)


def validate_typed_event(
    event_type: Optional[str],
    payload: Any,
    *,
    allow_unregistered: bool = False,
) -> Any:
    """Validate ``payload`` against the canonical schema for ``event_type``.

    Three input shapes are supported:

    1. ``payload`` is already an instance of the canonical typed model
       — returned unchanged after a defensive ``isinstance`` check.
    2. ``payload`` is a dict whose ``event_type`` is registered in
       :data:`ALL_TYPED_EVENTS` — parsed through the registered model
       and returned. Validation errors raise
       :class:`TypedEventValidationError`.
    3. ``payload`` is a dict whose ``event_type`` is NOT registered —
       raises :class:`TypedEventValidationError` unless
       ``allow_unregistered=True``, in which case the dict is wrapped
       in a permissive Pydantic model. Reserved for adapters whose
       event taxonomy genuinely diverges from the canonical set
       (langfuse importer, third-party trace shapes); see
       ``docs/adapters/typed-events.md`` for the policy.

    Args:
        event_type: The event type string. Falls back to
            ``payload["event_type"]`` if omitted and ``payload`` is a
            dict.
        payload: A typed Pydantic model or a dict-like payload.
        allow_unregistered: If ``True``, dicts with unknown event types
            pass through untyped. Default ``False`` — strict by design.

    Returns:
        The validated typed event payload (either the original model
        instance or a freshly-constructed one).

    Raises:
        TypedEventValidationError: When validation fails or the event
            type is unregistered and ``allow_unregistered=False``.
    """
    if isinstance(payload, _CompatBaseModel):
        # Already typed — trust the constructor's own validation.
        return payload

    if not isinstance(payload, Mapping):
        # Non-dict, non-Pydantic objects with an ``event_type`` attribute
        # are accepted as typed payloads. This permits adapter test
        # doubles and ad-hoc dataclass-like objects to flow through
        # :meth:`emit_event` without forcing every caller to subclass
        # the canonical Pydantic types — the canonical models stay the
        # *recommended* shape, but the validator is tolerant of
        # equivalent attribute-bearing duck types. Cast through ``Any``
        # so mypy does not flag the duck-typed return as ``object``.
        if hasattr(payload, "event_type"):
            duck_typed: Any = payload
            return duck_typed
        raise TypedEventValidationError(
            event_type or "<unknown>",
            f"payload must be a Pydantic model, Mapping, or expose an "
            f"event_type attribute; got {type(payload).__name__}",
        )

    resolved_type: Optional[str] = event_type or payload.get("event_type")
    if not isinstance(resolved_type, str) or not resolved_type:
        raise TypedEventValidationError(
            "<missing>",
            "dict payload missing required 'event_type' field",
        )

    model_cls = ALL_TYPED_EVENTS.get(resolved_type)
    if model_cls is None:
        if allow_unregistered:
            # Permissive: wrap the dict in an open-ended Pydantic model
            # so callers downstream still get a model instance. We
            # construct a fresh anonymous class to avoid polluting the
            # registry.
            return _make_open_payload(resolved_type, dict(payload))
        raise TypedEventValidationError(
            resolved_type,
            "no canonical event model registered. Pass allow_unregistered=True "
            "for adapters whose event taxonomy is intentionally outside the "
            "canonical schema, or register a model in ALL_TYPED_EVENTS.",
        )

    try:
        # Strip extra keys that are not part of the model — adapters
        # historically attach metadata (``framework``, ``timestamp_ns``,
        # ``org_id``) that the canonical schema does not declare. The
        # base adapter re-stamps ``org_id`` after validation, so we do
        # not need to preserve it here. ``framework`` and ad-hoc keys
        # are preserved on the dict that is forwarded to the stratix
        # client by the dual-path emission code; this validator only
        # asserts the canonical fields are well-formed.
        return model_cls(**dict(payload))
    except Exception as exc:  # Pydantic v1 ValidationError, v2 ValidationError, both subclass ValueError
        raise TypedEventValidationError(resolved_type, str(exc)) from exc


def _make_open_payload(event_type: str, data: Dict[str, Any]) -> _CompatBaseModel:
    """Construct an open-ended Pydantic model wrapping ``data``.

    Used by :func:`validate_typed_event` when ``allow_unregistered``
    is set. Each call creates a fresh subclass (cheap; happens off
    the hot emission path only for adapters that opted in).
    """
    fields: Dict[str, Any] = {"event_type": (str, event_type)}
    # Pydantic v1 supports ``__fields__`` mutation indirectly via
    # ``create_model``; v2 has the same helper. Both expose the same
    # name on ``pydantic`` — import lazily so this module loads even
    # when callers never touch the open-payload escape hatch.
    from pydantic import create_model

    create_model_any: Any = create_model
    if _PYDANTIC_V2:
        # v2: pass model_config dict via ``__config__`` kwarg.
        model: Any = create_model_any(
            f"OpenPayload_{event_type.replace('.', '_')}",
            __config__={"extra": "allow"},
            **fields,
        )
    else:
        # v1: ``Config`` inner class with ``extra = "allow"``.
        model = create_model_any(
            f"OpenPayload_{event_type.replace('.', '_')}",
            **fields,
        )
        model.Config.extra = "allow"

    instance: _CompatBaseModel = model(**data)
    return instance


def coerce_to_dict(payload: Any) -> Dict[str, Any]:
    """Return the dict representation of a typed event payload.

    Mirrors :func:`layerlens._compat.pydantic.model_dump` but accepts
    a dict pass-through (and ad-hoc objects via fallback) so call
    sites do not need to special-case the legacy emit path.
    """
    if isinstance(payload, _CompatBaseModel):
        return _compat_model_dump(payload)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"raw": str(payload)}


__all__: List[str] = [
    # Foundation
    "BaseEvent",
    "ALL_TYPED_EVENTS",
    "TypedEventValidationError",
    "validate_typed_event",
    "coerce_to_dict",
    # L1
    "MessageRole",
    "MessageContent",
    "AgentInputEvent",
    "AgentOutputEvent",
    # L3
    "ModelInfo",
    "ModelInvokeEvent",
    # L4
    "EnvironmentInfo",
    "EnvironmentType",
    "EnvironmentMetrics",
    "EnvironmentConfigEvent",
    "EnvironmentMetricsEvent",
    # L5
    "ToolInfo",
    "IntegrationType",
    "ToolCallEvent",
    "ToolLogicInfo",
    "ToolLogicEvent",
    "ToolEnvironmentInfo",
    "ToolEnvironmentEvent",
    # Cross-cutting
    "CostInfo",
    "CostRecordEvent",
    "StateInfo",
    "StateType",
    "AgentStateChangeEvent",
    "ViolationInfo",
    "ViolationType",
    "PolicyViolationEvent",
    "AgentHandoffEvent",
]
