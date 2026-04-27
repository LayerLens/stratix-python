"""LayerLens Base Adapter.

Provides the abstract :class:`BaseAdapter` class that all framework
adapters must extend. Implements circuit-breaker-protected event
emission, :class:`CaptureConfig` filtering, lifecycle management, and
replay serialization.

Ported from ``ateam/stratix/sdk/python/adapters/base.py`` with the
following adaptations for the ``stratix-python`` SDK:

* ``StrEnum`` (3.11+) replaced with ``(str, Enum)`` mixin (3.8+ compat).
* Pydantic imports routed through ``layerlens._compat.pydantic`` so v1
  and v2 are both supported.
* Payload serialization uses ``layerlens._compat.pydantic.model_dump``
  (handles v1 ``.dict()`` vs v2 ``.model_dump()``).
"""

from __future__ import annotations

import time
import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.sinks import EventSink

from layerlens._compat.pydantic import Field, BaseModel, model_dump
from layerlens.instrument.adapters._base.capture import (
    ALWAYS_ENABLED_EVENT_TYPES,
    CaptureConfig,
)
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

# Forward reference: EventSink is defined in sinks.py, which itself does not
# import from this module, but adapter.py is imported by sinks.py via the
# package's _base/__init__.py order. To avoid circular imports we use a
# string annotation in the BaseAdapter constructor and the public sink
# methods, and import EventSink lazily inside add_sink at call time.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class AdapterStatus(str, Enum):
    """Health status of an adapter."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class AdapterCapability(str, Enum):
    """Capabilities an adapter may declare."""

    TRACE_TOOLS = "trace_tools"
    TRACE_MODELS = "trace_models"
    TRACE_STATE = "trace_state"
    TRACE_HANDOFFS = "trace_handoffs"
    TRACE_PROTOCOL_EVENTS = "trace_protocol_events"
    REPLAY = "replay"
    STREAMING = "streaming"


class AdapterHealth(BaseModel):
    """Snapshot of adapter health."""

    status: AdapterStatus = Field(description="Current status")
    framework_name: str = Field(description="Framework this adapter targets")
    framework_version: Optional[str] = Field(default=None, description="Detected framework version")
    adapter_version: str = Field(description="Adapter version string")
    message: Optional[str] = Field(default=None, description="Human-readable status detail")
    error_count: int = Field(default=0, description="Consecutive error count")
    circuit_open: bool = Field(default=False, description="True if circuit breaker is open")


class AdapterInfo(BaseModel):
    """Metadata describing an adapter."""

    name: str = Field(description="Adapter name")
    version: str = Field(description="Adapter version")
    framework: str = Field(description="Target framework name")
    framework_version: Optional[str] = Field(default=None, description="Detected framework version")
    capabilities: List[AdapterCapability] = Field(default_factory=list)
    author: str = Field(default="LayerLens")
    description: str = Field(default="")
    requires_pydantic: PydanticCompat = Field(
        default=PydanticCompat.V1_OR_V2,
        description=(
            "Declared Pydantic major-version compatibility. Surfaced in the "
            "manifest so the atlas-app catalog UI can warn users before they "
            "pin an incompatible runtime."
        ),
    )


class ReplayableTrace(BaseModel):
    """A trace serialized for replay.

    Contains enough information to re-execute the original agent run
    with identical or modified inputs.
    """

    adapter_name: str = Field(description="Adapter that produced the trace")
    framework: str = Field(description="Framework used")
    trace_id: str = Field(description="Original trace ID")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Ordered event dicts")
    state_snapshots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Checkpoint state snapshots",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter/framework config at time of trace",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Null-object sentinel
# ---------------------------------------------------------------------------


class _NullStratix:
    """Null-object sentinel used when an adapter is constructed without a
    LayerLens client instance.

    Silently discards all calls so adapters can still be used stand-alone
    or in tests. Evaluates to falsy so ``if self._stratix:`` guards work
    correctly.

    Note on multi-tenancy: ``_NullStratix`` does **not** expose an
    ``org_id``. Callers constructing an adapter without a real client
    must supply ``org_id`` explicitly to :class:`BaseAdapter` — see the
    fail-fast logic in :meth:`BaseAdapter.__init__`.
    """

    def __bool__(self) -> bool:
        return False

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _emit_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def agent_id(self) -> str:
        return "null"

    @property
    def framework(self) -> Optional[str]:
        return None

    @property
    def is_policy_violated(self) -> bool:
        return False


_NULL_STRATIX = _NullStratix()


# ---------------------------------------------------------------------------
# Multi-tenancy: org_id resolution
# ---------------------------------------------------------------------------

# Reserved field name carried on every emitted event payload. The
# canonical multi-tenant key throughout the LayerLens platform. Cache
# keys, queue topics, RLS policies, and downstream stores all read this
# field to scope data to a single tenant. See CLAUDE.md ("EVERY data
# operation must be scoped by tenant").
ORG_ID_FIELD: str = "org_id"


def _resolve_org_id(stratix: Any, explicit: Optional[str]) -> str:
    """Resolve the tenant ``org_id`` for an adapter at construction time.

    Resolution order:

    1. The explicit ``org_id`` keyword argument (highest precedence).
    2. ``stratix.org_id`` if present and non-empty.
    3. ``stratix.organization_id`` if present and non-empty (the public
       :class:`layerlens.Stratix` client exposes this field).

    Args:
        stratix: The LayerLens client instance attached to the adapter.
        explicit: Explicit ``org_id`` passed to the adapter constructor.

    Returns:
        The resolved non-empty ``org_id`` string.

    Raises:
        ValueError: If no non-empty ``org_id`` can be resolved. Adapters
            MUST be tenant-scoped at construction; silent fallback to a
            blank or null value would violate the multi-tenancy contract
            (CLAUDE.md). This is an explicit fail-fast — never a silent
            skip.
    """
    if explicit is not None and isinstance(explicit, str) and explicit.strip():
        return explicit
    if stratix is not None and not isinstance(stratix, _NullStratix):
        for attr in ("org_id", "organization_id"):
            value = getattr(stratix, attr, None)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(
        "BaseAdapter requires a non-empty org_id for multi-tenant event "
        "scoping. Pass `org_id=...` explicitly to the adapter constructor "
        "or attach a LayerLens client whose `org_id`/`organization_id` "
        "field is populated. Silent fallback is prohibited — see "
        "CLAUDE.md multi-tenancy requirements."
    )


# ---------------------------------------------------------------------------
# Circuit breaker constants
# ---------------------------------------------------------------------------

_CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive errors before opening
_CIRCUIT_BREAKER_COOLDOWN_S = 60.0  # seconds before attempting recovery


# ---------------------------------------------------------------------------
# BaseAdapter ABC
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """Abstract base class for all LayerLens framework adapters.

    Provides:

    * Circuit-breaker-protected :meth:`emit_event`.
    * :class:`CaptureConfig` filtering.
    * Lifecycle management (:meth:`connect` / :meth:`disconnect` / :meth:`health_check`).
    * Replay serialization hook (:meth:`serialize_for_replay`).
    * **Multi-tenant ``org_id`` binding** — every adapter is bound to
      exactly one tenant at construction. The bound value is stamped
      into every emitted event payload (see :meth:`_stamp_org_id`).
      Construction without a resolvable ``org_id`` raises
      ``ValueError`` (fail-fast). See ``docs/adapters/multi-tenancy.md``
      for the full contract.
    """

    # Subclasses MUST set these.
    FRAMEWORK: str = ""
    VERSION: str = "0.0.0"

    # Per-adapter Pydantic v1/v2 compatibility declaration (Round-2 item 20).
    # Subclasses MUST set this explicitly to one of the three
    # :class:`PydanticCompat` values — the lint test in
    # ``tests/instrument/adapters/test_pydantic_compat.py`` enforces that
    # no framework adapter relies on the V1_OR_V2 default by accident.
    requires_pydantic: PydanticCompat = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
        event_sinks: Optional[List["EventSink"]] = None,
        *,
        org_id: Optional[str] = None,
    ) -> None:
        # Resolve the tenant org_id BEFORE attaching the stratix client.
        # If no org_id can be resolved, _resolve_org_id raises ValueError
        # — adapters fail fast at construction so emissions can never
        # ship without a tenant binding (CLAUDE.md multi-tenancy).
        self._org_id: str = _resolve_org_id(stratix, org_id)
        self._stratix = stratix or _NULL_STRATIX
        self._capture_config = capture_config or CaptureConfig()
        self._connected = False
        self._status: AdapterStatus = AdapterStatus.DISCONNECTED

        # Circuit breaker state (protected by _lock).
        self._lock = threading.Lock()
        self._error_count = 0
        self._circuit_open = False
        self._circuit_opened_at: float = 0.0

        # Collected events for replay serialization.
        self._trace_events: List[Dict[str, Any]] = []

        # Pluggable event sinks for persistence / export. Use add_sink /
        # remove_sink to mutate; direct list manipulation is not part of
        # the public API and may change in v2.
        self._event_sinks: List["EventSink"] = list(event_sinks) if event_sinks else []

    # --- Sink management (public API) ---

    def add_sink(self, sink: "EventSink") -> None:
        """Register an :class:`EventSink` to receive emitted events.

        Sinks are dispatched in registration order. A sink that raises
        from ``send`` / ``flush`` / ``close`` is logged at DEBUG and
        does not affect other sinks or the adapter's emission path.
        """
        self._event_sinks.append(sink)

    def remove_sink(self, sink: "EventSink") -> bool:
        """Remove a previously-registered sink.

        Returns ``True`` if the sink was present, ``False`` otherwise.
        """
        try:
            self._event_sinks.remove(sink)
            return True
        except ValueError:
            return False

    @property
    def sinks(self) -> List["EventSink"]:
        """Snapshot of currently-registered sinks (defensive copy)."""
        return list(self._event_sinks)

    # --- Properties ---

    @property
    def is_connected(self) -> bool:
        """True when the adapter has a live connection to its framework."""
        return self._connected

    @property
    def status(self) -> AdapterStatus:
        return self._status

    @property
    def capture_config(self) -> CaptureConfig:
        return self._capture_config

    @property
    def has_stratix(self) -> bool:
        """True when a real (non-null) client instance is attached."""
        return bool(self._stratix)

    @property
    def org_id(self) -> str:
        """The tenant ``org_id`` bound to this adapter at construction.

        Always non-empty — :meth:`__init__` raises ``ValueError`` if no
        ``org_id`` can be resolved. Every event emitted through this
        adapter is tagged with this value via the ``org_id`` payload
        field. See :data:`ORG_ID_FIELD`.
        """
        return self._org_id

    # --- Abstract lifecycle methods ---

    @abstractmethod
    def connect(self) -> None:
        """Verify framework availability and prepare the adapter.

        Implementations should import the framework, validate the
        version, and set ``self._connected = True`` /
        ``self._status = AdapterStatus.HEALTHY``.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Flush pending events and release resources.

        Implementations should set ``self._connected = False`` and
        ``self._status = AdapterStatus.DISCONNECTED``.
        """

    @abstractmethod
    def health_check(self) -> AdapterHealth:
        """Return a health snapshot."""

    @abstractmethod
    def get_adapter_info(self) -> AdapterInfo:
        """Return metadata about this adapter."""

    def info(self) -> AdapterInfo:
        """Return :class:`AdapterInfo` with the class-level compat decl applied.

        Subclasses populate the bulk of :class:`AdapterInfo` via
        :meth:`get_adapter_info`. This wrapper guarantees the
        ``requires_pydantic`` field reflects the subclass class attribute
        even when the subclass omits it from its constructor call —
        avoiding the need to repeat the value at every site. Used by
        :meth:`AdapterRegistry.info` and the manifest emitter.
        """
        base_info = self.get_adapter_info()
        if base_info.requires_pydantic != self.requires_pydantic:
            try:
                # Pydantic v2 path: copy with overrides.
                base_info = base_info.model_copy(update={"requires_pydantic": self.requires_pydantic})
            except AttributeError:
                # Pydantic v1 path.
                base_info = base_info.copy(update={"requires_pydantic": self.requires_pydantic})
        return base_info

    @abstractmethod
    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize the current trace data for replay."""

    # --- Replay execution hook ---

    async def execute_replay(
        self,
        inputs: Dict[str, Any],
        original_trace: Any,
        request: Any,
        replay_trace_id: str,
    ) -> Any:
        """Re-execute through this adapter's framework.

        Subclasses override this to provide actual re-execution. The
        default raises :class:`NotImplementedError` (synthetic replay
        used instead).

        Args:
            inputs: Reconstructed inputs for the replay.
            original_trace: The original SerializedTrace.
            request: The ReplayRequest.
            replay_trace_id: ID for the new replay trace.

        Returns:
            A SerializedTrace from the replay execution.

        Raises:
            NotImplementedError: If the adapter does not support replay.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support execute_replay()")

    # --- Concrete event emission ---

    def _stamp_org_id(self, payload: Any) -> Any:
        """Inject ``self._org_id`` into a payload's ``org_id`` field.

        Mutates Pydantic models via attribute write where possible and
        dict payloads via key write. The injection is unconditional —
        if the caller already populated ``org_id`` matching this
        adapter's tenant, the write is a no-op; if a *different* value
        was supplied (cross-tenant leak), this method overwrites it
        with the adapter's bound ``org_id``. That overwrite is
        intentional: an adapter constructed for tenant A must never
        emit events tagged with tenant B, regardless of caller error.

        Args:
            payload: The event payload — either a dict or a typed
                Pydantic model.

        Returns:
            The same payload reference, with ``org_id`` set.
        """
        if isinstance(payload, dict):
            payload[ORG_ID_FIELD] = self._org_id
            return payload
        # Pydantic model — attempt attribute write. Some models freeze
        # on construction; fall back to a meta dict on the model when
        # direct write is rejected.
        try:
            setattr(payload, ORG_ID_FIELD, self._org_id)
        except (AttributeError, TypeError, ValueError):
            meta = getattr(payload, "meta", None)
            if isinstance(meta, dict):
                meta[ORG_ID_FIELD] = self._org_id
        return payload

    def emit_event(
        self,
        payload: Any,
        privacy_level: Any = None,
    ) -> None:
        """Emit a typed event payload through the LayerLens pipeline.

        This method:

        1. Checks the circuit breaker — drops events if open (unless
           cooldown expired).
        2. Checks :class:`CaptureConfig` — silently drops events whose
           layer is disabled (cross-cutting events are never dropped).
        3. **Stamps the adapter's bound ``org_id`` onto the payload**
           (CLAUDE.md multi-tenancy requirement — every emission is
           tenant-scoped).
        4. Delegates to ``self._stratix.emit(payload, privacy_level)``
           with error counting for circuit-breaker state management.

        Args:
            payload: A Pydantic event payload (e.g.,
                ``ToolCallEvent.create(...)``).
            privacy_level: Optional ``PrivacyLevel`` override.
        """
        event_type = getattr(payload, "event_type", None)

        if not self._pre_emit_check(event_type):
            return

        payload = self._stamp_org_id(payload)

        try:
            if privacy_level is not None:
                self._stratix.emit(payload, privacy_level)
            else:
                self._stratix.emit(payload)

            self._post_emit_success(event_type, payload)
        except Exception:
            self._post_emit_failure()

    def emit_dict_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Emit a dict-based event through the LayerLens pipeline.

        Provides the same circuit-breaker and CaptureConfig gating as
        :meth:`emit_event` but accepts raw ``(event_type, dict)`` pairs
        used by the legacy adapter emission path. This avoids bypassing
        the BaseAdapter protections.

        **Multi-tenancy:** the adapter's bound ``org_id`` is stamped
        into ``payload[ORG_ID_FIELD]`` before the event is forwarded.
        Any caller-supplied value is overwritten with the adapter's
        own tenant binding to prevent cross-tenant leaks (CLAUDE.md).

        Args:
            event_type: Event type string (e.g., ``"model.invoke"``).
            payload: Raw event payload dict.
        """
        if not self._pre_emit_check(event_type):
            return

        payload = self._stamp_org_id(payload)

        try:
            self._stratix.emit(event_type, payload)
            self._post_emit_success(event_type, payload)
        except Exception:
            self._post_emit_failure()

    # --- Circuit breaker internals ---

    def _pre_emit_check(self, event_type: Optional[str]) -> bool:
        """Run circuit-breaker and CaptureConfig checks.

        Returns ``True`` to proceed with emission.
        """
        with self._lock:
            if self._circuit_open and not self._attempt_recovery():
                return False

        if event_type and event_type not in ALWAYS_ENABLED_EVENT_TYPES:
            # ``is_layer_enabled`` itself handles cross-cutting layer
            # families (commerce.* etc.) via prefix bypass — see
            # capture.py. The early-out above only catches exact
            # matches in the freeze-listed set.
            if not self._capture_config.is_layer_enabled(event_type):
                return False

        return True

    def _post_emit_success(self, event_type: Optional[str], payload: Any) -> None:
        """Handle successful emission: reset errors, record for replay.

        The serialized ``payload_data`` always carries ``org_id`` —
        either propagated from the typed/dict payload that
        :meth:`_stamp_org_id` already stamped, or re-injected here for
        Pydantic models whose ``model_dump`` strips unknown fields.
        Trace records and downstream sinks therefore see a payload
        whose ``org_id`` matches the adapter's bound tenant.
        """
        with self._lock:
            if self._error_count > 0:
                self._error_count = 0
                if self._status == AdapterStatus.DEGRADED:
                    self._status = AdapterStatus.HEALTHY

        if event_type:
            try:
                payload_data = model_dump(payload)
            except Exception:
                payload_data = {"raw": str(payload)}
            if isinstance(payload_data, dict):
                # Belt-and-braces: model_dump may have dropped a field
                # the model schema does not declare. Re-inject from the
                # adapter binding so the trace record + sinks always
                # carry org_id.
                payload_data[ORG_ID_FIELD] = self._org_id
            timestamp_ns = time.time_ns()
            self._trace_events.append(
                {
                    "event_type": event_type,
                    "payload": payload_data,
                    "timestamp_ns": timestamp_ns,
                    ORG_ID_FIELD: self._org_id,
                }
            )

            # Dispatch to pluggable event sinks. Sinks receive the
            # tenant binding both inside the payload dict and as the
            # explicit ``org_id`` keyword for tenant-aware persistence.
            # The :class:`EventSink` ABC requires the ``org_id`` keyword
            # — sinks that omit it raise at registration time, not here
            # at the hot dispatch path.
            if self._event_sinks:
                for sink in self._event_sinks:
                    try:
                        sink.send(
                            event_type,
                            payload_data,
                            timestamp_ns,
                            org_id=self._org_id,
                        )
                    except Exception:
                        logger.debug(
                            "EventSink %s.send() failed",
                            type(sink).__name__,
                            exc_info=True,
                        )

    def _post_emit_failure(self) -> None:
        """Handle emission failure: increment errors, maybe open circuit."""
        with self._lock:
            self._error_count += 1
            logger.debug(
                "Adapter %s emit error #%d",
                self.FRAMEWORK,
                self._error_count,
                exc_info=True,
            )
            if self._error_count >= _CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_open = True
                self._circuit_opened_at = time.monotonic()
                self._status = AdapterStatus.ERROR
                logger.warning(
                    "Adapter %s circuit breaker OPEN after %d consecutive errors",
                    self.FRAMEWORK,
                    self._error_count,
                )
            elif self._error_count >= _CIRCUIT_BREAKER_THRESHOLD // 2:
                self._status = AdapterStatus.DEGRADED

    def _attempt_recovery(self) -> bool:
        """Check if the circuit-breaker cooldown has elapsed.

        Caller MUST hold ``self._lock``.

        Returns:
            ``True`` if the circuit is now closed (ready to emit).
            ``False`` if still open.
        """
        elapsed = time.monotonic() - self._circuit_opened_at
        if elapsed >= _CIRCUIT_BREAKER_COOLDOWN_S:
            self._circuit_open = False
            self._error_count = 0
            self._status = AdapterStatus.DEGRADED
            logger.info("Adapter %s circuit breaker attempting recovery", self.FRAMEWORK)
            return True
        return False

    # --- Event sink lifecycle ---

    def _close_sinks(self) -> None:
        """Flush and close all attached event sinks."""
        for sink in self._event_sinks:
            try:
                sink.flush()
                sink.close()
            except Exception:
                logger.debug(
                    "EventSink %s close failed",
                    type(sink).__name__,
                    exc_info=True,
                )
