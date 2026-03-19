"""
STRATIX Base Adapter

Provides the abstract BaseAdapter class that all framework adapters must extend.
Implements circuit-breaker-protected event emission, CaptureConfig filtering,
lifecycle management, and replay serialization.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from layerlens.instrument.adapters._capture import ALWAYS_ENABLED_EVENT_TYPES, CaptureConfig

if TYPE_CHECKING:
    from layerlens.instrument.schema.privacy import PrivacyLevel

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


class AdapterHealth(BaseModel):
    """Snapshot of adapter health."""
    status: AdapterStatus = Field(description="Current status")
    framework_name: str = Field(description="Framework this adapter targets")
    framework_version: str | None = Field(default=None, description="Detected framework version")
    adapter_version: str = Field(description="Adapter version string")
    message: str | None = Field(default=None, description="Human-readable status detail")
    error_count: int = Field(default=0, description="Consecutive error count")
    circuit_open: bool = Field(default=False, description="True if circuit breaker is open")


class AdapterCapability(str, Enum):
    """Capabilities an adapter may declare."""
    TRACE_TOOLS = "trace_tools"
    TRACE_MODELS = "trace_models"
    TRACE_STATE = "trace_state"
    TRACE_HANDOFFS = "trace_handoffs"
    TRACE_PROTOCOL_EVENTS = "trace_protocol_events"
    REPLAY = "replay"
    STREAMING = "streaming"


class AdapterInfo(BaseModel):
    """Metadata describing an adapter."""
    name: str = Field(description="Adapter name")
    version: str = Field(description="Adapter version")
    framework: str = Field(description="Target framework name")
    framework_version: str | None = Field(default=None, description="Detected framework version")
    capabilities: list[AdapterCapability] = Field(default_factory=list)
    author: str = Field(default="STRATIX Team")
    description: str = Field(default="")


class ReplayableTrace(BaseModel):
    """
    A trace serialized for replay.

    Contains enough information to re-execute the original agent run
    with identical or modified inputs.
    """
    adapter_name: str = Field(description="Adapter that produced the trace")
    framework: str = Field(description="Framework used")
    trace_id: str = Field(description="Original trace ID")
    events: list[dict[str, Any]] = Field(default_factory=list, description="Ordered event dicts")
    state_snapshots: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Checkpoint state snapshots",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter/framework config at time of trace",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Null-object sentinel
# ---------------------------------------------------------------------------


class _NullStratix:
    """
    Null-object sentinel used when an adapter is constructed without an
    STRATIX instance. Silently discards all calls so adapters can still
    be used stand-alone or in tests.

    Evaluates to falsy so ``if self._stratix:`` guards work correctly.
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
    def framework(self) -> str | None:
        return None

    @property
    def is_policy_violated(self) -> bool:
        return False


_NULL_STRATIX = _NullStratix()


# ---------------------------------------------------------------------------
# Circuit breaker constants
# ---------------------------------------------------------------------------

_CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive errors before opening
_CIRCUIT_BREAKER_COOLDOWN_S = 60.0  # seconds before attempting recovery


# ---------------------------------------------------------------------------
# BaseAdapter ABC
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """
    Abstract base class for all STRATIX framework adapters.

    Provides:
    - Circuit-breaker-protected ``emit_event()``
    - CaptureConfig filtering
    - Lifecycle management (connect / disconnect / health_check)
    - Replay serialization hook
    """

    # Subclasses MUST set these
    FRAMEWORK: str = ""
    VERSION: str = "0.0.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        event_sinks: list | None = None,
    ) -> None:
        self._stratix = stratix or _NULL_STRATIX
        self._capture_config = capture_config or CaptureConfig()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

        # Circuit breaker state (protected by _lock)
        self._lock = threading.Lock()
        self._error_count = 0
        self._circuit_open = False
        self._circuit_opened_at: float = 0.0

        # Collected events for replay serialization
        self._trace_events: list[dict[str, Any]] = []

        # Pluggable event sinks for persistence / export
        self._event_sinks: list = list(event_sinks) if event_sinks else []

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
        """True when a real (non-null) STRATIX instance is attached."""
        return bool(self._stratix)

    # --- Abstract lifecycle methods ---

    @abstractmethod
    def connect(self) -> None:
        """
        Verify framework availability and prepare the adapter.

        Implementations should import the framework, validate the version,
        and set ``self._connected = True`` / ``self._status = AdapterStatus.HEALTHY``.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        Flush pending events and release resources.

        Implementations should set ``self._connected = False`` and
        ``self._status = AdapterStatus.DISCONNECTED``.
        """
        ...

    @abstractmethod
    def health_check(self) -> AdapterHealth:
        """Return a health snapshot."""
        ...

    @abstractmethod
    def get_adapter_info(self) -> AdapterInfo:
        """Return metadata about this adapter."""
        ...

    @abstractmethod
    def serialize_for_replay(self) -> ReplayableTrace:
        """
        Serialize the current trace data for replay.

        Returns a ``ReplayableTrace`` that can be stored and later
        fed into the replay controller.
        """
        ...

    # --- Replay execution hook ---

    async def execute_replay(
        self,
        inputs: dict[str, Any],
        original_trace: Any,
        request: Any,
        replay_trace_id: str,
    ) -> Any:
        """
        Re-execute through this adapter's framework.

        Subclasses override this to provide actual re-execution.
        Default raises NotImplementedError (synthetic replay used instead).

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support execute_replay()"
        )

    # --- Concrete event emission ---

    def emit_event(
        self,
        payload: Any,
        privacy_level: Any | None = None,
    ) -> None:
        """
        Emit a typed event payload through the STRATIX pipeline.

        This method:
        1. Checks the circuit breaker — drops events if open (unless cooldown expired).
        2. Checks CaptureConfig — silently drops events whose layer is disabled
           (cross-cutting events are never dropped).
        3. Delegates to ``self._stratix.emit(payload, privacy_level)``
           with error counting for circuit-breaker state management.

        Args:
            payload: A Pydantic event payload (e.g., ToolCallEvent.create(...))
            privacy_level: Optional PrivacyLevel override
        """
        event_type = getattr(payload, "event_type", None)

        if not self._pre_emit_check(event_type):
            return

        # Delegate to STRATIX
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
        payload: dict[str, Any],
    ) -> None:
        """
        Emit a dict-based event through the STRATIX pipeline.

        Provides the same circuit-breaker and CaptureConfig gating as
        :meth:`emit_event` but accepts raw ``(event_type, dict)`` pairs
        used by the legacy adapter emission path.  This avoids bypassing
        the BaseAdapter protections.

        Args:
            event_type: Event type string (e.g., ``"model.invoke"``).
            payload: Raw event payload dict.
        """
        if not self._pre_emit_check(event_type):
            return

        try:
            self._stratix.emit(event_type, payload)
            self._post_emit_success(event_type, payload)
        except Exception:
            self._post_emit_failure()

    # --- Circuit breaker internals ---

    def _pre_emit_check(self, event_type: str | None) -> bool:
        """Run circuit-breaker and CaptureConfig checks. Return True to proceed."""
        with self._lock:
            if self._circuit_open:
                if not self._attempt_recovery():
                    return False

        if event_type and event_type not in ALWAYS_ENABLED_EVENT_TYPES:
            if not self._capture_config.is_layer_enabled(event_type):
                return False

        return True

    def _post_emit_success(self, event_type: str | None, payload: Any) -> None:
        """Handle successful emission: reset errors, record for replay."""
        with self._lock:
            if self._error_count > 0:
                self._error_count = 0
                if self._status == AdapterStatus.DEGRADED:
                    self._status = AdapterStatus.HEALTHY

        if event_type:
            try:
                payload_data = (
                    payload.model_dump()
                    if hasattr(payload, "model_dump")
                    else payload if isinstance(payload, dict) else str(payload)
                )
            except Exception:
                payload_data = str(payload)
            timestamp_ns = time.time_ns()
            self._trace_events.append({
                "event_type": event_type,
                "payload": payload_data,
                "timestamp_ns": timestamp_ns,
            })

            # Dispatch to pluggable event sinks
            if self._event_sinks:
                for sink in self._event_sinks:
                    try:
                        sink.send(event_type, payload_data, timestamp_ns)
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
                "Adapter %s emit error #%d", self.FRAMEWORK, self._error_count,
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
        """
        Check if the circuit-breaker cooldown has elapsed and attempt
        recovery.  Caller MUST hold ``self._lock``.

        Returns:
            True if the circuit is now closed (ready to emit).
            False if still open.
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
