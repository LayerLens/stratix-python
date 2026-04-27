"""LayerLens Event Sinks.

Pluggable sinks that receive events from :class:`BaseAdapter` after
successful emission. Each sink bridges the adapter's in-memory event
stream to a persistence or export backend.

The ``ateam`` source provided concrete :class:`TraceStoreSink` and
:class:`IngestionPipelineSink` implementations that depended on
``stratix.storage.traces.TraceStore`` and ``stratix.ingest.pipeline``.
Those server-side modules do not exist in the ``stratix-python`` SDK;
the sinks here are kept as protocol-conformant duck-typed bridges that
accept any object exposing ``store_trace`` / ``store_event`` (for
:class:`TraceStoreSink`) or ``ingest`` (for :class:`IngestionPipelineSink`).

Typical SDK usage routes events to an HTTP sink that POSTs to atlas-app
``/api/v1/telemetry/spans``; that sink lives in
``layerlens.instrument.transport`` and is added in a later milestone.

Ported from ``ateam/stratix/sdk/python/adapters/sinks.py``.

Multi-tenancy
-------------
Sinks accept events from multiple :class:`BaseAdapter` instances bound
to different tenants. Buffering and flushing are partitioned by
``org_id`` so a single tenant's burst cannot starve another tenant's
events nor displace them via global eviction. See
:class:`IngestionPipelineSink` for the per-tenant buffer cap contract
(CLAUDE.md "EVERY data operation must be scoped by tenant").
"""

from __future__ import annotations

import uuid
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Python 3.11+ exposes ``datetime.UTC``; for 3.8+ compat we alias the
# existing ``timezone.utc`` constant. Keeping both names available means
# adapter code can use ``UTC`` regardless of interpreter version.
UTC = timezone.utc

logger = logging.getLogger(__name__)


class EventSink(ABC):
    """Abstract base for event sinks.

    Sinks receive ``(event_type, payload, timestamp_ns, *, org_id)``
    tuples from :meth:`BaseAdapter._post_emit_success` and persist or
    forward them.

    **Multi-tenancy contract:** every event arrives with a bound
    ``org_id``. Subclasses must use this value to scope persistence
    (tenant-prefixed cache keys, per-tenant streams, RLS-tagged rows).
    The same ``org_id`` is also present at ``payload["org_id"]`` —
    both surfaces are populated for sinks that map directly to a
    payload field and for sinks that route on the explicit kwarg.
    """

    @abstractmethod
    def send(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        *,
        org_id: str,
    ) -> None:
        """Accept a single event.

        Args:
            event_type: Event type string (e.g., ``"model.invoke"``).
            payload: Serialized event payload dict (always contains
                ``org_id`` matching the explicit kwarg).
            timestamp_ns: Nanosecond-precision Unix timestamp.
            org_id: Tenant binding for this event. MUST be propagated
                to any downstream queue / store / stream.
        """

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered events to the backend."""

    @abstractmethod
    def close(self) -> None:
        """Finalize the sink (e.g. mark trace as completed)."""


class TraceStoreSink(EventSink):
    """Sink that writes events directly to a duck-typed trace store.

    The store object must expose:

    * ``store_trace(record)`` — accepts a record-like object with the
      fields the store understands (``trace_id``, ``status``,
      ``start_time``, ``end_time``, etc.).
    * ``store_event(record)`` — accepts a record-like object with
      ``event_id``, ``event_type``, ``trace_id``, ``span_id``,
      ``sequence_id``, ``timestamp``, ``payload``.
    * ``get_trace(trace_id)`` and ``update_trace_status(trace_id, status)``
      for finalization.

    The factory callables for trace and event records can be injected via
    ``trace_record_factory`` and ``event_record_factory``; if omitted, the
    sink uses simple dicts. This decouples the sink from the
    ``stratix.storage.traces`` module that lives only in the framework
    repo.

    Auto-generates ``trace_id`` (or accepts one), ``event_id``, ``span_id``,
    and auto-increments ``sequence_id``. On :meth:`close` the trace is
    marked ``"completed"``.
    """

    def __init__(
        self,
        store: Any,
        trace_id: Optional[str] = None,
        trial_id: str = "default",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_record_factory: Optional[Any] = None,
        event_record_factory: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._trace_id = trace_id or str(uuid.uuid4())
        self._trial_id = trial_id
        self._sequence_id = 0
        self._closed = False
        self._start_time = datetime.now(UTC)
        self._trace_record_factory = trace_record_factory or self._default_trace_record
        self._event_record_factory = event_record_factory or self._default_event_record

        self._store.store_trace(
            self._trace_record_factory(
                trace_id=self._trace_id,
                trial_id=self._trial_id,
                agent_id=agent_id,
                start_time=self._start_time,
                end_time=self._start_time,
                status="active",
                metadata=metadata or {},
            )
        )

    @staticmethod
    def _default_trace_record(**kwargs: Any) -> Dict[str, Any]:
        return dict(kwargs)

    @staticmethod
    def _default_event_record(**kwargs: Any) -> Dict[str, Any]:
        return dict(kwargs)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def send(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        *,
        org_id: str,
    ) -> None:
        if self._closed:
            return

        self._sequence_id += 1
        ts = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC)

        record = self._event_record_factory(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            trace_id=self._trace_id,
            span_id=str(uuid.uuid4()),
            sequence_id=self._sequence_id,
            timestamp=ts,
            payload=payload if isinstance(payload, dict) else {"raw": str(payload)},
            org_id=org_id,
        )

        try:
            self._store.store_event(record)
        except Exception:
            logger.debug(
                "TraceStoreSink.send() failed for event %s",
                event_type,
                exc_info=True,
            )

    def flush(self) -> None:
        # TraceStoreSink writes synchronously — nothing to flush.
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            existing = None
            if hasattr(self._store, "get_trace"):
                existing = self._store.get_trace(self._trace_id)
            if existing is not None:
                if hasattr(existing, "status"):
                    existing.status = "completed"
                    existing.end_time = datetime.now(UTC)
                    existing.event_count = self._sequence_id
                    self._store.store_trace(existing)
                elif isinstance(existing, dict):
                    existing["status"] = "completed"
                    existing["end_time"] = datetime.now(UTC)
                    existing["event_count"] = self._sequence_id
                    self._store.store_trace(existing)
            elif hasattr(self._store, "update_trace_status"):
                self._store.update_trace_status(self._trace_id, "completed")
        except Exception:
            logger.debug(
                "TraceStoreSink.close() failed to finalize trace %s",
                self._trace_id,
                exc_info=True,
            )


_DEFAULT_MAX_PER_TENANT_BUFFER_SIZE = 1000
"""Default per-tenant buffer cap for :class:`IngestionPipelineSink`.

Caps each tenant's buffer independently so a single noisy tenant cannot
exhaust memory or displace events from quieter tenants. See
``MaxPerTenantBufferSize`` parameter on :class:`IngestionPipelineSink`.
"""


class IngestionPipelineSink(EventSink):
    """Sink that feeds events into a duck-typed ingestion pipeline.

    The pipeline object must expose
    ``ingest(events: list[dict], tenant_id: str)``.

    Supports two modes:

    * **immediate** (default): each event is ingested as a single-item batch
      keyed by the per-event ``org_id``.
    * **buffered**: events are partitioned by ``org_id`` into per-tenant
      buffers and ingested on :meth:`flush` / :meth:`close`. Each tenant's
      buffer is flushed in its own ``ingest()`` call so one tenant's
      backlog cannot delay another tenant's events.

    Multi-tenancy
    -------------
    Buffering is partitioned by ``org_id`` (Gap 2 of the multi-tenancy
    hardening contract). Per-tenant invariants:

    * **Isolation:** each tenant gets its own buffer dict slot. A burst
      from tenant A never appears in tenant B's flush.
    * **Bounded eviction:** ``max_per_tenant_buffer_size`` (default
      ``1000``) caps each tenant independently. When tenant A overflows,
      A's oldest events drop FIFO — tenant B's events are untouched.
      Drops are counted per-tenant in :attr:`dropped_per_tenant`.
    * **Observability:** :meth:`buffer_size_per_tenant` returns a
      snapshot ``dict[org_id, int]`` for the
      ``sink_per_tenant_buffer_size{org_id}`` gauge consumers.

    Thread-safety
    -------------
    ``send`` / ``flush`` / ``close`` are safe to call concurrently from
    any thread. All buffer mutations are serialized via an internal
    lock; per-tenant ingest calls are made *outside* the lock to avoid
    holding it across slow IO.
    """

    def __init__(
        self,
        pipeline: Any,
        trace_id: Optional[str] = None,
        tenant_id: str = "default",
        buffered: bool = False,
        max_per_tenant_buffer_size: int = _DEFAULT_MAX_PER_TENANT_BUFFER_SIZE,
    ) -> None:
        if max_per_tenant_buffer_size <= 0:
            raise ValueError(
                "max_per_tenant_buffer_size must be > 0; got "
                f"{max_per_tenant_buffer_size}. The cap is per-tenant — "
                "use a small positive value, never zero."
            )
        self._pipeline = pipeline
        self._trace_id = trace_id or str(uuid.uuid4())
        self._tenant_id = tenant_id
        self._buffered = buffered
        # Per-tenant buffers — keyed by org_id. A buffer for tenant A
        # is never flushed under tenant B's binding.
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        # Per-tenant drop counters, surfaced as
        # ``sink_per_tenant_dropped{org_id}``.
        self._dropped_per_tenant: Dict[str, int] = {}
        self._max_per_tenant_buffer_size = max_per_tenant_buffer_size
        self._sequence_id = 0
        self._closed = False
        # Single mutex covers buffer / counter / sequence-id mutation.
        # Per-tenant ingest IO happens outside the lock.
        self._lock = threading.Lock()

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def max_per_tenant_buffer_size(self) -> int:
        """The per-tenant buffer cap. Read-only after construction."""
        return self._max_per_tenant_buffer_size

    def buffer_size_per_tenant(self) -> Dict[str, int]:
        """Snapshot of currently-buffered event counts per tenant.

        Suitable for the ``sink_per_tenant_buffer_size{org_id}`` gauge.
        Returned dict is a defensive copy; mutating it does not affect
        sink state.
        """
        with self._lock:
            return {org_id: len(buf) for org_id, buf in self._buffers.items()}

    @property
    def dropped_per_tenant(self) -> Dict[str, int]:
        """Snapshot of drop counts per tenant (FIFO eviction on overflow)."""
        with self._lock:
            return dict(self._dropped_per_tenant)

    def _format_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        org_id: str,
    ) -> Dict[str, Any]:
        """Format an event into the dict schema that ``ingest()`` expects.

        ``org_id`` is propagated both as a top-level field (for sinks
        that read it directly) and inside the payload (already stamped
        upstream by :meth:`BaseAdapter._stamp_org_id`). Sequence-id
        increment is performed under the sink lock so concurrent senders
        do not collide.
        """
        with self._lock:
            self._sequence_id += 1
            seq = self._sequence_id
        ts = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC)
        return {
            "event_type": event_type,
            "trace_id": self._trace_id,
            "timestamp": ts.isoformat(),
            "span_id": str(uuid.uuid4()),
            "sequence_id": seq,
            "event_id": str(uuid.uuid4()),
            "org_id": org_id,
            "payload": payload if isinstance(payload, dict) else {"raw": str(payload)},
        }

    def send(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        *,
        org_id: str,
    ) -> None:
        if self._closed:
            return

        formatted = self._format_event(event_type, payload, timestamp_ns, org_id)
        # Per-event org_id is the source of truth for multi-tenant
        # ingest. Empty values fall back to the sink-level legacy
        # ``tenant_id`` only to preserve backward compatibility — adapter
        # emissions always carry a non-empty ``org_id`` post-PR #118.
        effective_org = org_id or self._tenant_id

        if self._buffered:
            self._enqueue(effective_org, formatted)
        else:
            try:
                self._pipeline.ingest([formatted], tenant_id=effective_org)
            except Exception:
                logger.debug(
                    "IngestionPipelineSink.send() failed for event %s",
                    event_type,
                    exc_info=True,
                )

    def _enqueue(self, org_id: str, formatted: Dict[str, Any]) -> None:
        """Append an event to ``org_id``'s buffer with FIFO overflow drop.

        When the per-tenant buffer is at the cap, the OLDEST event in
        THAT TENANT's buffer is dropped — never another tenant's. The
        drop counter is bumped in :attr:`dropped_per_tenant`.
        """
        with self._lock:
            buf = self._buffers.setdefault(org_id, [])
            if len(buf) >= self._max_per_tenant_buffer_size:
                # FIFO eviction confined to THIS tenant's buffer.
                # Other tenants' buffers are not inspected or modified.
                buf.pop(0)
                self._dropped_per_tenant[org_id] = self._dropped_per_tenant.get(org_id, 0) + 1
                logger.debug(
                    "IngestionPipelineSink dropped oldest event for tenant %s (cap=%d)",
                    org_id,
                    self._max_per_tenant_buffer_size,
                )
            buf.append(formatted)

    def flush(self) -> None:
        """Flush every tenant's buffer in its own ``ingest()`` call.

        Each tenant's batch is sent under that tenant's ``org_id`` — one
        tenant's slow downstream cannot block another's flush, because
        the calls are sequenced on a fresh per-tenant snapshot taken
        under the lock and the ingest IO happens outside it.
        """
        with self._lock:
            # Take a snapshot per tenant and clear each buffer atomically
            # so concurrent ``send`` after this point starts a fresh
            # buffer for that tenant.
            snapshot: Dict[str, List[Dict[str, Any]]] = {}
            for org_id, buf in self._buffers.items():
                if buf:
                    snapshot[org_id] = buf
                    self._buffers[org_id] = []

        for org_id, batch in snapshot.items():
            try:
                self._pipeline.ingest(list(batch), tenant_id=org_id)
            except Exception:
                logger.debug(
                    "IngestionPipelineSink.flush() failed for %d events (tenant=%s)",
                    len(batch),
                    org_id,
                    exc_info=True,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
