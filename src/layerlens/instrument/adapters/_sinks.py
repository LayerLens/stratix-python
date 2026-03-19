"""
STRATIX Event Sinks

Pluggable sinks that receive events from BaseAdapter after successful emission.
SDK-side sinks that bridge the adapter's in-memory event stream to the
LayerLens platform API or to local logging for development.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventSink(ABC):
    """
    Abstract base for event sinks.

    Sinks receive ``(event_type, payload, timestamp_ns)`` triples from
    ``BaseAdapter._post_emit_success`` and persist or forward them.
    """

    @abstractmethod
    def send(self, event_type: str, payload: dict[str, Any], timestamp_ns: int) -> None:
        """
        Accept a single event.

        Args:
            event_type: Event type string (e.g. ``"model.invoke"``).
            payload: Serialized event payload (dict or str).
            timestamp_ns: Nanosecond-precision Unix timestamp.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered events to the backend."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Finalize the sink (e.g. mark trace as completed)."""
        ...


class APIUploadSink(EventSink):
    """
    Sink that buffers events and uploads them as JSONL via the LayerLens
    ``Stratix.traces.upload()`` API method on flush/close.

    This is the bridge between ``layerlens.instrument`` (capture) and
    ``layerlens.Stratix`` (platform).

    Args:
        client: A ``layerlens.Stratix`` (or ``layerlens.Client``) instance.
        trace_id: Optional trace ID; auto-generated if not provided.
        agent_id: Optional agent identifier for the trace.
        metadata: Optional metadata dict attached to the trace.
        buffer_size: Number of events to buffer before auto-flushing.
            Defaults to 1000. Set to 0 to disable auto-flush.
    """

    def __init__(
        self,
        client: Any,
        trace_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        buffer_size: int = 1000,
    ) -> None:
        self._client = client
        self._trace_id = trace_id or str(uuid.uuid4())
        self._agent_id = agent_id
        self._metadata = metadata or {}
        self._buffer_size = buffer_size
        self._buffer: list[dict[str, Any]] = []
        self._sequence_id = 0
        self._closed = False
        self._start_time = datetime.now(timezone.utc)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def send(self, event_type: str, payload: dict[str, Any], timestamp_ns: int) -> None:
        if self._closed:
            return

        self._sequence_id += 1
        ts = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)

        record = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "trace_id": self._trace_id,
            "span_id": str(uuid.uuid4()),
            "sequence_id": self._sequence_id,
            "timestamp": ts.isoformat(),
            "payload": payload if isinstance(payload, dict) else {"raw": str(payload)},
        }
        if self._agent_id:
            record["agent_id"] = self._agent_id

        self._buffer.append(record)

        if self._buffer_size > 0 and len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        try:
            # Write buffered events as JSONL to a temp file and upload
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, prefix="stratix_trace_"
            )
            try:
                for event in self._buffer:
                    tmp.write(json.dumps(event, default=str) + "\n")
                tmp.close()

                # Upload via the client's traces resource
                self._client.traces.upload(
                    file=Path(tmp.name),
                    trace_id=self._trace_id,
                    metadata=self._metadata,
                )
            finally:
                Path(tmp.name).unlink(missing_ok=True)

            self._buffer.clear()
        except Exception:
            logger.debug(
                "APIUploadSink.flush() failed for %d events on trace %s",
                len(self._buffer),
                self._trace_id,
                exc_info=True,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()


class LoggingSink(EventSink):
    """
    Sink that logs events via Python's logging module.

    Useful for development, debugging, and local testing.

    Args:
        logger_name: Logger name. Defaults to ``"layerlens.instrument.events"``.
        level: Logging level. Defaults to ``logging.DEBUG``.
    """

    def __init__(
        self,
        logger_name: str = "layerlens.instrument.events",
        level: int = logging.DEBUG,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._closed = False

    def send(self, event_type: str, payload: dict[str, Any], timestamp_ns: int) -> None:
        if self._closed:
            return

        ts = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
        self._logger.log(
            self._level,
            "[%s] %s: %s",
            ts.isoformat(),
            event_type,
            json.dumps(payload, default=str)[:500],
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
