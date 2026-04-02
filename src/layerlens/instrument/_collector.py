from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from layerlens.attestation import HashChain

from ._capture_config import CaptureConfig
from ._upload import upload_trace, async_upload_trace

log: logging.Logger = logging.getLogger(__name__)


class TraceCollector:
    """Collects flat events for a single trace, with CaptureConfig gating and attestation."""

    def __init__(self, client: Any, config: CaptureConfig) -> None:
        self._client = client
        self._config = config
        self._trace_id = uuid.uuid4().hex[:16]
        self._events: List[Dict[str, Any]] = []
        self._sequence: int = 0
        self._chain = HashChain()

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def config(self) -> CaptureConfig:
        return self._config

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: str,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        """Emit an event. Checks CaptureConfig, strips content if needed, hashes, appends."""
        if not self._config.is_layer_enabled(event_type):
            return

        payload = self._config.redact_payload(event_type, payload)

        self._sequence += 1
        event: Dict[str, Any] = {
            "event_type": event_type,
            "trace_id": self._trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "span_name": span_name,
            "sequence_id": self._sequence,
            "timestamp_ns": time.time_ns(),
            "payload": payload,
        }
        self._chain.add_event(event)
        self._events.append(event)

    def _build_trace_payload(self) -> Dict[str, Any]:
        """Build the attestation envelope and trace payload."""
        try:
            trial = self._chain.finalize()
            attestation: Dict[str, Any] = {
                "chain": self._chain.to_dict(),
                "root_hash": trial.hash,
                "schema_version": "1.0",
            }
        except Exception as exc:
            log.warning("Failed to build attestation chain", exc_info=True)
            attestation = {"attestation_error": str(exc)}

        return {
            "trace_id": self._trace_id,
            "events": self._events,
            "capture_config": self._config.to_dict(),
            "attestation": attestation,
        }

    def flush(self) -> None:
        """Build attestation and upload the trace."""
        if not self._events:
            return
        upload_trace(self._client, self._build_trace_payload())

    async def async_flush(self) -> None:
        """Async version of flush."""
        if not self._events:
            return
        await async_upload_trace(self._client, self._build_trace_payload())
