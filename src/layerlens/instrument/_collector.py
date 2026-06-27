from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any, Dict, List, Callable, Optional

from layerlens.attestation import HashChain

from ._upload import enqueue_upload
from ._secret_scrub import scrub_payload, scrub_secrets
from ._capture_config import CaptureConfig

log: logging.Logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Coerce *value* to JSON-native types (non-native -> str), matching the
    upload path's ``json.dump(default=str)``. The emit() fallback (F-L1-003) so a
    Decimal/bytes/etc. payload value cannot raise out of the attestation hash and
    crash the host app — instrumentation must never crash the app it observes."""
    import json

    return json.loads(json.dumps(value, default=str))


# Attestation fail-CLOSED quarantine sink (A9 / R8 / LAY-3628). A trace whose
# hash chain can't be built (or was terminate()-d by a safety-stop) must NOT be
# uploaded as if attested. flush() routes it here instead of enqueue_upload.
# Default None => the trace is dropped from the normal upload path and logged at
# ERROR (never silently presented as attested). An integrator can set a sink
# (set_quarantine_sink) to persist quarantined payloads to a separate
# inspection/quarantine destination — quarantine preserves the events, it is not
# a silent drop.
_quarantine_sink: Optional[Callable[[Dict[str, Any]], None]] = None


def set_quarantine_sink(sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Route quarantined (unattested) trace payloads to *sink* instead of dropping."""
    global _quarantine_sink
    _quarantine_sink = sink


# Collector-seam trace observer (A4 / R5 / LAY-3627). flush() calls this with the
# FINAL trace payload at the REAL upload boundary, BEFORE enqueue_upload — the one
# synchronous point every uploaded trace passes, independent of the load-bearing
# _sync_mode and of whether a suite uses the capture_trace helper. Default None
# (zero prod overhead). The test harness installs an observer that runs the
# schema-lock + secret-scan over the real uploaded events, so EVERY flushing suite
# is validated population-completely (not just the capture-helper suites).
_trace_observer: Optional[Callable[[Dict[str, Any]], None]] = None


def set_trace_observer(observer: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install a test-only observer of the final uploaded trace payload (A4 seam)."""
    global _trace_observer
    _trace_observer = observer


def _is_attested(payload: Dict[str, Any]) -> bool:
    """True iff the trace carries a verifiable attestation (a root_hash and no
    build error). The fail-closed gate for upload."""
    attestation = payload.get("attestation") or {}
    return "attestation_error" not in attestation and attestation.get("root_hash") is not None


def _quarantine(payload: Dict[str, Any]) -> None:
    """Fail closed: log + route to the quarantine sink; never upload as attested."""
    attestation = payload.get("attestation") or {}
    log.error(
        "layerlens: trace %s quarantined — unattested (%s); NOT uploaded",
        payload.get("trace_id"),
        attestation.get("attestation_error") or attestation.get("terminated_reason") or "no root_hash",
    )
    if _quarantine_sink is not None:
        try:
            _quarantine_sink(payload)
        except Exception:  # a broken sink must not crash the emit thread
            log.warning("layerlens: quarantine sink raised", exc_info=True)


class TraceCollector:
    """Collects flat events for a single trace, with CaptureConfig gating and attestation.

    Thread-safe: all mutations go through ``self._lock``.
    Once ``flush()`` is called the collector is sealed — further ``emit()`` calls are no-ops.
    """

    MAX_EVENTS = 10_000

    def __init__(self, client: Any, config: CaptureConfig) -> None:
        self._client = client
        self._config = config
        self._trace_id = uuid.uuid4().hex[:16]
        self._events: List[Dict[str, Any]] = []
        self._sequence: int = 0
        self._chain = HashChain()
        self._capped = False
        self._sealed = False
        self._lock = threading.Lock()

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def sealed(self) -> bool:
        """True once :meth:`flush` has run — further ``emit`` calls are no-ops."""
        return self._sealed

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
        # Credential-sprawl chokepoint: scrub secrets from free-text error fields
        # on EVERY event, regardless of capture_content (provider exceptions echo
        # API keys and are uploaded even under the default config). One place
        # covers every adapter's str(exc) site (LAY-3567 P2).
        payload = scrub_payload(payload)

        # Centralized price-on-emit chokepoint (A1 / A11 / LAY-3626): the single
        # place every cost.record passes. FILL-WHEN-ABSENT — if a priced model's
        # record arrives without a cost_usd (an emit path forgot the hook), price
        # it here from its own payload (model + usage + service_tier + provider),
        # so no path can ship a tokens-only record. Adapter-computed costs
        # (langfuse vendor cost, bedrock_agents, the _fire helpers) are PRESERVED
        # — we only fill, never clobber. Tier (A2) / cache-write (A3) pricing
        # flows through the shared calculate_cost both here and at the sites.
        # Lazy import avoids an adapters-package init cycle.
        if event_type == "cost.record" and payload.get("cost_usd") is None:
            from .adapters.providers.pricing import price_cost_record

            priced = price_cost_record(payload)
            if priced is not None:
                payload = {**payload, "cost_usd": priced}

        # Scrub secrets from the span_name envelope field too (F-L5-001): redact_payload
        # and scrub_payload only touch ``payload``, so a credential templated into a
        # span_name (a tool/agent label) would otherwise ship cleartext even under
        # capture_content=False. Secrets are orthogonal to capture_content.
        if span_name:
            span_name = scrub_secrets(span_name)

        with self._lock:
            if self._sealed:
                return

            if len(self._events) >= self.MAX_EVENTS:
                if not self._capped:
                    self._capped = True
                    log.warning(
                        "layerlens: trace %s hit %d event limit, further events dropped",
                        self._trace_id,
                        self.MAX_EVENTS,
                    )
                return

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
            try:
                self._chain.add_event(event)
            except Exception:
                # F-L1-003: a non-JSON-native payload value (Decimal/bytes/...) must
                # not raise out of the attestation hash and crash the host app.
                # Coerce to JSON-safe (matching the upload path) and retry; if it is
                # still unhashable, drop the single event (rolling back the sequence
                # bump) with a warning rather than propagate.
                event["payload"] = _json_safe(event["payload"])
                try:
                    self._chain.add_event(event)
                except Exception:
                    self._sequence -= 1
                    log.warning("layerlens: dropping unhashable event %s", event_type, exc_info=True)
                    return
            self._events.append(event)

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Read-only snapshot of the events captured so far."""
        with self._lock:
            return list(self._events)

    def to_replay_dict(self) -> Dict[str, Any]:
        """Return the trace as a replay-ready dict.

        Same shape as the payload uploaded to the API: ``trace_id``,
        ``events``, ``capture_config``, ``attestation``. Safe to call at
        any time — even before flush — and idempotent (does not seal the
        collector or the hash chain). Use this to persist a trace for
        later replay via :mod:`layerlens.replay.snapshot`.
        """
        with self._lock:
            return self._build_trace_payload(seal=False)

    def _build_trace_payload(self, *, seal: bool = True) -> Dict[str, Any]:
        """Build the attestation envelope and trace payload.

        When ``seal`` is True (default, used by :meth:`flush`) the hash
        chain is finalized — no more events can be added. When False
        (used by :meth:`to_replay_dict`) the root hash is computed
        non-destructively so the collector stays usable.
        """
        try:
            if seal:
                trial = self._chain.finalize()
                root_hash: Optional[str] = trial.hash
            else:
                # Non-destructive: compute root_hash without finalizing.
                envelopes = self._chain.envelopes
                if envelopes:
                    from layerlens.attestation._hash import compute_hash

                    event_hashes = [e.hash for e in envelopes]
                    root_hash = compute_hash({"event_hashes": event_hashes})
                else:
                    root_hash = None
            attestation: Dict[str, Any] = {
                "chain": self._chain.to_dict(),
                "schema_version": "1.0",
            }
            if root_hash is not None:
                attestation["root_hash"] = root_hash
        except Exception as exc:
            log.warning("Failed to build attestation chain", exc_info=True)
            attestation = {"attestation_error": str(exc)}

        trace_payload: Dict[str, Any] = {
            "trace_id": self._trace_id,
            "events": list(self._events) if not seal else self._events,
            "capture_config": self._config.to_dict(),
            "attestation": attestation,
        }
        if self._capped:
            trace_payload["truncated"] = True
            trace_payload["max_events"] = self.MAX_EVENTS
        return trace_payload

    def terminate(self, reason: str) -> None:
        """Permanently mark this trace non-attestable (a safety-stop / policy
        halt). The next flush() fails CLOSED: the chain can no longer be
        finalized, so the trace is quarantined rather than uploaded (A9)."""
        with self._lock:
            self._chain.terminate(reason)

    def flush(self) -> None:
        """Seal the collector and either UPLOAD a well-attested trace or
        QUARANTINE one whose attestation could not be built (fail-closed, A9)."""
        with self._lock:
            if self._sealed or not self._events:
                return
            self._sealed = True
            payload = self._build_trace_payload()
        # Collector seam (A4): observe the FINAL payload at the real upload
        # boundary, before enqueue/quarantine routing, so the test nets see every
        # flushing trace regardless of the capture-helper / _sync_mode.
        if _trace_observer is not None:
            _trace_observer(payload)
        if _is_attested(payload):
            enqueue_upload(self._client, payload)
        else:
            _quarantine(payload)
