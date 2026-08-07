from __future__ import annotations

import json
import time
import uuid
import logging
import threading
from typing import Any, Dict, List, Callable, Optional

from layerlens.attestation import HashChain

from ._events import TRACE_ROOT, AGENT_IDENTITY
from ._upload import enqueue_upload
from ._identity import honest_agent_identity
from ._secret_scrub import scrub_payload, scrub_secrets
from ._capture_config import CaptureConfig

log: logging.Logger = logging.getLogger(__name__)

#: ``cost.record.cost_status`` marker (LAY-3622 / A4b): the model resolves to a
#: rate, but this payload carries no token dimension the pricing formula can read
#: (a totals-only usage), so the cost is UNKNOWABLE rather than zero. Distinguishes
#: a legitimately-absent cost from the A11 dropped-price bug, which is also an
#: absent ``cost_usd`` on a priced model. Never accompanied by a ``cost_usd``.
UNPRICEABLE_TOKEN_SHAPE = "unpriceable_token_shape"

#: ``cost.record.cost_status`` marker (LAY-3622 / F4): a cost WAS computed, but the
#: provider reported more billed tokens than any rate was applied to, so the figure
#: UNDERSTATES the bill. The canonical case is Gemini, which reports
#: ``thoughtsTokenCount`` outside ``candidatesTokenCount`` while ``total_token_count``
#: includes it — thinking tokens bill at the output rate and we priced none of them.
#:
#: Unlike :data:`UNPRICEABLE_TOKEN_SHAPE` this marker ALWAYS accompanies a
#: ``cost_usd``, and it deliberately changes no money: inventing a rate for tokens we
#: cannot attribute would be a guess billed to a customer, whereas an
#: under-report we have recorded is a fact someone can act on. The magnitude rides
#: along as ``unpriced_tokens`` so a reader need not re-derive the pricing
#: arithmetic (which has a subtlety — see ``pricing.priced_token_count``).
PARTIAL_TOKEN_SHAPE = "partial_token_shape"


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
    # Per-event payload byte cap (F-L12-003). A single pathological event (e.g. a
    # multi-MB string) must not bloat the in-memory trace or the upload unbounded;
    # the count cap (MAX_EVENTS) alone doesn't bound size.
    MAX_EVENT_BYTES = 256 * 1024

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
            from .adapters.providers.pricing import is_priced, price_cost_record

            priced = price_cost_record(payload)
            if priced is not None:
                payload = {**payload, "cost_usd": priced}
            elif is_priced(payload.get("model"), payload.get("provider")):
                # FILL-WHEN-ABSENT could not fill (LAY-3622 / A4b): the model HAS a
                # rate, but this payload carries no dimension the formula can price
                # — a totals-only usage, since the formula reads prompt / cached /
                # cache-write / completion and never the total. It used to answer
                # 0.0 here (a sum over four zeroes) and ship it as a derived cost,
                # so a real billed call reached the customer as free.
                #
                # The cost is UNKNOWABLE from this payload, not zero. Say so
                # explicitly: a bare missing cost_usd on a priced model is the A11
                # dropped-price bug, and a fail-closed reader must be able to tell
                # the two apart. The token counts are preserved either way.
                payload = {**payload, "cost_status": UNPRICEABLE_TOKEN_SHAPE}

        # Detectable UNDER-report (LAY-3622 / F4): we DID price this record, but the
        # provider reported more billed tokens than any rate was applied to. Record
        # it; do not re-price it. Attributing the residual to a rate we did not
        # observe would be a guess billed to a customer — the opposite failure from
        # A4b but the same class (a number presented as derived when it is not).
        #
        # Only OUR arithmetic is checked. A vendor/gateway-reported charge
        # (``cost_source``: langfuse's billing figure, OpenRouter's usage accounting)
        # is a billed FACT, not an estimate of one, so a token gap against it says
        # nothing about its accuracy. An existing cost_status is never clobbered —
        # the unpriceable branch above is mutually exclusive with this one (it
        # requires no cost_usd; this requires one), so a marker already present came
        # from an adapter and is its own statement.
        if (
            event_type == "cost.record"
            and payload.get("cost_usd") is not None
            and payload.get("cost_status") is None
            and not payload.get("cost_source")
        ):
            from .adapters.providers.pricing import unpriced_token_count

            unpriced = unpriced_token_count(payload)
            if unpriced:
                payload = {
                    **payload,
                    "cost_status": PARTIAL_TOKEN_SHAPE,
                    "unpriced_tokens": unpriced,
                }

        # Scrub secrets from the span_name envelope field too (F-L5-001): redact_payload
        # and scrub_payload only touch ``payload``, so a credential templated into a
        # span_name (a tool/agent label) would otherwise ship cleartext even under
        # capture_content=False. Secrets are orthogonal to capture_content.
        if span_name:
            span_name = scrub_secrets(span_name)

        # Per-event byte cap (F-L12-003): replace an oversized payload with a small
        # marker so one pathological event can't bloat the trace/upload. Runs after
        # redaction/scrub/pricing so a normal (small) event is untouched.
        try:
            payload_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
        except Exception:
            payload_bytes = 0
        if payload_bytes > self.MAX_EVENT_BYTES:
            log.warning(
                "layerlens: event %s payload %d bytes exceeds cap %d; truncated",
                event_type,
                payload_bytes,
                self.MAX_EVENT_BYTES,
            )
            payload = {
                "_truncated": True,
                "_original_bytes": payload_bytes,
                "_cap_bytes": self.MAX_EVENT_BYTES,
                "_reason": "event payload exceeded MAX_EVENT_BYTES",
            }

        with self._lock:
            self._append_locked(event_type, payload, span_id, parent_span_id, span_name)

    def _append_locked(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: str,
        parent_span_id: Optional[str],
        span_name: Optional[str],
    ) -> None:
        """Append one already-gated/scrubbed event to the chain. Caller holds ``self._lock``.

        Split out of :meth:`emit` so the flush-time root synthesizer
        (:meth:`_synthesize_root_if_needed`) reuses the identical sequence-bump +
        attestation-hash + append path — one place builds the event dict, one
        place hashes it, so a synthesized root is chained exactly like any other
        event (attestation still verifies)."""
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
            envelope = self._chain.add_event(event)
        except Exception:
            # F-L1-003: a non-JSON-native payload value (Decimal/bytes/...) must
            # not raise out of the attestation hash and crash the host app.
            # Coerce to JSON-safe (matching the upload path) and retry; if it is
            # still unhashable, drop the single event (rolling back the sequence
            # bump) with a warning rather than propagate.
            event["payload"] = _json_safe(event["payload"])
            try:
                envelope = self._chain.add_event(event)
            except Exception:
                self._sequence -= 1
                log.warning("layerlens: dropping unhashable event %s", event_type, exc_info=True)
                return
        # Attach the per-event attestation hash onto the wire event so a consumer
        # (ateam) can verify per-event and record origin='sdk', rather than
        # relying on positional alignment with the parallel
        # ``attestation.chain.events[]`` array (still emitted, for back-compat).
        # Attached AFTER the chain has hashed the event so the hash never feeds
        # into its own digest: the recompute must strip ``hash``/``previous_hash``
        # and re-inject ``_previous_hash`` (the chain hashes ``{**event,
        # '_previous_hash': prev}``). Proven by test_attached_hash_recomputes.
        event["hash"] = envelope.hash
        event["previous_hash"] = envelope.previous_hash
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

    # The synthesized trace root is its OWN dedicated, registered event type
    # (``trace.root``) — NOT an ``agent.lifecycle`` event: it is a content-free
    # structural marker (no agent started), so reusing agent.lifecycle would be
    # semantically misleading and would pollute the real agent.lifecycle stream on
    # the ~55% of traces that need a synthesized root. See _events.TRACE_ROOT.
    _ROOT_EVENT_TYPE = TRACE_ROOT

    def _synthesize_root_if_needed(self) -> None:
        """Ensure the trace has a REAL, captured root span (companion to atlas-app
        PR #2042). Caller holds ``self._lock``; runs at flush before sealing.

        A provider-only / bare-adapter / ``trace_context`` / framework
        ``_begin_run`` trace emits its leaf events (``model.invoke`` etc.) parented
        to an AMBIENT span the SDK never emitted an event for — a "dangling parent".
        The frontend then has to synthesize a root. When the events reference
        EXACTLY ONE such dangling parent and no captured root already exists, emit
        ONE lightweight, content-free ``trace.root`` marker ON that dangling span
        so every leaf's parent resolves to a captured span and the tree has a real
        root.

        Deliberately conservative — it does NOT fire when:
          * there are no events (nothing to root);
          * a captured root already exists (the ``@trace`` decorator emits
            ``agent.input``/``agent.output`` on its root span — do not double-root
            the ~36.5% of already-clean traces);
          * there is more than one distinct dangling parent (genuine fragmentation,
            ~7.4%) — inventing a single wrapper would misrepresent the topology, so
            leave it for the FE's multi-root synthesis and ``fragmented`` flag.

        The marker is emitted regardless of the L1 layer toggle / ``capture_content``
        (it is structure, not content — the tree must always have a root) but
        carries NO agent name and NO content, so it never fabricates an agent or
        leaks PII. It is appended via :meth:`_append_locked`, so it flows through
        the same attestation hash chain and verifies like any other event."""
        if not self._events:
            return

        captured_span_ids = {e["span_id"] for e in self._events if e.get("span_id")}
        dangling: set[str] = set()
        for e in self._events:
            parent = e.get("parent_span_id")
            if parent is None or parent == e.get("span_id"):
                continue  # null / self-parent == a real root marker already
            if parent not in captured_span_ids:
                dangling.add(parent)

        # A captured root already exists (no orphaned leaf) -> nothing to do.
        # More than one distinct missing parent == genuine fragmentation -> leave
        # it to the FE's multi-root handling rather than inventing a false wrapper.
        if len(dangling) != 1:
            return

        root_span_id = next(iter(dangling))
        # Bypass is_layer_enabled + redact_payload (this is structural, content-free,
        # and must exist regardless of L1/capture_content); append straight through
        # the shared hashing path. Emitting the root LAST keeps sequence order
        # monotonic and never reorders the observed leaf events.
        self._append_locked(
            self._ROOT_EVENT_TYPE,
            {"synthesized": True},
            span_id=root_span_id,
            parent_span_id=None,
            span_name="trace",
        )

    def _synthesize_identity_if_needed(self) -> None:
        """Ensure a trace with a producer-DECLARED agent name carries ONE
        canonical ``agent.identity`` event. Caller holds ``self._lock``; runs at
        flush before sealing.

        The honest name a producer already declared (a @stratix.trace name, a
        crew/agent name, a langgraph node) lives scattered across per-adapter
        payload keys the server never reads. :func:`honest_agent_identity`
        resolves the ONE honest name (or None), and this appends a single
        structural marker so the server + FE surface the Agent column from one
        place. It REFUSES to synthesize from a model name, an API-method label, a
        span_name, or a class default — an honest "—" beats a fabricated name.

        Deliberately conservative — it does NOT fire when:
          * there is no honestly-declared name (provider-only / bare traces stay
            "—" — the identity is genuinely absent, not hidden);
          * an ``agent.identity`` event already exists (an adapter emitted it
            explicitly — do not double).

        The marker is co-located on the source event's span (never a new tree
        node) and, being content-free structural metadata (the name is a
        declared identifier, like ``from_agent``/``to_agent`` topology), is
        emitted regardless of the L1 layer / ``capture_content`` and flows
        through the same attestation hash chain (verifies like any other event)."""
        ident = honest_agent_identity(self._events)
        if ident is None:
            return
        payload: Dict[str, Any] = {"agent_name": ident["agent_name"], "source": ident["source"]}
        if ident.get("framework"):
            payload["framework"] = ident["framework"]
        self._append_locked(
            AGENT_IDENTITY,
            payload,
            span_id=ident["span_id"],  # always set (copied from the source event)
            parent_span_id=ident.get("parent_span_id"),
            span_name=None,
        )

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
            # Surface the producer-declared agent identity (if any) into ONE
            # canonical event so the Agent column fills honestly — BEFORE the
            # root synthesis / seal so it flows through the attestation chain.
            self._synthesize_identity_if_needed()
            # Give the trace a real captured root BEFORE sealing (companion to
            # atlas-app PR #2042) so the FE never has to synthesize one.
            self._synthesize_root_if_needed()
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
