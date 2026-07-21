"""L1 + L2 + linkage engine for framework adapters.

Reuses the provider harness primitives (collect / attestation / upload / poll /
teardown) and adds a loose framework contract plus platform-side linkage
verification (``_linkage.verify_linkage``).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from layerlens.instrument._capture_config import CaptureConfig

from ._timing import SELF_FLUSH_DEADLINE_S, SELF_FLUSH_DRAIN_TIMEOUT_S, SELF_FLUSH_POLL_INTERVAL_S
from ._harness import (
    _collect,
    _poll_get,
    _teardown,
    _known_cost,
    _events_by_type,
    _upload_capture,
    _assert_attestation,
)
from ._linkage import verify_linkage
from ._scenarios import SENTINEL
from ._framework_registry import FrameworkCase


def _assert_variant_depth(variant: str, by_type: Dict[str, Any], tag: str) -> None:
    """A depth variant must prove the depth it claims (ADP-partials Cluster F), or
    it is a toothless lane. "async" just relies on the standard model.invoke
    contract (that the async path emits at all). ``by_type`` values may be full
    events (ambient path) or bare payloads (self-flushing path) — normalize."""

    def _pl(e: Any) -> Dict[str, Any]:
        return e.get("payload", e) if isinstance(e, dict) else {}

    if variant == "tool":
        assert by_type.get("tool.call"), f"{tag} tool variant emitted no tool.call; types={sorted(by_type)}"
    elif variant == "multi":
        assert by_type.get("agent.handoff"), (
            f"{tag} multi-agent variant emitted no agent.handoff; types={sorted(by_type)}"
        )
    elif variant == "streaming":
        mis = by_type.get("model.invoke", [])
        streamed = any((_pl(p).get("streaming") or _pl(p).get("streamed_chunks")) for p in mis)
        assert streamed, f"{tag} streaming variant emitted no streamed model.invoke; types={sorted(by_type)}"


def run_framework_case(client: Any, case: FrameworkCase, variant: str) -> Dict[str, Any]:
    """Run one (framework, variant): collect, assert, upload, verify linkage, tear down."""
    config = CaptureConfig(capture_content=False) if variant == "redaction" else CaptureConfig.standard()

    payload = _collect(client, case, variant, config)
    events = payload.get("events", [])
    by_type = _events_by_type(events)
    tag = f"[{case.id}/{variant}]"

    _assert_attestation(payload, events)
    if variant == "error":
        # The scenario provoked a model/tool failure (and swallowed the expected
        # exception); the adapter's whole contract here is that the failure
        # surfaces as agent.error. The default-flow contract (min_events /
        # expected_types) does not apply to the short error path.
        assert by_type.get("agent.error"), f"{tag} expected an agent.error event, got types {sorted(by_type)}"
    else:
        assert len(events) >= case.min_events, (
            f"{tag} {len(events)} events < min {case.min_events}; types={sorted(by_type)}"
        )
        for t in case.expected_types:
            assert t in by_type, f"{tag} missing expected event type {t!r}; got {sorted(by_type)}"
        _assert_variant_depth(variant, by_type, tag)
    if variant == "redaction":
        blob = json.dumps(events, default=str)
        assert SENTINEL not in blob, f"{tag} redaction failed: sentinel leaked into trace payload"

    backend_id = _upload_capture(client, payload)
    trace = _poll_get(client, backend_id)
    if trace is None:
        raise AssertionError(f"{tag} trace {backend_id} not found after polling")
    linkage = verify_linkage(client, backend_id)
    _teardown(client, backend_id)

    return {
        "framework": case.id,
        "variant": variant,
        "trace_id": backend_id,
        "n_events": len(events),
        "event_types": {k: len(v) for k, v in by_type.items()},
        # The report's cost column defaults to 0 when this key is absent, which
        # silently reported every paid framework lane as $0.000000. Sum the real
        # cost.record events instead — an unpriced local model (ollama) then
        # reports a truthful 0, and a paid lane reports what it truly spent.
        "total_cost_usd": _known_cost(by_type.get("cost.record", [])),
        # Same story as the cost column: absent -> the report rendered "0
        # tool.call" for a lane that really made several (browser_use makes one
        # per real browser action).
        "tool_calls": len(by_type.get("tool.call", [])),
        "attestation_ok": True,
        "redaction_ok": variant == "redaction",
        "linked": linkage.get("linked"),
        "integration_id": linkage.get("integration_id"),
        "status": linkage.get("status"),
        "verdict": "pass",
    }


def run_self_flushing_case(client: Any, case: FrameworkCase, variant: str = "default") -> Dict[str, Any]:
    """Verify an adapter that creates + uploads its own trace (global processor model).

    Such adapters (e.g. ``openai_agents``, ``crewai``, ``llamaindex``) bypass the
    ambient collector, so we wrap the client's ``traces.upload`` to capture the
    trace id(s) the adapter flushes — eagerly or deferred (drained via
    ``shutdown_uploads``) — then verify linkage on the captured trace.
    """
    import time as _time

    import layerlens.instrument._upload as _upload

    # Capture the backend trace id(s) the adapter flushes by wrapping upload on
    # the client's traces resource. Some self-flushing adapters upload eagerly
    # (openai_agents, on trace-end) and some defer via an event bus / background
    # thread (crewai), so after the workload we drain the background upload queue
    # — every upload, sync or background, routes through the wrapped method.
    captured: list = []
    uploaded_by_type: Dict[str, list] = {}
    traces_res = client.traces
    orig_upload = traces_res.upload

    def _wrapped_upload(path, **kw):  # type: ignore[no-untyped-def]
        # Read the serialized payload the adapter is uploading so a depth variant
        # (e.g. crewai "multi") can be asserted locally on the real event types.
        try:
            with open(path) as _f:
                for _p in json.load(_f):
                    for _e in _p.get("events", []):
                        uploaded_by_type.setdefault(_e.get("event_type"), []).append(_e.get("payload", {}))
        except Exception:
            pass
        result = orig_upload(path, **kw)
        if result is not None and getattr(result, "trace_ids", None):
            captured.extend(result.trace_ids)
        return result

    try:
        traces_res.upload = _wrapped_upload  # type: ignore[method-assign]
        case.runner(variant, client)  # self-flushing runners take (flow, client)
        # Deferred/event-bus flushes (e.g. crewai) can enqueue several seconds
        # after the runner returns, and the upload itself is an HTTP call —
        # poll-drain until the wrapped upload has captured a trace id (or the
        # deadline passes). shutdown_uploads is safe to call repeatedly:
        # channels re-create on demand for anything enqueued later.
        # Budgets: see _timing (the live suite's timing contract).
        deadline = _time.time() + SELF_FLUSH_DEADLINE_S
        while not captured and _time.time() < deadline:
            _time.sleep(SELF_FLUSH_POLL_INTERVAL_S)
            _upload.shutdown_uploads(timeout=SELF_FLUSH_DRAIN_TIMEOUT_S)
    finally:
        traces_res.upload = orig_upload  # type: ignore[method-assign]

    assert captured, f"[{case.id}/{variant}] adapter produced no uploaded trace"
    _assert_variant_depth(variant, uploaded_by_type, f"[{case.id}/{variant}]")
    trace_id = captured[0]
    linkage = verify_linkage(client, trace_id)
    trace = client.traces.get(trace_id)
    return {
        "framework": case.id,
        "variant": variant,
        "trace_id": trace_id,
        "event_count": getattr(trace, "event_count", None),
        "event_types": {k: len(v) for k, v in uploaded_by_type.items()},
        "tool_calls": len(uploaded_by_type.get("tool.call", [])),
        "linked": linkage.get("linked"),
        "integration_id": linkage.get("integration_id"),
        "status": linkage.get("status"),
        "verdict": "pass",
    }
