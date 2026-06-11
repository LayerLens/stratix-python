"""L1 + L2 + linkage engine for framework adapters.

Reuses the provider harness primitives (collect / attestation / upload / poll /
teardown) and adds a loose framework contract plus platform-side linkage
verification (``_linkage.verify_linkage``).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from layerlens.instrument._capture_config import CaptureConfig

from ._harness import (
    _collect,
    _poll_get,
    _teardown,
    _events_by_type,
    _upload_capture,
    _assert_attestation,
)
from ._linkage import verify_linkage
from ._scenarios import SENTINEL
from ._framework_registry import FrameworkCase


def run_framework_case(client: Any, case: FrameworkCase, variant: str) -> Dict[str, Any]:
    """Run one (framework, variant): collect, assert, upload, verify linkage, tear down."""
    config = CaptureConfig(capture_content=False) if variant == "redaction" else CaptureConfig.standard()

    payload = _collect(client, case, variant, config)
    events = payload.get("events", [])
    by_type = _events_by_type(events)
    tag = f"[{case.id}/{variant}]"

    _assert_attestation(payload, events)
    assert len(events) >= case.min_events, (
        f"{tag} {len(events)} events < min {case.min_events}; types={sorted(by_type)}"
    )
    for t in case.expected_types:
        assert t in by_type, f"{tag} missing expected event type {t!r}; got {sorted(by_type)}"
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
    traces_res = client.traces
    orig_upload = traces_res.upload

    def _wrapped_upload(path, **kw):  # type: ignore[no-untyped-def]
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
        deadline = _time.time() + 30.0
        while not captured and _time.time() < deadline:
            _time.sleep(0.5)
            _upload.shutdown_uploads(timeout=20)
    finally:
        traces_res.upload = orig_upload  # type: ignore[method-assign]

    assert captured, f"[{case.id}/{variant}] adapter produced no uploaded trace"
    trace_id = captured[0]
    linkage = verify_linkage(client, trace_id)
    trace = client.traces.get(trace_id)
    return {
        "framework": case.id,
        "variant": variant,
        "trace_id": trace_id,
        "event_count": getattr(trace, "event_count", None),
        "linked": linkage.get("linked"),
        "integration_id": linkage.get("integration_id"),
        "status": linkage.get("status"),
        "verdict": "pass",
    }
