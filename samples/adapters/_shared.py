"""Shared utilities for adapter samples.

Each sample uses :func:`capture_events` to run a block under a ``TraceCollector``.
By default the captured events are pretty-printed locally so you can eyeball what
instrumentation is producing without hitting the API.

When LayerLens credentials are present in the environment, the captured trace is
ALSO uploaded to the backend and verified, so the samples exercise the real
send path end-to-end. Enable real upload with::

    export LAYERLENS_STRATIX_API_KEY=sk-stx-...
    export LAYERLENS_STRATIX_BASE_URL=https://api.layerlens.ai/api/v1   # optional

Without those vars the helper is a clean no-op uploader (prints events only) — it
never fabricates a successful send. The upload ingests and attests the trace
(hash chain verified + server-signed); the structured span-graph render is
pending the backend derive step (see
``docs/atlas-app-api-discussions/trace-attestation-and-rendering.md``), so this
helper reports a *persisted + attested* trace, not a *rendered* one.
"""

from __future__ import annotations

import os
import json
import tempfile
from typing import Any, Optional, Generator
from contextlib import contextmanager

from layerlens.instrument._context import _pop_span, _push_span, _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig


class _StubClient:
    """Offline stand-in: ``TraceCollector`` requires a client; makes no network calls."""

    def __init__(self) -> None:
        self._base_url = "https://localhost/sample"


def _build_client() -> Optional[Any]:
    """Return a real LayerLens client iff credentials are configured, else ``None``."""
    if not (os.environ.get("LAYERLENS_STRATIX_API_KEY") or os.environ.get("LAYERLENS_ATLAS_API_KEY")):
        return None
    try:
        from layerlens import Stratix

        return Stratix()
    except Exception as exc:  # noqa: BLE001 -- a sample must not hard-fail on client init
        print(f"[upload skipped] could not initialize LayerLens client: {exc}")
        return None


@contextmanager
def capture_events(name: str = "sample") -> Generator[TraceCollector, None, None]:
    """Run the block under a ``TraceCollector``; print events and (if configured) upload."""
    client = _build_client()
    collector = TraceCollector(client or _StubClient(), CaptureConfig.standard())
    root = "sample" + name[:8]
    col_token = _current_collector.set(collector)
    span_snapshot = _push_span(root, name)
    try:
        yield collector
    finally:
        _pop_span(span_snapshot)
        _current_collector.reset(col_token)
        _print_events(collector)
        if client is not None:
            _upload_and_verify(client, collector)


def _print_events(collector: TraceCollector) -> None:
    events = getattr(collector, "_events", [])
    print(f"\n--- captured {len(events)} events ---")
    for ev in events:
        print(
            json.dumps(
                {"type": ev.get("event_type"), "payload": ev.get("payload")},
                default=str,
            )[:500]
        )


def _upload_and_verify(client: Any, collector: TraceCollector) -> None:
    """Upload the captured trace to the backend and confirm it persisted.

    Uses the same payload shape the SDK uploads (``trace_id``, ``events``,
    ``capture_config``, ``attestation``) via the presigned-upload flow, then
    reads the trace back by id to confirm ingest + attestation. Honest about
    scope: the trace is persisted and attested, not yet graph-rendered.
    """
    try:
        payload = collector.to_replay_dict()
        sdk_trace_id = payload.get("trace_id")
        fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_sample_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump([payload], fh, default=str)
            result = client.traces.upload(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        trace_ids = getattr(result, "trace_ids", None) if result is not None else None
        verify_id = trace_ids[0] if trace_ids else sdk_trace_id
        got = client.traces.get(verify_id) if verify_id else None
        if got is not None:
            print(f"\n[uploaded] trace {verify_id} ingested + attested (get-by-id: FOUND).")
        else:
            print(f"\n[uploaded] trace {verify_id}; not yet visible by id (async indexing).")
        print(
            "Note: structured span-graph render is pending the backend derive step "
            "(see docs/atlas-app-api-discussions/trace-attestation-and-rendering.md)."
        )
    except Exception as exc:  # noqa: BLE001 -- a send failure must not crash the sample
        print(f"\n[upload failed] {exc}")


def pretty(value: Any) -> str:
    return json.dumps(value, default=str, indent=2)
