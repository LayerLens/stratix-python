"""Runnable sample: drive the Langfuse adapter end-to-end.

This sample is **fully mocked** — every call to the Langfuse HTTP API is
intercepted in-process. No real network traffic is generated and no
Langfuse credentials are required. The sample exists to demonstrate the
adapter's API surface and act as a smoke test that the
``[langfuse-importer]`` extra installs cleanly.

Three flows are exercised end-to-end:

    1. ``[import]`` Backfill traces from Langfuse into LayerLens. The
       Langfuse REST surface is replaced with a small in-process client
       that returns two synthetic traces (one with a ``GENERATION``
       observation, one with a ``TOOL`` ``SPAN``). Each Langfuse trace
       is mapped into LayerLens canonical events and emitted through a
       recording stratix sink.
    2. ``[export]`` Push three LayerLens events (``agent.input``,
       ``model.invoke``, ``agent.output``) for a single trace back into
       Langfuse via the batch ingestion endpoint. The mocked client
       records the ingestion payload so the sample can print a summary.
    3. ``[bidirectional]`` Run a combined import + export cycle through
       the high-level ``adapter.sync()`` API and print the merged
       ``SyncResult``.

Run::

    pip install 'layerlens[langfuse-importer]'   # extra is empty (stdlib)
    python -m samples.instrument.langfuse.main

Exits 0 on success.

Optional environment for an additional live ``connect()`` smoke against a
real Langfuse instance (skipped when not set):

* ``LANGFUSE_PUBLIC_KEY``
* ``LANGFUSE_SECRET_KEY``
* ``LANGFUSE_HOST``  — defaults to ``https://cloud.langfuse.com``
"""

from __future__ import annotations

import os
import sys
from typing import Any
from datetime import datetime, timezone

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter
from layerlens.instrument.adapters.frameworks.langfuse.config import (
    SyncDirection,
    LangfuseConfig,
)

UTC = timezone.utc  # 3.9 / 3.10 compat alias for ``datetime.UTC``


# ---------------------------------------------------------------------------
# In-process recording sink (stand-in for HttpEventSink / OTLP)
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Stand-in for a real LayerLens client — records every emit() call."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append((args[0], args[1]))


# ---------------------------------------------------------------------------
# In-process Langfuse API client mock
# ---------------------------------------------------------------------------


class _MockLangfuseClient:
    """Mock of ``LangfuseAPIClient`` with the methods the adapter calls.

    The adapter's importer, exporter, and sync orchestrator only call a
    small surface of the real client — health_check, get_all_traces,
    get_trace, ingestion_batch. We implement those with in-memory
    fixture data so the sample is fully offline.
    """

    def __init__(self) -> None:
        # Fixture: two Langfuse traces, one with a generation, one with a tool.
        self._traces: list[dict[str, Any]] = [
            {
                "id": "lf-trace-001",
                "name": "support-agent",
                "timestamp": "2026-04-25T10:00:00Z",
                "endTime": "2026-04-25T10:00:05Z",
                "input": {"question": "Where is my order?"},
                "output": {"answer": "Your order shipped on 2026-04-24."},
                "tags": ["production", "tier-pro"],
                "metadata": {"customer_id": "cust-42"},
                "observations": [
                    {
                        "id": "obs-gen-001",
                        "type": "GENERATION",
                        "name": "gpt-4o-mini",
                        "model": "gpt-4o-mini",
                        "startTime": "2026-04-25T10:00:01Z",
                        "endTime": "2026-04-25T10:00:04Z",
                        "usage": {
                            "promptTokens": 150,
                            "completionTokens": 32,
                            "totalTokens": 182,
                        },
                        "totalCost": 0.000273,
                        "modelParameters": {"temperature": 0.0, "max_tokens": 64},
                    }
                ],
            },
            {
                "id": "lf-trace-002",
                "name": "tool-using-agent",
                "timestamp": "2026-04-25T10:05:00Z",
                "endTime": "2026-04-25T10:05:02Z",
                "input": {"question": "Look up order O-123"},
                "output": {"answer": "Found order O-123 (status=shipped)."},
                "tags": ["production"],
                "observations": [
                    {
                        "id": "obs-tool-001",
                        "type": "SPAN",
                        "name": "tool_lookup_order",
                        "startTime": "2026-04-25T10:05:00Z",
                        "endTime": "2026-04-25T10:05:01Z",
                        "input": {"order_id": "O-123"},
                        "output": {"status": "shipped"},
                        "metadata": {"type": "TOOL", "tool_name": "lookup_order"},
                    }
                ],
            },
        ]
        # Records every batch ingested via ``ingestion_batch`` — used to
        # verify the export path round-trips correctly.
        self.ingested_batches: list[list[dict[str, Any]]] = []

    # --- Methods the adapter actually invokes ---

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    def get_all_traces(
        self,
        limit: int = 50,  # noqa: ARG002
        tags: list[str] | None = None,
        from_timestamp: datetime | None = None,  # noqa: ARG002
        to_timestamp: datetime | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        out = self._traces
        if tags:
            tag_set = set(tags)
            out = [t for t in out if tag_set.intersection(t.get("tags", []))]
        # Trim to summary-shape (the real API returns summaries, not full bodies).
        return [
            {"id": t["id"], "tags": t.get("tags", []), "timestamp": t["timestamp"]}
            for t in out
        ]

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        for t in self._traces:
            if t["id"] == trace_id:
                return t
        raise KeyError(f"Unknown trace id {trace_id!r}")

    def ingestion_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        # Real Langfuse responds with ``{"successes": [...], "errors": [...]}``.
        self.ingested_batches.append(events)
        return {
            "successes": [{"id": e.get("id"), "status": 201} for e in events],
            "errors": [],
        }


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


def _build_adapter(stratix: _RecordingStratix) -> tuple[LangfuseAdapter, _MockLangfuseClient]:
    """Construct a LangfuseAdapter wired to the in-process mock client.

    We construct the adapter *without* a config so ``connect()`` does
    not try to stand up a real ``LangfuseAPIClient`` (which would attempt
    a live health check). We then swap the mock client in directly and
    re-initialise the importer / exporter / sync sub-components — this
    is the same wiring path ``connect(config=...)`` would normally run.
    """
    from layerlens.instrument.adapters._base.adapter import AdapterStatus
    from layerlens.instrument.adapters.frameworks.langfuse.sync import BidirectionalSync
    from layerlens.instrument.adapters.frameworks.langfuse.exporter import TraceExporter
    from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

    adapter = LangfuseAdapter(
        stratix=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()  # no-config path → HEALTHY but no client

    mock_client = _MockLangfuseClient()
    adapter._client = mock_client  # type: ignore[assignment]
    adapter._importer = TraceImporter(mock_client, adapter._sync_state)  # type: ignore[arg-type]
    adapter._exporter = TraceExporter(mock_client, adapter._sync_state)  # type: ignore[arg-type]
    adapter._sync = BidirectionalSync(
        importer=adapter._importer,
        exporter=adapter._exporter,
        state=adapter._sync_state,
    )
    adapter._langfuse_healthy = True
    adapter._status = AdapterStatus.HEALTHY
    return adapter, mock_client


# ---------------------------------------------------------------------------
# Flow 1 — Import traces from Langfuse into LayerLens
# ---------------------------------------------------------------------------


def _flow_import(adapter: LangfuseAdapter, stratix: _RecordingStratix) -> int:
    print("[import] importing all traces from Langfuse...")
    result = adapter.import_traces()
    if result.errors:
        print(f"[import] FAILED: {result.errors}", file=sys.stderr)
        return 1

    print(
        f"[import] imported={result.imported_count} skipped={result.skipped_count} "
        f"failed={result.failed_count} duration_ms={result.duration_ms:.1f}"
    )

    # Group emitted events by type for a tidy summary.
    by_type: dict[str, int] = {}
    for event_type, _payload in stratix.events:
        by_type[event_type] = by_type.get(event_type, 0) + 1
    print(f"[import] events emitted by type: {dict(sorted(by_type.items()))}")

    if result.imported_count != 2:
        print(
            f"[import] expected 2 traces imported, got {result.imported_count}",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Flow 2 — Export LayerLens events back to Langfuse
# ---------------------------------------------------------------------------


def _flow_export(adapter: LangfuseAdapter, mock_client: _MockLangfuseClient) -> int:
    print("[export] exporting one synthetic LayerLens trace to Langfuse...")
    events_by_trace: dict[str, list[dict[str, Any]]] = {
        "ll-trace-001": [
            {
                "event_type": "agent.input",
                "trace_id": "ll-trace-001",
                "timestamp": "2026-04-25T11:00:00Z",
                "sequence_id": 0,
                "payload": {
                    "agent_id": "billing-agent",
                    "input_text": "Refund my last invoice",
                    "input": {"intent": "refund", "invoice_id": "INV-9001"},
                },
            },
            {
                "event_type": "model.invoke",
                "trace_id": "ll-trace-001",
                "timestamp": "2026-04-25T11:00:00Z",
                "sequence_id": 1,
                "payload": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "tokens_prompt": 200,
                    "tokens_completion": 40,
                    "tokens_total": 240,
                    "latency_ms": 1500,
                },
            },
            {
                "event_type": "cost.record",
                "trace_id": "ll-trace-001",
                "timestamp": "2026-04-25T11:00:00Z",
                "sequence_id": 2,
                "payload": {
                    "model": "gpt-4o-mini",
                    "cost_usd": 0.00036,
                    "tokens_prompt": 200,
                    "tokens_completion": 40,
                },
            },
            {
                "event_type": "agent.output",
                "trace_id": "ll-trace-001",
                "timestamp": "2026-04-25T11:00:01Z",
                "sequence_id": 3,
                "payload": {
                    "agent_id": "billing-agent",
                    "output_text": "Refund processed.",
                    "output": {"status": "refunded", "amount": 49.0},
                },
            },
        ],
    }

    result = adapter.export_traces(events_by_trace=events_by_trace)
    if result.errors:
        print(f"[export] FAILED: {result.errors}", file=sys.stderr)
        return 1

    print(
        f"[export] exported={result.exported_count} skipped={result.skipped_count} "
        f"failed={result.failed_count} duration_ms={result.duration_ms:.1f}"
    )
    print(f"[export] mock recorded {len(mock_client.ingested_batches)} ingestion batch(es)")

    # Confirm the batch contained a trace-create + at least one observation.
    if not mock_client.ingested_batches:
        print("[export] no batches ingested", file=sys.stderr)
        return 1
    batch = mock_client.ingested_batches[-1]
    types = sorted({e.get("type") for e in batch})
    print(f"[export] batch event types: {types}")
    if "trace-create" not in types:
        print("[export] missing trace-create event in batch", file=sys.stderr)
        return 1

    # Confirm the loop-prevention tag is present on the exported trace.
    trace_event = next((e for e in batch if e.get("type") == "trace-create"), None)
    if trace_event is None:
        print("[export] no trace-create body found", file=sys.stderr)
        return 1
    tags = trace_event.get("body", {}).get("tags", [])
    if "layerlens-exported" not in tags:
        print(
            f"[export] expected ``layerlens-exported`` tag, got {tags}",
            file=sys.stderr,
        )
        return 1
    print(f"[export] loop-prevention tags: {tags}")
    return 0


# ---------------------------------------------------------------------------
# Flow 3 — Bidirectional sync()
# ---------------------------------------------------------------------------


def _flow_bidirectional(adapter: LangfuseAdapter) -> int:
    print("[bidirectional] running a dry-run sync() in BIDIRECTIONAL mode...")
    result = adapter.sync(
        direction=SyncDirection.BIDIRECTIONAL,
        dry_run=True,
        events_by_trace={
            "ll-trace-002": [
                {
                    "event_type": "agent.input",
                    "trace_id": "ll-trace-002",
                    "timestamp": "2026-04-25T12:00:00Z",
                    "sequence_id": 0,
                    "payload": {
                        "agent_id": "qa-agent",
                        "input_text": "Run a smoke test.",
                    },
                }
            ]
        },
    )
    print(
        f"[bidirectional] direction={result.direction.value} dry_run={result.dry_run} "
        f"imported={result.imported_count} exported={result.exported_count} "
        f"skipped={result.skipped_count} failed={result.failed_count}"
    )
    return 0


# ---------------------------------------------------------------------------
# Optional: live connect smoke (only if LANGFUSE_* env vars present)
# ---------------------------------------------------------------------------


def _have_langfuse_env() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def _flow_live_connect_smoke() -> int:
    cfg = LangfuseConfig(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    adapter = LangfuseAdapter(config=cfg, capture_config=CaptureConfig.standard())
    try:
        adapter.connect()
        status = adapter.get_status()
        print(
            f"[live-connect] connected={status['connected']} "
            f"langfuse_healthy={status['langfuse_healthy']} host={status['host']}"
        )
    finally:
        adapter.disconnect()
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    stratix = _RecordingStratix()
    adapter, mock_client = _build_adapter(stratix)

    rc = _flow_import(adapter, stratix)
    if rc:
        adapter.disconnect()
        return rc

    rc = _flow_export(adapter, mock_client)
    if rc:
        adapter.disconnect()
        return rc

    rc = _flow_bidirectional(adapter)
    if rc:
        adapter.disconnect()
        return rc

    state = adapter.sync_state
    print(
        f"[summary] sink recorded {len(stratix.events)} events; "
        f"sync_state imported={len(state.imported_trace_ids)} "
        f"exported={len(state.exported_trace_ids)} "
        f"quarantined={len(state.quarantined_trace_ids)}"
    )

    adapter.disconnect()

    if _have_langfuse_env():
        rc = _flow_live_connect_smoke()
        if rc:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
