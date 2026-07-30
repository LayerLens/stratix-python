"""Interleaved-run isolation for the LlamaIndex adapter (LAY-3576 / T2).

Two runs through ONE adapter instance whose lifecycle events interleave
(start A, start B, events A, events B, end A, end B) must produce two
uploaded traces with distinct trace_ids, each containing exactly its own
run's events — no cross-contamination, no lost events.

Expected GREEN: the adapter keys collectors and open spans per root span
id (``_collectors`` / ``_open_spans`` in llamaindex.py), so concurrent
queries each get their own trace.
"""

from __future__ import annotations

import json
import uuid
import inspect
import threading
from typing import Any, Dict, List, Optional
from collections import Counter
from unittest.mock import MagicMock

import pytest

from .conftest import record_for_schema_lock

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.base.llms.types import (  # noqa: E402
    ChatMessage,
    MessageRole,
    ChatResponse,
)
from llama_index.core.instrumentation import get_dispatcher  # noqa: E402
from llama_index.core.base.response.schema import Response as LlamaResponse  # noqa: E402
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent  # noqa: E402
from llama_index.core.instrumentation.events.query import (  # noqa: E402
    QueryEndEvent,
    QueryStartEvent,
)

from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._upload import shutdown_uploads  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter  # noqa: E402

# -- Helpers (copied from test_llamaindex.py — do not import private
#    helpers across test modules) --


def _emit_event_via_dispatcher(event: Any, span_id: Optional[str] = None) -> None:
    """Emit an event through the LlamaIndex dispatcher."""
    if span_id is not None:
        # LlamaIndex events have span_id as a field
        object.__setattr__(event, "span_id", span_id)
    dispatcher = get_dispatcher()
    dispatcher.event(event)


def _create_span(adapter: LlamaIndexAdapter, parent_span_id: Optional[str] = None) -> str:
    """Create a span in the adapter's span handler, return span_id."""
    span_id = f"Test.method-{uuid.uuid4().hex}"
    handler = adapter._span_handler
    mock_bound = MagicMock(spec=inspect.BoundArguments)
    handler.span_enter(
        id_=span_id,
        bound_args=mock_bound,
        instance=None,
        parent_id=parent_span_id,
    )
    return span_id


def _close_span(adapter: LlamaIndexAdapter, span_id: str) -> None:
    """Close a span, triggering flush if root."""
    handler = adapter._span_handler
    mock_bound = MagicMock(spec=inspect.BoundArguments)
    handler.span_exit(
        id_=span_id,
        bound_args=mock_bound,
        instance=None,
        result=None,
    )


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate SEPARATE trace payloads per upload."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))
        # Non-empty trace_ids: an empty/None return is treated as a REJECT
        # (F-L7-002), which would drop the trace from the isolation check.
        return CreateTracesResponse(trace_ids=[data[0].get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


def _assert_isolated_traces(
    traces: List[Dict[str, Any]],
    markers: List[str],
    expected_counts: Dict[str, int],
) -> None:
    """The LAY-3576 invariant: one trace per run, keyed by content marker.

    Each marker must appear in exactly one uploaded trace; that trace must
    contain none of the other runs' markers, and exactly its own run's
    events (by event_type counts).
    """
    # flush() enqueues each trace onto the client's BACKGROUND upload channel;
    # drain it before asserting so the collected `traces` are complete (else the
    # count is racy — the upload thread may not have run yet).
    shutdown_uploads(10.0)
    assert len(traces) == len(markers), (
        f"Expected {len(markers)} uploaded traces (one per run), got {len(traces)} — "
        "interleaved runs merged or lost a trace"
    )
    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == len(markers), f"Traces must have distinct trace_ids, got {trace_ids}"

    for marker in markers:
        own = [t for t in traces if marker in json.dumps(t["events"])]
        assert len(own) == 1, f"Run marker {marker!r} must appear in exactly 1 trace, found in {len(own)}"
        trace = own[0]
        blob = json.dumps(trace["events"])
        for other in markers:
            if other != marker:
                assert other not in blob, f"Trace for run {marker!r} is contaminated with run {other!r} events"
        counts = Counter(e["event_type"] for e in trace["events"])
        assert counts == Counter(expected_counts), (
            f"Trace for run {marker!r} lost or gained events: got {dict(counts)}, expected {expected_counts}"
        )
        assert all(e["trace_id"] == trace["trace_id"] for e in trace["events"]), (
            "Events within a trace must share its trace_id"
        )


def _start_run(adapter: LlamaIndexAdapter, marker: str) -> str:
    """Open a root span and fire the run's QueryStartEvent. Returns root span id."""
    root = _create_span(adapter)
    _emit_event_via_dispatcher(
        QueryStartEvent(query=f"{marker} question", span_id=root),
        span_id=root,
    )
    return root


def _run_mid_events(adapter: LlamaIndexAdapter, root: str, marker: str) -> None:
    """Fire the run's LLM chat end event (model.invoke + cost.record)."""
    response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=f"{marker} answer"),
        raw={
            "model": "gpt-4",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )
    _emit_event_via_dispatcher(
        LLMChatEndEvent(
            messages=[ChatMessage(role=MessageRole.USER, content=f"{marker} question")],
            response=response,
            span_id=root,
        ),
        span_id=root,
    )


def _end_run(adapter: LlamaIndexAdapter, root: str, marker: str) -> None:
    """Fire the run's QueryEndEvent and close the root span (flush)."""
    _emit_event_via_dispatcher(
        QueryEndEvent(
            query=f"{marker} question",
            response=LlamaResponse(response=f"{marker} answer"),
            span_id=root,
        ),
        span_id=root,
    )
    _close_span(adapter, root)


#: Events one run produces via _start_run + _run_mid_events + _end_run.
_EXPECTED_RUN_COUNTS = {
    "agent.input": 1,
    "model.invoke": 1,
    "cost.record": 1,
    "agent.output": 1,
}


# -- Fixtures --


@pytest.fixture
def adapter(mock_client):
    # Full capture so the run-content markers the isolation check searches for
    # are actually captured — the default standard() config redacts content
    # (capture_content=False), which would erase the markers from the events.
    return LlamaIndexAdapter(mock_client, capture_config=CaptureConfig.full())


#: The adapter installs these handler classes on the global dispatcher (see
#: ``llamaindex.py`` ``_make_span_handler`` / ``_make_event_handler``). The old
#: filter matched ``"LayerLens"`` in the class name, which never hit these
#: names — so cleanup was a no-op and handlers leaked across the module.
_ADAPTER_HANDLER_NAMES = {"_SpanHandler", "_EventHandler"}


@pytest.fixture(autouse=True)
def clean_dispatcher():
    """Remove our handlers after each test to prevent leaks."""
    yield
    dispatcher = get_dispatcher()
    dispatcher.event_handlers = [h for h in dispatcher.event_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]


# -- Tests --


class TestInterleavedRunIsolation:
    def test_interleaved_runs_produce_two_isolated_traces(self, adapter, mock_client):
        """start A, start B, events A, events B, end A, end B → 2 clean traces."""
        traces = _collect_traces(mock_client)
        adapter.connect()

        root_a = _start_run(adapter, "alpha")  # start A
        root_b = _start_run(adapter, "bravo")  # start B
        _run_mid_events(adapter, root_a, "alpha")  # events A
        _run_mid_events(adapter, root_b, "bravo")  # events B
        _end_run(adapter, root_a, "alpha")  # end A
        _end_run(adapter, root_b, "bravo")  # end B

        adapter.disconnect()

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    def test_threaded_runs_isolated(self, adapter, mock_client):
        """Whole runs racing on one adapter from worker threads stay isolated."""
        traces = _collect_traces(mock_client)
        adapter.connect()

        markers = [f"thread-run-{i}" for i in range(4)]
        barrier = threading.Barrier(len(markers))
        errors: List[BaseException] = []

        def _run(marker: str) -> None:
            try:
                barrier.wait(timeout=5)
                root = _start_run(adapter, marker)
                _run_mid_events(adapter, root, marker)
                _end_run(adapter, root, marker)
            except BaseException as exc:  # surfaced below — threads swallow otherwise
                errors.append(exc)

        threads = [threading.Thread(target=_run, args=(m,)) for m in markers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        adapter.disconnect()

        assert not errors, f"Adapter hooks raised in worker threads: {errors!r}"
        _assert_isolated_traces(traces, markers, _EXPECTED_RUN_COUNTS)
