"""Interleaved-run isolation for the OpenAI Agents adapter (LAY-3576 / T2).

Two runs through ONE adapter instance whose lifecycle events interleave
(start A, start B, events A, events B, end A, end B) must produce two
uploaded traces with distinct trace_ids, each containing exactly its own
run's events — no cross-contamination, no lost events.

Expected GREEN: the adapter keys RunState by trace_id under a lock
(``_trace_runs`` in openai_agents.py), so concurrent traces are isolated.
"""

from __future__ import annotations

import sys
import json
import threading
from typing import Any, Dict, List, Optional
from collections import Counter

import pytest

from .conftest import record_for_schema_lock

if sys.version_info < (3, 10):
    pytest.skip("openai-agents requires Python >= 3.10", allow_module_level=True)
try:
    import agents  # noqa: F401
except (ImportError, Exception):
    pytest.skip("openai-agents not installed or incompatible", allow_module_level=True)

from agents.tracing import TracingProcessor, set_trace_processors  # noqa: E402
from agents.tracing.spans import SpanImpl  # noqa: E402
from agents.tracing.traces import TraceImpl  # noqa: E402
from agents.tracing.span_data import (  # noqa: E402
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
)

from layerlens.instrument.adapters.frameworks.openai_agents import (  # noqa: E402
    OpenAIAgentsAdapter,
)

# -- Helpers (copied from test_openai_agents.py — do not import private
#    helpers across test modules) --


class _NoOpProcessor(TracingProcessor):
    """Minimal processor that does nothing — used to reset global state."""

    def on_trace_start(self, trace):
        pass

    def on_trace_end(self, trace):
        pass

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self):
        pass


_noop = _NoOpProcessor()


def _make_trace(trace_id: str, name: str = "test_trace") -> TraceImpl:
    """Create a real TraceImpl wired to a no-op processor.

    Tests drive the adapter's processor hooks directly.
    """
    return TraceImpl(name=name, trace_id=trace_id, group_id=None, metadata=None, processor=_noop)


def _make_span(trace_id: str, span_id: str, span_data: Any, parent_id: Optional[str] = None) -> SpanImpl:
    """Create a real SpanImpl wired to a no-op processor.

    span.start()/finish() therefore don't double-trigger the adapter;
    tests call adapter.on_span_end() manually.
    """
    return SpanImpl(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        processor=_noop,
        span_data=span_data,
    )


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate SEPARATE trace payloads per upload."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))

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


def _run_span_events(adapter: OpenAIAgentsAdapter, trace_id: str, prefix: str, marker: str) -> None:
    """Fire one run's worth of span-end events: agent + generation + function."""
    agent_span = _make_span(
        trace_id,
        f"{prefix}_agent",
        AgentSpanData(name=f"{marker}_agent", tools=[f"{marker}_tool"]),
    )
    agent_span.start()
    adapter.on_span_start(agent_span)

    gen_span = _make_span(
        trace_id,
        f"{prefix}_gen",
        GenerationSpanData(
            input=[{"role": "user", "content": f"{marker} question"}],
            output=[{"role": "assistant", "content": f"{marker} answer"}],
            model="gpt-4o",
            model_config={},
            usage={"input_tokens": 11, "output_tokens": 7},
        ),
        parent_id=f"{prefix}_agent",
    )
    gen_span.start()
    gen_span.finish()
    adapter.on_span_end(gen_span)

    tool_span = _make_span(
        trace_id,
        f"{prefix}_tool",
        FunctionSpanData(name=f"{marker}_lookup", input=f'{{"q": "{marker}"}}', output=f'{{"r": "{marker}"}}'),
        parent_id=f"{prefix}_agent",
    )
    tool_span.start()
    tool_span.finish()
    adapter.on_span_end(tool_span)

    agent_span.finish()
    adapter.on_span_end(agent_span)


#: Events one run produces via _run_span_events.
_EXPECTED_RUN_COUNTS = {
    "agent.input": 1,
    "agent.output": 1,
    "model.invoke": 1,
    "cost.record": 1,
    "tool.call": 1,
    "tool.result": 1,
    "trace.root": 1,  # the captured structural root span, emitted once per trace (trace-root)
    "agent.identity": 1,  # canonical declared-identity marker, once per trace (from the agent span name)
}


# -- Fixtures --


@pytest.fixture(autouse=True)
def clean_processors():
    """Reset global trace processors after each test."""
    yield
    set_trace_processors([])


# -- Tests --


class TestInterleavedRunIsolation:
    def test_interleaved_runs_produce_two_isolated_traces(self, mock_client):
        """start A, start B, events A, events B, end A, end B → 2 clean traces."""
        traces = _collect_traces(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()

        trace_a = _make_trace(trace_id="t_run_1")
        trace_b = _make_trace(trace_id="t_run_2")

        adapter.on_trace_start(trace_a)  # start A
        adapter.on_trace_start(trace_b)  # start B
        _run_span_events(adapter, "t_run_1", "s1", "alpha")  # events A
        _run_span_events(adapter, "t_run_2", "s2", "bravo")  # events B
        adapter.on_trace_end(trace_a)  # end A
        adapter.on_trace_end(trace_b)  # end B

        adapter.disconnect()

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    def test_threaded_runs_isolated(self, mock_client):
        """Whole runs racing on one adapter from worker threads stay isolated."""
        traces = _collect_traces(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()

        markers = [f"thread-run-{i}" for i in range(4)]
        barrier = threading.Barrier(len(markers))
        errors: List[BaseException] = []

        def _run(i: int, marker: str) -> None:
            try:
                barrier.wait(timeout=5)
                trace_id = f"t_thr_{i}"
                trace = _make_trace(trace_id=trace_id)
                adapter.on_trace_start(trace)
                _run_span_events(adapter, trace_id, f"s_thr_{i}", marker)
                adapter.on_trace_end(trace)
            except BaseException as exc:  # surfaced below — threads swallow otherwise
                errors.append(exc)

        threads = [threading.Thread(target=_run, args=(i, m)) for i, m in enumerate(markers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        adapter.disconnect()

        assert not errors, f"Adapter hooks raised in worker threads: {errors!r}"
        _assert_isolated_traces(traces, markers, _EXPECTED_RUN_COUNTS)
