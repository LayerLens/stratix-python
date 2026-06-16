"""Interleaved-run isolation for the CrewAI adapter (LAY-3576 / T2).

Two runs through ONE adapter instance whose lifecycle events interleave
(start A, start B, events A, events B, end A, end B) must produce two
uploaded traces with distinct trace_ids, each containing exactly its own
run's events — no cross-contamination, no lost events.

Expected RED (strict xfail): the adapter stores run state in instance
scalars (``self._collector`` / ``self._crew_span_id`` /
``self._task_span_ids`` / ``self._timers`` in crewai.py), so run B's
kickoff replaces run A's collector. A's events then land in B's trace,
A's orphaned collector is never flushed, and B's completion fires into a
``None`` collector — one corrupted upload instead of two clean ones.

The sequential baseline test stays GREEN, proving the harness itself is
sound and the xfail can only fail for the corruption, not for setup.
"""

from __future__ import annotations

import sys
import json
from typing import Any, Dict, List
from collections import Counter

import pytest

if sys.version_info < (3, 10):
    pytest.skip("crewai requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("crewai.events")

from crewai.events import (  # noqa: E402
    TaskStartedEvent,
    TaskCompletedEvent,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput  # noqa: E402

from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter  # noqa: E402

# -- Helpers --


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate SEPARATE trace payloads per upload."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])

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


def _start_crew(adapter: CrewAIAdapter, marker: str) -> None:
    """Fire the run's crew kickoff (agent.input)."""
    adapter._on_crew_started(
        None,
        CrewKickoffStartedEvent(crew_name=f"{marker}-crew", inputs={"topic": f"{marker}-topic"}),
    )


def _crew_events(adapter: CrewAIAdapter, marker: str) -> None:
    """Fire the run's mid-lifecycle events: one task start + complete."""
    adapter._on_task_started(
        None,
        TaskStartedEvent(
            context=f"{marker} context",
            task_name=f"{marker}-task",
            agent_role=f"{marker}-researcher",
        ),
    )
    output = TaskOutput(description=f"{marker}-task", raw=f"{marker}-task-result", agent=f"{marker}-researcher")
    adapter._on_task_completed(None, TaskCompletedEvent(output=output, task_name=f"{marker}-task"))


def _end_crew(adapter: CrewAIAdapter, marker: str) -> None:
    """Fire the run's crew completion (agent.output + flush)."""
    output = TaskOutput(description="final", raw=f"{marker}-final-result", agent=f"{marker}-researcher")
    adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name=f"{marker}-crew", output=output))


#: Events one run produces: crew agent.input/agent.output + task agent.input/agent.output.
_EXPECTED_RUN_COUNTS = {
    "agent.input": 2,
    "agent.output": 2,
}


# -- Fixtures --


@pytest.fixture
def adapter(mock_client):
    """Create a connected CrewAI adapter inside a scoped event-bus context."""
    a = CrewAIAdapter(mock_client)
    with crewai_event_bus.scoped_handlers():
        a.connect()
        yield a
    a.disconnect()


# -- Tests --


class TestInterleavedRunIsolation:
    def test_sequential_runs_are_isolated(self, adapter, mock_client):
        """GREEN baseline: back-to-back runs produce two clean traces.

        Proves the helpers and assertions are sound, so the strict xfail
        below can only fail for the interleaving corruption itself.
        """
        traces = _collect_traces(mock_client)

        _start_crew(adapter, "alpha")
        _crew_events(adapter, "alpha")
        _end_crew(adapter, "alpha")

        _start_crew(adapter, "bravo")
        _crew_events(adapter, "bravo")
        _end_crew(adapter, "bravo")

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "LAY-3576: crewai keeps run state in instance scalars (crewai.py — "
            "self._collector/_crew_span_id/...); interleaved runs on one instance "
            "corrupt traces. RED guard for the D1b collector-convergence work "
            "(stability report §3.1) — NOT fixed in phase 4."
        ),
    )
    def test_interleaved_runs_are_isolated(self, adapter, mock_client):
        """start A, start B, events A, events B, end A, end B → 2 clean traces."""
        traces = _collect_traces(mock_client)

        _start_crew(adapter, "alpha")  # start A
        _start_crew(adapter, "bravo")  # start B — clobbers A's collector/_crew_span_id
        _crew_events(adapter, "alpha")  # events A — land in B's collector
        _crew_events(adapter, "bravo")  # events B
        _end_crew(adapter, "alpha")  # end A — flushes B's collector, resets to None
        _end_crew(adapter, "bravo")  # end B — fires into None collector, lost

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)
