"""Interleaved-run isolation guard for the Strands adapter (LAY-3576 / T2).

THE INVARIANT: two runs driven through ONE ``StrandsAdapter`` instance,
with lifecycle events interleaved (start A, start B, mid-run A, mid-run B,
end A, end B), must upload two traces with distinct trace_ids, each holding
exactly its own run's events — no cross-contamination, no lost events.

The adapter keeps run state in instance scalars
(``src/layerlens/instrument/adapters/frameworks/strands.py`` —
``self._collector``, ``self._run_span_id``, ``self._timers``,
``self._model_span_ids``), so run B's BeforeInvocationEvent replaces run
A's collector: run A's opening events are dropped unflushed, both runs'
model events land in run B's collector, one merged trace flushes when run
A's AfterInvocationEvent arrives, and run B's closing events vanish (no
collector left).

Expected XFAIL (strict): this is the RED guard for the D1b
collector-convergence work (stability report §3.1), which moves the
self-flushing adapters onto the per-run RunState/ContextVar isolation that
PydanticAIAdapter already uses (see test_concurrency.py). When D1b lands,
the strict xfail turns into a strict XPASS failure and the marker must be
removed. The interleave is single-threaded direct hook-handler calls, so
the corruption is deterministic and the xfail cannot flap.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from .conftest import record_for_schema_lock

strands_mod = pytest.importorskip("strands")
from strands.hooks.events import (  # noqa: E402
    AfterModelCallEvent,
    AfterInvocationEvent,
    BeforeModelCallEvent,
    BeforeInvocationEvent,
)

from layerlens.instrument.adapters.frameworks.strands import (
    StrandsAdapter,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (copied minimal — do not import private helpers from siblings)
# ---------------------------------------------------------------------------


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Accumulate each uploaded trace payload separately (one entry per flush)."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))

    mock_client.traces.upload.side_effect = _capture
    return traces


def _make_agent(name: str, model_id: str) -> Mock:
    """Mock Strands agent — only the attributes the adapter reads.

    ``agent_invocations`` is an empty list so ``_emit_per_cycle_tokens``
    has no Mock cycles to leak into payloads.
    """
    agent = Mock()
    agent.name = name
    agent.model = Mock()
    agent.model.config = {"model_id": model_id}
    agent.tool_names = []
    agent.system_prompt = None
    agent.event_loop_metrics = Mock()
    agent.event_loop_metrics.agent_invocations = []
    return agent


def _make_result(message: str) -> Mock:
    result = Mock()
    result.stop_reason = "end_turn"
    result.message = message
    return result


def _stop_response(stop_reason: str = "end_turn") -> Any:
    return AfterModelCallEvent.ModelStopResponse(message=Mock(), stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Isolation invariant
# ---------------------------------------------------------------------------

XFAIL_REASON = (
    "LAY-3576: StrandsAdapter keeps run state in instance scalars "
    "(strands.py — self._collector, self._run_span_id, self._timers, "
    "self._model_span_ids); interleaved runs on one instance corrupt traces. "
    "RED guard for the D1b collector-convergence work (stability report §3.1) — "
    "NOT fixed in phase 4."
)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_interleaved_invocations_produce_two_isolated_traces(mock_client):
    traces = _collect_traces(mock_client)
    adapter = StrandsAdapter(mock_client)
    adapter.connect()
    agent_a = _make_agent(name="agent-alpha", model_id="model-alpha")
    agent_b = _make_agent(name="agent-beta", model_id="model-beta")

    # Deterministic single-thread interleave: start A, start B, mid A, mid B,
    # end A, end B.
    adapter._on_before_invocation(BeforeInvocationEvent(agent=agent_a, invocation_state={}, messages="prompt-alpha"))
    adapter._on_before_invocation(BeforeInvocationEvent(agent=agent_b, invocation_state={}, messages="prompt-beta"))
    adapter._on_before_model(BeforeModelCallEvent(agent=agent_a, invocation_state={}))
    adapter._on_after_model(AfterModelCallEvent(agent=agent_a, invocation_state={}, stop_response=_stop_response()))
    adapter._on_before_model(BeforeModelCallEvent(agent=agent_b, invocation_state={}))
    adapter._on_after_model(AfterModelCallEvent(agent=agent_b, invocation_state={}, stop_response=_stop_response()))
    adapter._on_after_invocation(
        AfterInvocationEvent(agent=agent_a, invocation_state={}, result=_make_result("answer-alpha"))
    )
    adapter._on_after_invocation(
        AfterInvocationEvent(agent=agent_b, invocation_state={}, result=_make_result("answer-beta"))
    )
    adapter.disconnect()

    summaries = [[(e["event_type"], e.get("span_name")) for e in t["events"]] for t in traces]
    assert traces, "SETUP BUG (not the isolation invariant): no trace was uploaded at all"
    assert len(traces) == 2, (
        f"Each invocation must flush its own trace: expected 2 uploads, got {len(traces)}. "
        f"Uploaded traces as (event_type, span_name): {summaries}"
    )

    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == 2, f"The two runs must not share a trace_id: {trace_ids}"

    markers = {
        "prompt-alpha": ("agent-alpha", "model-alpha", "answer-alpha"),
        "prompt-beta": ("agent-beta", "model-beta", "answer-beta"),
    }
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        inputs = [e for e in trace["events"] if e["event_type"] == "agent.input"]
        assert len(inputs) == 1, (
            f"Each trace must hold exactly one run's agent.input, got {[e['payload'].get('input') for e in inputs]}"
        )
        by_prompt[inputs[0]["payload"]["input"]] = trace
    assert set(by_prompt) == set(markers), f"Expected one trace per prompt, got {sorted(by_prompt)}"

    for prompt, own_markers in markers.items():
        text = json.dumps(by_prompt[prompt]["events"])
        for marker in (prompt, *own_markers):
            assert marker in text, f"Trace for {prompt!r} lost its own event marker {marker!r}"
        (other_prompt,) = [p for p in markers if p != prompt]
        for marker in (other_prompt, *markers[other_prompt]):
            assert marker not in text, f"Trace for {prompt!r} contaminated by other run's marker {marker!r}"
