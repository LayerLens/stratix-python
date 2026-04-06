"""Concurrency test: prove that RunState gives per-task isolation.

Two asyncio.gather runs on the same PydanticAI adapter must produce
two separate traces with independent events and distinct trace_ids.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter  # noqa: E402


def _make_agent(output_text: str = "Hello!", tools: list | None = None) -> Agent:
    agent = Agent(
        model=TestModel(custom_output_text=output_text, model_name="test-model"),
        name="test_agent",
    )
    if tools:
        for fn in tools:
            agent.tool_plain(fn)
    return agent


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate individual trace payloads."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])

    mock_client.traces.upload.side_effect = _capture
    return traces


class TestConcurrentRunIsolation:
    def test_concurrent_runs_produce_separate_traces(self, mock_client: Any) -> None:
        """Two asyncio.gather runs on the same adapter → two distinct traces."""
        traces = _collect_traces(mock_client)

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"72F in {city}"

        agent = _make_agent(output_text="done", tools=[get_weather])
        adapter = PydanticAIAdapter(mock_client)
        adapter.connect(target=agent)

        async def run_both() -> None:
            await asyncio.gather(
                agent.run("question A"),
                agent.run("question B"),
            )

        asyncio.run(run_both())

        adapter.disconnect()

        # Two runs → two traces
        assert len(traces) == 2, f"Expected 2 traces, got {len(traces)}"

        # Distinct trace_ids
        trace_ids = {t["trace_id"] for t in traces}
        assert len(trace_ids) == 2, f"Traces must have different trace_ids, got {trace_ids}"

        for trace in traces:
            events = trace["events"]
            event_types = [e["event_type"] for e in events]

            # Each trace has the core lifecycle events
            assert "agent.input" in event_types, f"Missing agent.input in {event_types}"
            assert "agent.output" in event_types, f"Missing agent.output in {event_types}"
            assert "model.invoke" in event_types, f"Missing model.invoke in {event_types}"

            # All events in a single trace share the same trace_id
            assert all(
                e["trace_id"] == trace["trace_id"] for e in events
            ), "Events within a trace must share trace_id"

            # agent.output has status ok
            output_events = [e for e in events if e["event_type"] == "agent.output"]
            assert len(output_events) == 1
            assert output_events[0]["payload"]["status"] == "ok"
