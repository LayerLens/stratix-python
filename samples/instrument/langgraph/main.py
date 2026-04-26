"""Sample: instrument a 3-node LangGraph state machine with LayerLens.

Builds a ``planner -> researcher -> writer`` ``StateGraph``, wraps it with
``LayerLensLangGraphAdapter.wrap_graph``, and invokes it. The adapter emits
``environment.config`` + ``agent.input`` + ``agent.output`` per execution
plus an ``agent.state.change`` for every node that mutates state.

The LLM call inside the ``writer`` node is mocked — the sample is designed
to be self-contained, network-free, and runnable as a smoke test for the
adapter plumbing itself. To plug in a real provider, swap the
``MockLLM.invoke(...)`` call for ``ChatOpenAI(...).invoke(...)`` (with
``OPENAI_API_KEY`` set) and instrument it through ``wrap_llm_for_langgraph``
so the call also emits ``model.invoke`` events.

A ``HandoffDetector`` is attached to the adapter so each node transition
emits an ``agent.handoff`` event — the standard signal for multi-agent
LangGraph topologies.

Run::

    pip install 'layerlens[langgraph]'
    cd samples/instrument/langgraph && python main.py

If the optional ``layerlens.instrument.transport.sink_http`` extra is
installed, telemetry ships to atlas-app over HTTP. Otherwise the sample
falls back to the in-memory event buffer that every adapter maintains for
local inspection / replay.
"""

from __future__ import annotations

import sys
from typing import Any

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.langgraph import (
    HandoffDetector,
    LayerLensLangGraphAdapter,
    wrap_llm_for_langgraph,
)


# ---------------------------------------------------------------------------
# Mock LLM — keeps the sample free of network / API keys.
# ---------------------------------------------------------------------------


class _MockLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.type = "ai"
        self.usage_metadata = {"input_tokens": 12, "output_tokens": 8}


class MockLLM:
    """Tiny stand-in for a chat model. Real samples should pass a
    ``langchain_openai.ChatOpenAI`` (or any provider with the same
    ``invoke(messages) -> response`` shape) instead.
    """

    model_name = "mock-llm-1"

    def invoke(self, messages: Any, **kwargs: Any) -> _MockLLMResponse:
        del messages, kwargs
        return _MockLLMResponse("Drafted summary using cached research.")


# ---------------------------------------------------------------------------
# Three-node graph: planner -> researcher -> writer.
# ---------------------------------------------------------------------------


def _build_compiled_graph() -> Any:
    """Construct a 3-node ``StateGraph`` and return the compiled form.

    Imported lazily so the file can still be imported as a module on a
    machine that does not have the ``langgraph`` extra installed.
    """
    from langgraph.graph import END, StateGraph

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        topic = state.get("topic", "untitled")
        return {**state, "plan": [f"research:{topic}", f"write:{topic}"], "agent": "planner"}

    def researcher(state: dict[str, Any]) -> dict[str, Any]:
        topic = state.get("topic", "untitled")
        return {**state, "research": f"facts about {topic}", "agent": "researcher"}

    def writer(state: dict[str, Any], llm: MockLLM | None = None) -> dict[str, Any]:
        # Real samples would call ``llm.invoke(...)`` with real prompts; we
        # call the mock so the sample stays network-independent.
        used_llm = llm or MockLLM()
        response = used_llm.invoke(
            [{"role": "user", "content": f"Summarize: {state.get('research')}"}],
        )
        return {**state, "summary": response.content, "agent": "writer"}

    graph: StateGraph = StateGraph(dict)
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Optional HTTP sink — the M1.E transport PR ships ``HttpEventSink``. Until
# that lands the sample falls back to the in-memory event buffer that every
# adapter maintains.
# ---------------------------------------------------------------------------


def _build_sink() -> Any | None:
    try:
        from layerlens.instrument.transport.sink_http import HttpEventSink  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    return HttpEventSink(
        adapter_name="langgraph",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )


def main() -> int:
    try:
        compiled = _build_compiled_graph()
    except ImportError:
        print(
            "langgraph not installed. Install with:\n"
            "    pip install 'layerlens[langgraph]'",
            file=sys.stderr,
        )
        return 2

    # HandoffDetector tracks the ``agent`` slot in state and emits an
    # ``agent.handoff`` event each time it changes — the standard pattern
    # for multi-agent LangGraph topologies.
    detector = HandoffDetector()
    detector.register_agents("planner", "researcher", "writer")

    adapter = LayerLensLangGraphAdapter(
        capture_config=CaptureConfig.standard(),
        handoff_detector=detector,
    )
    sink = _build_sink()
    if sink is not None:
        adapter.add_sink(sink)
    adapter.connect()

    # Demonstrate the wrapped LLM helper too — even though the writer node
    # constructs its own MockLLM internally, this shows the recommended
    # production pattern: pass a ``TracedLLM`` so each model call emits
    # ``model.invoke`` events alongside the graph events.
    traced_llm = wrap_llm_for_langgraph(MockLLM(), adapter=adapter)
    _ = traced_llm.invoke([{"role": "user", "content": "warm-up"}])

    try:
        traced_graph = adapter.wrap_graph(compiled)
        result = traced_graph.invoke({"topic": "agent observability"})
        print(f"Graph result: {result}")

        # The adapter buffers every emitted event in ``_trace_events`` for
        # local inspection / replay. Surface a quick summary so the sample
        # is also useful as an offline smoke test.
        events_by_type: dict[str, int] = {}
        for evt in adapter._trace_events:  # type: ignore[attr-defined]
            events_by_type[evt["event_type"]] = events_by_type.get(evt["event_type"], 0) + 1
        print(f"Events emitted: {dict(sorted(events_by_type.items()))}")
    finally:
        if sink is not None:
            sink.close()
        adapter.disconnect()

    if sink is None:
        print(
            "Note: layerlens.instrument.transport.sink_http not installed; "
            "events were buffered in-process only.",
        )
    else:
        print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
