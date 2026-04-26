"""Sample: instrument a LangGraph state machine with LayerLens.

Builds a tiny one-node ``StateGraph``, wraps it with
``LayerLensLangGraphAdapter.wrap_graph``, and invokes it. The adapter emits
``environment.config`` + ``agent.input`` + ``agent.output`` (and
``agent.state.change`` if the state changed) which ship to atlas-app via
``HttpEventSink``.

The sample does not require an LLM provider — the node is a pure Python
function — so no API key is needed. This keeps the sample fast, free, and
network-independent so it can be used as a smoke test for the adapter
plumbing itself.

Run::

    pip install 'layerlens[langgraph]'
    python -m samples.instrument.langgraph.main
"""

from __future__ import annotations

import sys
from typing import Any

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.langgraph import LayerLensLangGraphAdapter


def main() -> int:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        print(
            "langgraph not installed. Install with:\n"
            "    pip install 'layerlens[langgraph]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="langgraph",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = LayerLensLangGraphAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    def greet(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": ["hi"], "count": state.get("count", 0) + 1}

    graph: StateGraph = StateGraph(dict)
    graph.add_node("greet", greet)
    graph.set_entry_point("greet")
    graph.add_edge("greet", END)
    compiled = graph.compile()

    try:
        traced = adapter.wrap_graph(compiled)
        result = traced.invoke({"count": 0})
        print(f"Result: {result}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
