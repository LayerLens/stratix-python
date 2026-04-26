# LangGraph framework adapter

`layerlens.instrument.adapters.frameworks.langgraph.LayerLensLangGraphAdapter`
instruments LangGraph state machines, capturing graph execution, node
transitions, state snapshots, and agent handoffs.

## Install

```bash
pip install 'layerlens[langgraph]'
```

Pulls `langgraph>=0.2,<0.4`. The `langchain` extra is recommended too if you
use LangChain-based nodes inside the graph.

## Quick start

```python
from langgraph.graph import StateGraph, END

from layerlens.instrument.adapters.frameworks.langgraph import LayerLensLangGraphAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="langgraph")
adapter = LayerLensLangGraphAdapter()
adapter.add_sink(sink)
adapter.connect()

graph = StateGraph(dict)
graph.add_node("greet", lambda s: {"msg": "hi"})
graph.set_entry_point("greet")
graph.add_edge("greet", END)
compiled = graph.compile()

traced = adapter.wrap_graph(compiled)
result = traced.invoke({})

adapter.disconnect()
sink.close()
```

## What's wrapped

`adapter.wrap_graph(compiled_graph)` returns a wrapper that proxies
`invoke`, `ainvoke`, `stream`, and `astream`. Each call:

- Begins a `GraphExecution` and emits `environment.config` + `agent.input`.
- Tracks state hashes before/after to detect mutations.
- On completion emits `agent.output` and (if state changed) `agent.state.change`.
- On error emits `agent.output` with the exception captured.

Companion utilities:

- `trace_node(fn)` — decorator that wraps an individual node function and
  emits `agent.action` on entry/exit.
- `trace_langgraph_tool(fn)` — decorator for tool nodes; emits `tool.call`.
- `wrap_llm_for_langgraph(llm, ...)` — wraps a LangGraph LLM node so each
  invocation emits `model.invoke`.
- `HandoffDetector` — pluggable detector that compares pre/post states for
  `__next__` agent transitions and emits `agent.handoff`.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First call into a wrapped graph (per execution). |
| `agent.input` | L1 | Beginning of every wrapped graph execution. |
| `agent.output` | L1 | End of every wrapped graph execution (success or error). |
| `agent.state.change` | cross-cutting | When the state hash changes during execution. |
| `agent.action` | L4a | One per node entry/exit when the node is wrapped with `trace_node`. |
| `tool.call` | L5a | One per tool node wrapped with `trace_langgraph_tool`. |
| `model.invoke` | L3 | One per LLM call wrapped with `wrap_llm_for_langgraph`. |
| `agent.handoff` | L4a | When a `HandoffDetector` is attached and a handoff is detected. |

## State serialization

State is serialized via `LangGraphStateAdapter.get_hash` (sha256 of a JSON
form) for the `before_hash` / `after_hash` fields. The full state appears in
`agent.input` / `agent.output` only when `CaptureConfig.capture_content` is
true. Non-JSON-serializable values are coerced via `repr()`; the original
state object is never mutated.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended: L1 + L3 + L4a + L5a + L6.
adapter = LayerLensLangGraphAdapter(capture_config=CaptureConfig.standard())

# Strip prompt/response content but keep structural events.
adapter = LayerLensLangGraphAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l4a_environment_config=True,
        capture_content=False,
    ),
)
```

## BYOK

LangGraph nodes call their underlying providers directly — `ChatOpenAI`,
`ChatAnthropic`, etc. The LangGraph adapter does not own model API keys.
For platform-managed BYOK see `docs/adapters/byok.md` (atlas-app M1.B).

## Backward compatibility

```python
from layerlens.instrument.adapters.frameworks.langgraph import STRATIXLangGraphAdapter
```

`STRATIXLangGraphAdapter` is an alias for `LayerLensLangGraphAdapter` and will
be removed in v2.0.
