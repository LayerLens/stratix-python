# LangGraph adapter

Instruments [LangGraph](https://langchain-ai.github.io/langgraph/) graphs with
LayerLens tracing. Subclasses the LangChain callback handler (M1.C reference
template) and adds graph-state hashing for replay.

## Install

```bash
pip install layerlens[langgraph]
```

Pulls `langgraph>=0.2.0` and `langchain-core>=0.1.0`. LangGraph 0.2+ requires
Pydantic v2 (`requires_pydantic="2"` on the handler).

## Usage

```python
from langgraph.graph import StateGraph
from layerlens.instrument.adapters.frameworks import LangGraphCallbackHandler

handler = LangGraphCallbackHandler(client=layerlens_client)

graph = StateGraph(state_schema=MyState)
# ... add nodes / edges ...
app = graph.compile()

app.invoke(initial_state, config={"callbacks": [handler]})
```

## Event surface

Inherits everything from the LangChain handler (`chain_start`/`chain_end`,
`llm_start`/`llm_end`, `tool_start`/`tool_end`) and adds:

- `agent.state` event per node transition with a SHA-256 hash of the
  serialized graph state. Hashing is gated by ``emit_state_hash=True`` (the
  default) and can be disabled if state is too large.
- `agent.handoff` event when one node hands control to another, derived from
  the LangGraph node ID transition rather than message-content heuristics.

## Sample

[`samples/instrument/langgraph/example.py`](../../../samples/instrument/langgraph/example.py)

## Compat

- LangGraph 0.2+ (Pydantic v2-only)
- Python 3.9+
