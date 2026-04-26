# LangChain framework adapter

`layerlens.instrument.adapters.frameworks.langchain.LayerLensCallbackHandler`
implements the LangChain callback interface to emit LayerLens telemetry on
every LLM call, tool invocation, agent step, and chain execution.

## Install

```bash
pip install 'layerlens[langchain]'
```

Pulls `langchain>=0.2,<0.4` and `langchain-core>=0.2,<0.4`.

## Quick start

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from layerlens.instrument.adapters.frameworks.langchain import (
    LayerLensCallbackHandler,
    instrument_chain,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="langchain")
handler = LayerLensCallbackHandler()
handler.add_sink(sink)
handler.connect()

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
prompt = ChatPromptTemplate.from_messages([("user", "{q}")])
chain = prompt | llm

result = chain.invoke({"q": "What is 2 + 2?"}, config={"callbacks": [handler]})

handler.disconnect()
sink.close()
```

The same handler can be passed to any LangChain component that accepts a
`callbacks` list — `ChatOpenAI`, `LLMChain`, `AgentExecutor`, custom tools, etc.

## What's wrapped

The handler implements the LangChain callback methods:

- `on_chat_model_start`, `on_llm_start`, `on_llm_end`, `on_llm_error`
- `on_tool_start`, `on_tool_end`, `on_tool_error`
- `on_agent_action`, `on_agent_finish`
- `on_chain_start`, `on_chain_end`, `on_chain_error`

Convenience helpers wrap whole objects:

- `instrument_chain(chain, stratix=...)` — returns a `TracedChain` that injects
  the handler into every `invoke`/`batch`/`stream` call.
- `instrument_agent(agent, stratix=...)` — returns a `TracedAgent` that wraps
  an `AgentExecutor`.
- `wrap_memory(memory, ...)` — returns a `TracedMemory` that emits
  `agent.state.change` on `save_context` / `clear`.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | `on_llm_end` (success) and `on_llm_error` (failure). |
| `tool.call` | L5a | `on_tool_end` and `on_tool_error`. |
| `agent.output` | L4a | `on_agent_finish`. |
| `agent.action` | L4a | `on_agent_action`. |
| `chain.start` / `chain.end` / `chain.error` | L4a | `on_chain_*`. |
| `agent.state.change` | L4a | When a wrapped memory is updated via `wrap_memory`. |

The `model.invoke` payload includes the resolved provider (extracted from
the LangChain `serialized` dict — `openai`, `anthropic`, `bedrock`, etc.),
model name, prompts, generations, token usage if present, and latency.

## LangGraph nodes

When the handler is used inside a LangGraph run, the `metadata.langgraph_node`
field in the LangChain callback metadata is propagated to the
`agent.action` / `chain.start` payloads as `node_name`. This lets the platform
correlate per-node events back to the graph topology — see also the
`langgraph` adapter for full graph instrumentation.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Production-light: only L1 + protocol discovery + lifecycle.
handler = LayerLensCallbackHandler(capture_config=CaptureConfig.minimal())

# Recommended: L1 + L3 + L4a + L5a + L6.
handler = LayerLensCallbackHandler(capture_config=CaptureConfig.standard())

# Hand-rolled — keep tokens/costs but redact prompt/response content.
handler = LayerLensCallbackHandler(
    capture_config=CaptureConfig(
        l3_model_metadata=True,
        capture_content=False,
    ),
)
```

## BYOK

LangChain manages model API keys via the underlying provider client
(`ChatOpenAI`, `ChatAnthropic`, etc.). The handler does not touch them. For
centrally-managed keys see the platform-side BYOK store in
`docs/adapters/byok.md` (atlas-app M1.B, in flight).

## Backward compatibility

Users coming from `ateam` can keep importing the old name:

```python
from layerlens.instrument.adapters.frameworks.langchain import STRATIXCallbackHandler
```

`STRATIXCallbackHandler` is an alias for `LayerLensCallbackHandler` and will
be removed in v2.0.
