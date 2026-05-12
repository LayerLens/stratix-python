# LlamaIndex framework adapter

`layerlens.instrument.adapters.frameworks.llama_index.LlamaIndexAdapter`
instruments [LlamaIndex](https://github.com/run-llama/llama_index) agents,
workflows, query engines, and retrievers using the framework's modern
**Instrumentation Module** (v0.10.20+) — non-invasive, no monkey-patching.

## Install

```bash
pip install 'layerlens[llama-index]'
```

Pulls `llama-index>=0.10,<0.13`. Requires Python 3.10+.

## Quick start

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from layerlens.instrument.adapters.frameworks.llama_index import (
    LlamaIndexAdapter,
    instrument_workflow,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="llama_index")
adapter = LlamaIndexAdapter()
adapter.add_sink(sink)
adapter.connect()
adapter.instrument_workflow(None)  # registers the global event handler

llm = OpenAI(model="gpt-4o-mini")
agent = ReActAgent.from_tools([], llm=llm)
response = agent.chat("What is 2+2?")

adapter.disconnect()
sink.close()
```

`instrument_workflow(workflow=None)` (called once per process) registers a
global LlamaIndex `BaseEventHandler` that captures every event LlamaIndex
dispatches.

## What's wrapped

`adapter.instrument_workflow(...)` registers a `BaseEventHandler` with
`llama_index.core.instrumentation.get_dispatcher()`. The handler observes:

- LLM events (`LLMChatStartEvent`, `LLMChatEndEvent`,
  `LLMCompletionStartEvent`, `LLMCompletionEndEvent`)
- Tool events (`AgentToolCallEvent`)
- Agent events (`AgentRunStepStartEvent`, `AgentRunStepEndEvent`,
  `AgentChatWithStepStartEvent`, `AgentChatWithStepEndEvent`)
- Retrieval events (`RetrievalStartEvent`, `RetrievalEndEvent`)
- Embedding events (`EmbeddingStartEvent`, `EmbeddingEndEvent`)

`disconnect()` removes the handler from the dispatcher's
`event_handlers` list, restoring the original behaviour.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First agent / workflow event per process. |
| `agent.input` | L1 | `AgentChatWithStepStartEvent` / agent step start. |
| `agent.output` | L1 | `AgentChatWithStepEndEvent` / agent step end. |
| `agent.action` | L4a | Per `AgentRunStepEndEvent`. |
| `tool.call` | L5a | Per `AgentToolCallEvent`. |
| `model.invoke` | L3 | Per LLM start/end pair. |

## LlamaIndex specifics

- **Workflows**: the new `Workflow` class emits dispatcher events the same
  way; the same handler captures both classic agents (`ReActAgent`,
  `OpenAIAgent`) and workflow `@step` runs.
- **RAG retrievers**: retrieval events are surfaced as `tool.call` with
  `tool_name="retriever"` and the resolved chunk count.
- **Streaming**: streamed LLM responses fire one `LLMChatEndEvent` after
  the final chunk; the adapter emits one consolidated `model.invoke`.
- **Span propagation**: LlamaIndex span IDs propagate into the event
  payload as `span_id` / `parent_span_id` for tree reconstruction.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = LlamaIndexAdapter(capture_config=CaptureConfig.standard())

# Production-light: drop retrieved chunks (large), keep query + result count.
adapter = LlamaIndexAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

LlamaIndex LLM integrations (`OpenAI`, `Anthropic`, `Bedrock`, etc.) read
their own credentials. The adapter does not own them. For platform-managed
BYOK see `docs/adapters/byok.md` (atlas-app M1.B).
