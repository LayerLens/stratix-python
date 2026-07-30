# Instrumentation

The `layerlens.instrument` module provides tracing primitives and provider/framework adapters for automatic LLM observability.

## Overview

### Using Synchronous Client

```python
from layerlens import Stratix
from layerlens.instrument import trace, span

client = Stratix()

@trace(client)
def my_agent(query: str):
    with span("process", kind="internal") as s:
        result = do_work(query)
        s.output = result
    return result

my_agent("Hello")
```

### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix
from layerlens.instrument import trace, span

client = AsyncStratix()

@trace(client)
async def my_agent(query: str):
    with span("process") as s:
        result = await do_work(query)
        s.output = result
    return result

asyncio.run(my_agent("Hello"))
```

## Core API

### `trace(client, name=None, metadata=None)`

Decorator that creates a root span and uploads the trace on function completion.

#### Parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `client` | `Stratix \| AsyncStratix` | Yes | SDK client used to upload the trace |
| `name` | `str \| None` | No | Override span name (defaults to function name) |
| `metadata` | `dict \| None` | No | Arbitrary metadata attached to the root span |

#### Behavior

- Creates a `TraceRecorder` and root `SpanData`
- Sets `_current_recorder` and `_current_span` context variables
- Captures function arguments as `input`
- Captures return value as `output`
- On error: sets `status="error"` and records the error message
- On completion: serializes span tree to a temp JSON file, calls `client.traces.upload()`, deletes the temp file
- Resets context variables in a `finally` block
- Works with both sync and async functions

#### Example

```python
@trace(client)
def my_agent(query: str):
    return process(query)

@trace(client, name="custom-name")
async def my_async_agent(query: str):
    return await process(query)
```

### `span(name, kind="internal", input=None, metadata=None)`

Context manager that creates a child span under the current active span.

#### Parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `name` | `str` | Yes | Display name for the span |
| `kind` | `str` | No | Span type: `"internal"`, `"llm"`, `"retriever"`, `"tool"`, `"chain"` |
| `input` | `Any` | No | Input data for the span |
| `metadata` | `dict \| None` | No | Arbitrary metadata attached to the span |

#### Returns

Returns a `SpanData` object (or a no-op dummy if no trace is active).

#### Behavior

- If called outside a `@trace` context, returns a no-op context manager
- Creates a `SpanData` with the given name and kind
- Appends the span to the current parent's `children` list
- Sets `_current_span` to the new span for the duration of the `with` block
- Restores the previous span on exit
- On error inside the block: sets `status="error"`, records error, re-raises

#### Example

```python
@trace(client)
def my_agent(query: str):
    with span("step-1", kind="tool") as s:
        s.input = query
        result = tool_call(query)
        s.output = result
        s.metadata["tool_version"] = "1.0"
    return result
```

### `SpanData`

Dataclass representing a single span in the trace tree.

#### Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `name` | `str` | (required) | Span display name |
| `span_id` | `str` | auto-generated | Unique identifier (UUID hex, 16 chars) |
| `parent_id` | `str \| None` | `None` | Parent span ID |
| `start_time` | `float` | `time.time()` | Unix timestamp |
| `end_time` | `float \| None` | `None` | Unix timestamp when finished |
| `status` | `str` | `"ok"` | `"ok"` or `"error"` |
| `kind` | `str` | `"internal"` | Span type |
| `input` | `Any` | `None` | Input data |
| `output` | `Any` | `None` | Output data |
| `error` | `str \| None` | `None` | Error message |
| `metadata` | `dict` | `{}` | Arbitrary key-value metadata |
| `children` | `list[SpanData]` | `[]` | Child spans |

#### Methods

##### `finish(error=None)`

Sets `end_time` to the current time. If `error` is provided, sets `status="error"` and records the error message.

##### `to_dict()`

Serializes the span tree to a JSON-compatible dictionary, recursively including all children.

### `TraceRecorder`

Collects the span tree and handles flushing to the LayerLens API.

#### Methods

##### `flush()`

Serializes the root span tree to a temporary JSON file, calls `client.traces.upload(path)`, and deletes the temp file. Used by the `@trace` decorator for sync functions.

##### `async_flush()`

Async version of `flush()`. Used by the `@trace` decorator for async functions.

## Provider Adapters

### `instrument_openai(client)`

Monkey-patches `client.chat.completions.create` on an OpenAI client instance.

```python
from layerlens.instrument.adapters.providers.openai import instrument_openai

instrument_openai(openai_client)
```

#### Classes

| Class | Description |
| ----- | ----------- |
| `OpenAIProvider` | Provider adapter with `connect_client()` / `disconnect()` |

### `instrument_anthropic(client)`

Monkey-patches `client.messages.create` on an Anthropic client instance.

```python
from layerlens.instrument.adapters.providers.anthropic import instrument_anthropic

instrument_anthropic(anthropic_client)
```

#### Classes

| Class | Description |
| ----- | ----------- |
| `AnthropicProvider` | Provider adapter with `connect_client()` / `disconnect()` |

### `instrument_litellm()`

Monkey-patches `litellm.completion` and `litellm.acompletion` at the module level.

```python
from layerlens.instrument.adapters.providers.litellm import instrument_litellm, uninstrument_litellm

instrument_litellm()      # Patch
uninstrument_litellm()    # Restore
```

## Framework Adapters

### `LangChainCallbackHandler(client)`

LangChain `BaseCallbackHandler` implementation that builds a span tree from chain/LLM/tool/retriever events.

```python
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

handler = LangChainCallbackHandler(client)
chain.invoke(input, config={"callbacks": [handler]})
```

#### Supported Callbacks

| Callback | Span Kind |
| -------- | --------- |
| `on_chain_start` / `on_chain_end` / `on_chain_error` | `chain` |
| `on_llm_start` / `on_llm_end` / `on_llm_error` | `llm` |
| `on_chat_model_start` | `llm` |
| `on_tool_start` / `on_tool_end` / `on_tool_error` | `tool` |
| `on_retriever_start` / `on_retriever_end` / `on_retriever_error` | `retriever` |

### `LangGraphCallbackHandler(client)`

Extends `LangChainCallbackHandler` with LangGraph node name extraction.

```python
from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

handler = LangGraphCallbackHandler(client)
graph.invoke(input, config={"callbacks": [handler]})
```

Extracts node names from `metadata.langgraph_node` or plain tags (skipping internal `graph:step:*` tags).

## Next Steps

- [Instrumentation Guide](../instrumentation/README.md) for usage patterns and examples
- [Traces API Reference](traces.md) for the underlying upload mechanism
