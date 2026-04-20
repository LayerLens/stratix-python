# Instrumentation Quick Start

This guide covers the core instrumentation API: the `@trace` decorator and the `span()` context manager.

## Installation

The instrumentation module is included in the base SDK — no extra dependencies needed:

```bash
pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```

Provider adapters require their respective SDK as an optional dependency:

```bash
pip install layerlens[openai]      # OpenAI
pip install layerlens[anthropic]   # Anthropic
pip install layerlens[litellm]     # LiteLLM (100+ providers)
pip install layerlens[langchain]   # LangChain / LangGraph
```

## The `@trace` Decorator

`@trace(client)` marks a function as the root of a trace. When the function returns (or raises), the complete span tree is serialized and uploaded.

### Using Synchronous Client

```python
from layerlens import Stratix
from layerlens.instrument import trace

client = Stratix()

@trace(client)
def my_agent(query: str):
    # Everything inside here is traced
    return process(query)

my_agent("Hello")
# → Trace uploaded automatically on return
```

### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix
from layerlens.instrument import trace

client = AsyncStratix()

@trace(client)
async def my_agent(query: str):
    return await process(query)

asyncio.run(my_agent("Hello"))
```

### Custom Trace Names

By default the trace is named after the function. Override with the `name` parameter:

```python
@trace(client, name="qa-pipeline")
def run_pipeline(query: str):
    ...
```

## The `span()` Context Manager

Use `span()` inside a traced function to create child spans:

```python
from layerlens.instrument import trace, span

@trace(client)
def my_agent(query: str):
    with span("retrieve", kind="retriever") as s:
        docs = search(query)
        s.output = docs

    with span("generate", kind="llm") as s:
        answer = call_llm(query, docs)
        s.output = answer

    return answer
```

### Span Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `name` | `str` | (required) | Display name for the span |
| `kind` | `str` | `"internal"` | Span type: `internal`, `llm`, `retriever`, `tool`, `chain` |
| `input` | `Any` | `None` | Input data for the span |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata attached to the span |

### Setting Span Data

Inside the `with` block, you can set properties on the span object:

```python
with span("my-step", kind="tool") as s:
    s.input = {"query": query}
    result = do_work(query)
    s.output = result
    s.metadata["custom_key"] = "custom_value"
```

### Nesting Spans

Spans nest automatically — the parent-child relationship is tracked via `contextvars`:

```python
@trace(client)
def my_agent(query: str):
    with span("outer") as outer:
        with span("inner") as inner:
            # inner is a child of outer
            ...
    with span("sibling"):
        # sibling is a child of root, not outer
        ...
```

This produces:

```
my_agent (root)
├── outer
│   └── inner
└── sibling
```

## Span Data Model

Each span captures:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name` | `str` | Span name |
| `span_id` | `str` | Unique identifier (auto-generated) |
| `parent_id` | `str \| None` | Parent span ID |
| `start_time` | `float` | Unix timestamp when span started |
| `end_time` | `float \| None` | Unix timestamp when span ended |
| `status` | `str` | `"ok"` or `"error"` |
| `kind` | `str` | `"internal"`, `"llm"`, `"retriever"`, `"tool"`, `"chain"` |
| `input` | `Any` | Input data (set manually or captured by adapters) |
| `output` | `Any` | Output data |
| `error` | `str \| None` | Error message if status is `"error"` |
| `metadata` | `dict` | Arbitrary metadata (model name, token usage, etc.) |
| `children` | `list` | Child spans |

## Error Handling

Errors are captured automatically. If an exception is raised inside a traced function or span, the span's status is set to `"error"` and the error message is recorded. The exception still propagates normally.

```python
@trace(client)
def my_agent(query: str):
    with span("risky-step") as s:
        raise ValueError("something broke")
    # → span status="error", error="something broke"
    # → trace still uploads with the error recorded
    # → ValueError propagates to caller
```

## Next Steps

- [LLM Providers](providers.md) — Auto-instrument OpenAI, Anthropic, and LiteLLM
- [Agent Frameworks](frameworks.md) — LangChain and LangGraph callback handlers
