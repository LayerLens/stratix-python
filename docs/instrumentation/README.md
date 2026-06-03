# Instrumentation

The `layerlens.instrument` module provides automatic tracing for LLM applications. It captures execution spans — function calls, LLM requests, tool invocations — as a tree structure and uploads them as traces to LayerLens for evaluation.

## How It Works

1. **`@trace(client)`** wraps a function as the root of a trace. When the function completes, the span tree is serialized to JSON and uploaded via `client.traces.upload()`.
2. **`span()`** creates child spans inside a traced function. Spans nest automatically using Python's `contextvars`.
3. **Provider adapters** (OpenAI, Anthropic, LiteLLM) monkey-patch SDK methods to create LLM spans automatically — no code changes needed inside your functions.
4. **Framework adapters** (LangChain, LangGraph) plug in as callback handlers to capture chain/tool/retriever spans from agent frameworks.

## Quick Example

```python
from layerlens import Stratix
from layerlens.instrument import trace, span
from layerlens.instrument.adapters.providers.openai import instrument_openai

client = Stratix()

# Auto-instrument OpenAI — all chat.completions.create calls
# inside a @trace will generate LLM spans automatically
import openai
openai_client = openai.OpenAI()
instrument_openai(openai_client)

@trace(client)
def my_agent(question: str):
    with span("retrieve", kind="retriever") as s:
        docs = search(question)
        s.output = docs

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context: {docs}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

my_agent("What is retrieval-augmented generation?")
```

This produces a trace with three spans:

```
my_agent (root, kind=internal)
├── retrieve (kind=retriever)
└── openai.chat.completions.create (kind=llm, auto-captured)
```

## Guides

- [Quick Start](quickstart.md) — `@trace`, `span()`, and manual instrumentation
- [LLM Providers](providers.md) — Auto-instrument OpenAI, Anthropic, and LiteLLM
- [Agent Frameworks](frameworks.md) — LangChain and LangGraph callback handlers

## Key Concepts

| Concept | Description |
| ------- | ----------- |
| **Trace** | A complete execution tree, rooted at a `@trace`-decorated function |
| **Span** | A single unit of work within a trace (function call, LLM request, tool use) |
| **Kind** | Span type: `internal`, `llm`, `retriever`, `tool`, `chain` |
| **Provider adapter** | Monkey-patches an LLM SDK to emit `llm` spans automatically |
| **Framework adapter** | Callback handler that captures spans from agent frameworks |

## No-Op Safety

All instrumentation is no-op safe:

- Provider adapters pass through to the original SDK method when called outside a `@trace` context
- `span()` returns a dummy context manager when called outside a `@trace` context
- No performance overhead when instrumentation is not active
