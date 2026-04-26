# LiteLLM provider adapter

`layerlens.instrument.adapters.providers.litellm_adapter.LiteLLMAdapter`
hooks into LiteLLM's callback system rather than monkey-patching client
methods. This avoids interfering with LiteLLM's own routing, fallback, and
retry behavior.

## Install

```bash
pip install 'layerlens[providers-litellm]'
```

Pulls `litellm>=1.40,<2`.

## Quick start

```python
import litellm
from layerlens.instrument.adapters.providers.litellm_adapter import LiteLLMAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="litellm")
adapter = LiteLLMAdapter()
adapter.add_sink(sink)
adapter.connect()  # registers the callback with litellm.callbacks

# No connect_client needed — the callback is module-global.
litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
)

adapter.disconnect()  # removes the callback
```

## Provider auto-detection

The adapter parses LiteLLM model strings and routes the `provider` field of
each event to the underlying provider name:

| Prefix | Provider |
|---|---|
| `openai/` | `openai` |
| `anthropic/` | `anthropic` |
| `azure/` | `azure_openai` |
| `bedrock/` | `aws_bedrock` |
| `vertex_ai/` | `google_vertex` |
| `ollama/` | `ollama` |
| `cohere/` | `cohere` |
| `groq/` | `groq` |
| (no prefix) | inferred from model name (`gpt-`, `claude-`, `gemini-`, ...) |

Unrecognized models get `provider="unknown"`.

## Cost calculation

Cost is sourced in this order:
1. LiteLLM's own `litellm.completion_cost(...)` — if it returns a non-None value,
   it's used and the event is tagged `cost_source: "litellm"`.
2. The canonical LayerLens pricing table for the resolved provider.

## Backward-compat alias

`STRATIXLiteLLMCallback` is preserved as an alias for `LayerLensLiteLLMCallback`
so users coming from the `ateam` framework codebase don't need to rewrite
imports immediately. The alias will be removed in v2.0.
