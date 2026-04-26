# LiteLLM provider adapter

`layerlens.instrument.adapters.providers.litellm.LiteLLMAdapter` hooks
into LiteLLM's callback system rather than monkey-patching client
methods. This avoids interfering with LiteLLM's own routing, fallback,
and retry behaviour, and lets one adapter cover every provider LiteLLM
supports.

## Install

```bash
pip install 'layerlens[providers-litellm]'
```

Pulls `litellm>=1.40,<2`. The default `pip install layerlens` does
**not** pull `litellm` — adapter modules are lazy-imported only inside
`LiteLLMAdapter.connect`.

## Quick start

```python
import litellm
from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter
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

The legacy flat-file path
`layerlens.instrument.adapters.providers.litellm_adapter` re-exports the
same symbols and continues to work for code pinned to the M1.B port.

## Subpackage layout

```
layerlens/instrument/adapters/providers/litellm/
├── __init__.py    # Public surface: LiteLLMAdapter, LayerLensLiteLLMCallback, ...
├── adapter.py     # LiteLLMAdapter — lifecycle (connect / disconnect / version)
├── callback.py    # LayerLensLiteLLMCallback — sync + async log_*_event hooks
└── routing.py     # detect_provider — model-string → canonical provider name
```

## Provider routing

LiteLLM accepts a `model` argument that may be either an explicit
`provider/model` prefix or a bare model name. The adapter's
`detect_provider` mirrors the LiteLLM dispatcher and normalises the
result to the canonical LayerLens provider name used everywhere else in
the platform.

| Prefix | Provider |
|---|---|
| `openai/` | `openai` |
| `anthropic/` | `anthropic` |
| `azure/` | `azure_openai` |
| `bedrock/` | `aws_bedrock` |
| `vertex_ai/` | `google_vertex` |
| `ollama/` | `ollama` |
| `cohere/` | `cohere` |
| `huggingface/` | `huggingface` |
| `together_ai/` | `together_ai` |
| `groq/` | `groq` |
| (no prefix) | inferred from model name (`gpt-`, `o1`, `o3` → `openai`; `claude-` → `anthropic`; `gemini-` → `google_vertex`; `llama` → `meta`; `mistral` → `mistral`) |

Unrecognised models get `provider="unknown"`. Adding a new prefix
requires an entry in `_PROVIDER_PREFIXES` in
[`routing.py`](../../src/layerlens/instrument/adapters/providers/litellm/routing.py)
plus a corresponding test case in `tests/instrument/adapters/providers/test_litellm.py`.

## Pricing inheritance

LiteLLM does **not** add new entries to the LayerLens pricing manifest
— it consumes the canonical
[`PRICING`](../../src/layerlens/instrument/adapters/providers/_base/pricing.py)
table maintained for the direct provider adapters.

The cost source is resolved in this order on every successful call:

1. **LiteLLM ground truth.** `litellm.completion_cost(model=..., completion_response=...)`
   is called first. If LiteLLM has its own pricing for the model and
   returns a non-`None` USD value, that value is recorded and the
   `cost.record` event is tagged with `cost_source: "litellm"`.
2. **Canonical LayerLens manifest.** If LiteLLM cannot price the call
   (returns `None`, raises, or LiteLLM is not installed), the adapter
   falls through to `_emit_cost_record`, which looks the model up in
   the canonical `PRICING` map. The event payload carries the
   `provider` field set by the routing layer, so cost rollups in the
   dashboard line up across every adapter.

Because the routing layer maps `bedrock/anthropic.claude-3-5-sonnet-…`
to `aws_bedrock`, Bedrock-routed Anthropic calls flow through
`BEDROCK_PRICING` (the model-id-keyed Bedrock table), not the
direct-Anthropic `PRICING` rates. The
`test_completion_emits_invoke_with_correct_provider` parametrised case
asserts this end-to-end.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | Every sync or async completion (success or failure), once per call. |
| `cost.record` | cross-cutting | Every successful call with token usage; sourced from LiteLLM first, then the canonical pricing manifest. |
| `policy.violation` | cross-cutting | When the underlying provider raises (rate limit, content-policy block, network error). |

Streaming completions emit a single consolidated `model.invoke`
(`streaming: true`) when the stream completes — not one per chunk.

## Async support

The same `LayerLensLiteLLMCallback` instance handles both
`litellm.completion(...)` (sync) and `litellm.acompletion(...)` (async).
LiteLLM dispatches the async path through `async_log_success_event` /
`async_log_failure_event` / `async_log_stream_event`, which delegate to
the sync handlers — every callback receives the same kwargs /
response_obj shape.

## Backward-compat alias

`STRATIXLiteLLMCallback` is preserved as an alias for
`LayerLensLiteLLMCallback` so users coming from the `ateam` framework
codebase don't need to rewrite imports immediately. The alias will be
removed in v2.0.
