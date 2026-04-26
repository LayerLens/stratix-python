# Google Vertex AI provider adapter

`layerlens.instrument.adapters.providers.google_vertex_adapter.GoogleVertexAdapter`
wraps `GenerativeModel.generate_content` from either the
`google.generativeai` or `vertexai.generative_models` SDK.

## Install

```bash
pip install 'layerlens[providers-vertex]'
```

Pulls `google-cloud-aiplatform>=1.50,<2`.

## Quick start

```python
from vertexai.generative_models import GenerativeModel
from layerlens.instrument.adapters.providers.google_vertex_adapter import GoogleVertexAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="google_vertex")
adapter = GoogleVertexAdapter()
adapter.add_sink(sink)
adapter.connect()

model = GenerativeModel("gemini-1.5-pro")
adapter.connect_client(model)

response = model.generate_content("Why is the sky blue?")
```

## Vertex-specific behavior

- **`models/` prefix stripping**: `model_name="models/gemini-1.5-pro"` is normalized to
  `gemini-1.5-pro` for pricing-table lookup.
- **Function calls**: extracted from `candidates[0].content.parts[].function_call`
  and emitted as `tool.call` events with the `args` dict.
- **`thoughts_token_count`**: when the model returns reasoning tokens, they
  populate `model.invoke.reasoning_tokens`.
- **`finish_reason`**: enum value name is captured (e.g., `"STOP"`, `"MAX_TOKENS"`).

## Streaming

`generate_content(stream=True)` is wrapped — the adapter accumulates
chunk-level usage and emits one consolidated `model.invoke` on stream
completion. Function calls in streamed responses follow the same accumulation
pattern.

## Cost calculation

Gemini models get the 75% cached-token discount per the canonical pricing table.
