# Mistral AI provider adapter

`layerlens.instrument.adapters.providers.mistral_adapter.MistralAdapter`
instruments the `mistralai` v1 SDK for `chat.complete`, `chat.stream`,
and `embeddings.create`.

## Install

```bash
pip install 'layerlens[providers-mistral]'
```

Pulls `mistralai>=1.0,<2`.

## Quick start

```python
from mistralai import Mistral
from layerlens.instrument.adapters.providers.mistral_adapter import MistralAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="mistral")
adapter = MistralAdapter()
adapter.add_sink(sink)
adapter.connect()

client = Mistral(api_key="...")
adapter.connect_client(client)

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## What's wrapped

- `client.chat.complete` — synchronous chat (OpenAI-shape response).
- `client.chat.stream` — streaming wrapper accumulates content + tool-call deltas; emits **one** consolidated `model.invoke` on iterator exhaustion.
- `client.embeddings.create` — embedding telemetry.

## Events emitted

Same set as OpenAI: `model.invoke`, `cost.record`, `tool.call`, `policy.violation`.

The streaming path emits a single `model.invoke` with `metadata.streaming=true`
on completion, not per chunk.

## Cost calculation

| Model | Input | Output |
|---|---|---|
| mistral-large / mistral-large-latest | $0.002 | $0.006 |
| mistral-small / mistral-small-latest | $0.0002 | $0.0006 |
| mistral-medium | $0.0027 | $0.0081 |
| open-mistral-7b | $0.00025 | $0.00025 |
| open-mixtral-8x7b | $0.0007 | $0.0007 |
| open-mixtral-8x22b | $0.002 | $0.006 |

Mistral-via-Bedrock uses `BEDROCK_PRICING`.

## BYOK

Pass `api_key` to `Mistral(api_key=...)` as normal. The platform-side
BYOK store ships in atlas-app M1.B.
