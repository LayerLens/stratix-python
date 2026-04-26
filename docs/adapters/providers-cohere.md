# Cohere provider adapter

`layerlens.instrument.adapters.providers.cohere_adapter.CohereAdapter`
instruments the Cohere Python SDK (v5+) for chat (v1 + v2) and embeddings.

## Install

```bash
pip install 'layerlens[providers-cohere]'
```

Pulls `cohere>=5.0,<6`.

## Quick start

```python
import cohere
from layerlens.instrument.adapters.providers.cohere_adapter import CohereAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="cohere")
adapter = CohereAdapter()
adapter.add_sink(sink)
adapter.connect()

client = cohere.Client()
adapter.connect_client(client)

# v1 chat (single message + optional preamble)
response = client.chat(model="command-r-plus", message="Hello", preamble="Be concise.")

# v2 chat (OpenAI-style messages list)
response = client.v2.chat(
    model="command-r-plus",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## What's wrapped

- `client.chat` (v1) — `message` is normalized to a `user` role; optional `preamble` becomes a `system` message at index 0.
- `client.v2.chat` — already OpenAI-style; messages pass through.
- `client.embed` — `meta.billed_units.input_tokens` populates the cost record.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | Every chat or embed call (success or failure). |
| `cost.record` | cross-cutting | Every successful call with billed units. |
| `tool.call` | L5a | One per tool call in the response (v1: `tool_calls[].name/parameters`; v2: `message.tool_calls[].function.{name,arguments}`). |
| `policy.violation` | cross-cutting | When the SDK raises (rate limit, invalid input, etc.). |

## Cost calculation

Pricing is sourced from the canonical `PRICING` table:

| Model | Input | Output |
|---|---|---|
| command-r-plus | $0.003 | $0.015 |
| command-r | $0.0005 | $0.0015 |
| command-r-plus-08-2024 | $0.0025 | $0.01 |
| command-r-08-2024 | $0.00015 | $0.0006 |
| command | $0.001 | $0.002 |
| command-light | $0.0003 | $0.0006 |

Cohere-via-Bedrock models use `BEDROCK_PRICING` instead.

## Streaming

The current adapter wraps non-streaming `chat` and `chat_stream`-style
calls. If you call `client.chat_stream(...)` directly, the underlying
function is not currently wrapped — open an issue if you need it.

## BYOK

Pass `api_key` to `cohere.Client(api_key=...)` as you would normally.
The platform-side BYOK store ships in atlas-app M1.B.
