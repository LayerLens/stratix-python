# OpenAI provider adapter

`layerlens.instrument.adapters.providers.openai_adapter.OpenAIAdapter` instruments
the OpenAI Python SDK to emit telemetry on every chat completion, embedding, or
streaming call.

## Install

```bash
pip install 'layerlens[providers-openai]'
```

Pulls `openai>=1.30,<2`.

## Quick start

```python
from openai import OpenAI
from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="openai")  # ships to atlas-app
adapter = OpenAIAdapter()
adapter._event_sinks.append(sink)
adapter.connect()

client = OpenAI()
adapter.connect_client(client)

# Every call from now on is instrumented.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

adapter.disconnect()
sink.close()
```

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | Every chat completion or embedding call (success or failure). |
| `cost.record` | cross-cutting | Every successful call with token usage in the response. |
| `tool.call` | L5a | One per tool call returned in a chat response. |
| `policy.violation` | cross-cutting | When the OpenAI SDK raises (rate limit, invalid input, etc.). |

The `model.invoke` payload includes:

- `provider`, `model`, `parameters` (temperature, max_tokens, top_p, ...)
- `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `cached_tokens` (when present), `reasoning_tokens` (o1/o3)
- `latency_ms`, `response_id`, `system_fingerprint`, `service_tier`
- `messages` (input) and `output_message` — captured only when
  `CaptureConfig.capture_content` is True (the default).

## Streaming

For streaming responses, the adapter wraps the iterator and accumulates
content + tool-call deltas + final usage. A **single** `model.invoke` is emitted
on stream completion with `metadata.streaming=true`, not one per chunk.

To get token usage for streamed responses, pass
`stream_options={"include_usage": True}` to `client.chat.completions.create`.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Production-light: only L1 + protocol discovery + lifecycle.
adapter = OpenAIAdapter(capture_config=CaptureConfig.minimal())

# Recommended: L1 + L3 + L4a + L5a + L6.
adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())

# Everything (development / debugging).
adapter = OpenAIAdapter(capture_config=CaptureConfig.full())

# Hand-rolled: redact prompt/response content but keep tokens + costs.
adapter = OpenAIAdapter(
    capture_config=CaptureConfig(
        l3_model_metadata=True,
        capture_content=False,
    ),
)
```

## Cost calculation

Costs are computed from the canonical pricing table in
`layerlens.instrument.adapters.providers._base.pricing.PRICING`. The table
hash is matched against `ateam` in CI to prevent drift.

If a model is not in the table the `cost.record` event still fires with
`api_cost_usd: null` and `pricing_unavailable: true`.

## BYOK

The adapter does NOT manage your OpenAI API key. Pass it to the OpenAI client
as you would normally:

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

The platform-side BYOK store (atlas-app `byok_credentials` table, encrypted in
AWS Secrets Manager) is for orgs that want their key managed centrally. In
that flow the SDK fetches the key from atlas-app at startup and passes it to
the OpenAI client. See `docs/adapters/byok.md` for setup.

## Restoring originals

`adapter.disconnect()` restores the original `client.chat.completions.create`
and `client.embeddings.create` methods. After disconnect, the client behaves
exactly as before `connect_client` was called.

## Circuit breaker

If `_stratix.emit()` fails 10 times in a row (transport down, server 5xx
storm), the circuit opens and events are silently dropped for 60 s. After
the cooldown a single attempt is made; success resumes normal flow.
This protects the user's program from a flaky telemetry pipeline.
