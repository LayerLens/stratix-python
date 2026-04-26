# Anthropic provider adapter

`layerlens.instrument.adapters.providers.anthropic_adapter.AnthropicAdapter`
instruments the Anthropic Python SDK to emit telemetry on every
`messages.create` and `messages.stream` call.

## Install

```bash
pip install 'layerlens[providers-anthropic]'
```

Pulls `anthropic>=0.30,<1`.

## Quick start

```python
from anthropic import Anthropic
from layerlens.instrument.adapters.providers.anthropic_adapter import AnthropicAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="anthropic")
adapter = AnthropicAdapter()
adapter.add_sink(sink)
adapter.connect()

client = Anthropic()
adapter.connect_client(client)

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=20,
    messages=[{"role": "user", "content": "Hello"}],
)

adapter.disconnect()
sink.close()
```

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | Every `messages.create` (success or failure) and once per stream completion |
| `cost.record` | cross-cutting | Every successful call with token usage |
| `tool.call` | L5a | One per `tool_use` block in the response |
| `policy.violation` | cross-cutting | When the SDK raises (rate limit, invalid input, etc.) |

The `model.invoke` payload includes Anthropic-specific fields:
- `cache_creation_input_tokens` / `cache_read_input_tokens` (when prompt caching is used)
- `parameters.has_system: true` when a system prompt is supplied
- `parameters.tools_count` when tools are passed
- `reasoning_tokens` (Claude extended thinking)

## Streaming

The adapter wraps both `messages.create(stream=True)` and the
`messages.stream()` context manager. A single consolidated `model.invoke`
fires on stream completion, accumulating content from `text_delta` events
and tool input from `input_json_delta` events.

## Cost calculation

Pricing comes from the canonical table — Claude models get the 90% cached-token
discount automatically.

## BYOK

Same pattern as the OpenAI adapter — pass `api_key` to the `Anthropic()` client.
The platform-side BYOK store ships in atlas-app M1.B.
