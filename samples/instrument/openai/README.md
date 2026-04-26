# OpenAI adapter sample

> ⚠ **The platform telemetry endpoint (`/api/v1/telemetry/spans`) is not
> live yet.** It lands in atlas-app M1.B (integrations + telemetry-ingest
> Go packages). Until then, the sink will log
> `layerlens.sink.batch_dropped` after the third consecutive failure and
> events are not persisted. The adapter and SDK side are fully functional
> — you can run this sample today against any HTTP server that accepts
> JSON POSTs, including a local ngrok tunnel or the harness in
> `tests/instrument/test_sink_http_e2e.py`.

This sample demonstrates the LayerLens OpenAI provider adapter wrapping a real
OpenAI client. Every chat completion or embedding call is intercepted and
turned into telemetry events shipped to atlas-app.

## What you'll see

Running `python -m samples.instrument.openai.main` produces three events for a
single chat completion:

- `model.invoke` (L3) — the request and response, with parameters, tokens, and
  latency.
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  pricing table for the requested model.
- `tool.call` (L5a, only if the model returned function calls) — one event per
  tool call.

The events are batched and POSTed to
`$LAYERLENS_STRATIX_BASE_URL/telemetry/spans` with `X-API-Key` auth. If no key
is present the sink runs anonymously and the platform may reject the events
depending on org policy.

## Install

```bash
pip install 'layerlens[providers-openai]'
```

The `providers-openai` extra installs `openai>=1.30,<2`. The default
`pip install layerlens` does NOT pull `openai` — that's the lazy-import
guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export OPENAI_API_KEY=sk-...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
python -m samples.instrument.openai.main
```

## Verify telemetry landed

After the sample exits, check the LayerLens dashboard adapter health page —
the `openai` adapter row should show a recent `last_seen` timestamp and a
non-zero invocation count.

## Streaming

To run the sample against a streaming response, modify the `client.chat.completions.create`
call to add `stream=True, stream_options={"include_usage": True}` and iterate
the stream. The adapter's stream wrapper emits a single consolidated
`model.invoke` on stream completion, not one per chunk.
