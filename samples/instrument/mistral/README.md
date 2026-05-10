# Mistral adapter sample

> WARNING: The platform telemetry endpoint (`/api/v1/telemetry/spans`) is not
> live yet. It lands in atlas-app M1.B (integrations + telemetry-ingest
> Go packages). Until then, the sink will log
> `layerlens.sink.batch_dropped` after the third consecutive failure and
> events are not persisted. The adapter and SDK side are fully functional
> — you can run this sample today against any HTTP server that accepts
> JSON POSTs, including a local ngrok tunnel or the harness in
> `tests/instrument/test_sink_http_e2e.py`.

> NOTE: This sample predates PR #154's typed-events migration. The adapter
> on this branch still emits dict-shaped events via
> `emit_dict_event(...)`. Updated patterns will land when this branch
> rebases on the typed-events foundation post-merge.

This sample demonstrates the LayerLens Mistral provider adapter wrapping a
real `mistralai.Mistral` client (the **native** Mistral Python SDK v1+, NOT
the OpenAI-compatible REST shim). The single chat call in `main.py` is
intercepted and turned into telemetry events shipped to atlas-app.

## What you'll see

Running `python -m samples.instrument.mistral.main` makes one call to
`client.chat.complete(...)` against `mistral-small-latest` and produces these
events:

- `model.invoke` (L3) — the request and response, with captured parameters
  (model, max_tokens, temperature, top_p, random_seed, response_format,
  tool_choice, safe_prompt), normalized token usage, and latency.
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  pricing table for the requested Mistral model.
- `tool.call` (L5a, only if the response includes `tool_calls`) — one event
  per tool call. The default prompt ("What is 2 + 2?") does not trigger
  tools, so this event is not emitted in the canonical run.

Events are batched and POSTed to
`$LAYERLENS_STRATIX_BASE_URL/telemetry/spans` with `X-API-Key` auth. If no key
is present the sink runs anonymously and the platform may reject the events
depending on org policy.

## Install

```bash
pip install 'layerlens[providers-mistral]'
```

The `providers-mistral` extra installs the `mistralai` Python SDK (v1+). The
default `pip install layerlens` does NOT pull `mistralai` — that's the
lazy-import guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export MISTRAL_API_KEY=...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
python -m samples.instrument.mistral.main
```

Console output prints:

- `Response: <mistral's reply text>` — from
  `response.choices[0].message.content`.
- `Tokens — prompt: <n>, completion: <n>, total: <n>` — from `response.usage`
  when present.
- `Telemetry shipped. Check the LayerLens dashboard adapter health page.`

## Verify telemetry landed

After the sample exits, check the LayerLens dashboard adapter health page —
the `mistral` adapter row should show a recent `last_seen` timestamp and a
non-zero invocation count.

## What this demonstrates

Traced from `main.py` and
`src/layerlens/instrument/adapters/providers/mistral_adapter.py`:

- **Native Mistral SDK** — the sample uses `mistralai.Mistral`, which is the
  official Mistral Python client. This is **distinct from** Mistral's
  OpenAI-compatible REST endpoint — that would be exercised by the OpenAI
  adapter against a custom base URL, not by `MistralAdapter`. The wrapped
  surface is `client.chat.complete` and `client.chat.stream` (plus
  `client.embeddings.create` when present on the client).
- **Standard `CaptureConfig`** — created via `CaptureConfig.standard()`,
  gating which layers the adapter emits.
- **HTTP sink batching** — `HttpEventSink` with `max_batch=10`,
  `flush_interval_s=1.0`, ships events to `/telemetry/spans`.
- **Mistral-specific parameters** — `random_seed`, `response_format`,
  `safe_prompt`, and `tool_choice` are part of the captured parameter set;
  these are Mistral-native kwargs that don't exist on the OpenAI surface.
- **Explicit `api_key=` argument** — `Mistral(api_key=os.environ["MISTRAL_API_KEY"])`
  is constructed with the key passed explicitly (the SDK does not pick it up
  from the environment automatically the way `openai.OpenAI()` does).

## Streaming

To run the sample against a streaming response, switch
`client.chat.complete(...)` to `client.chat.stream(...)` and iterate the
returned stream. The adapter's stream wrapper accumulates content + tool-call
deltas and emits a single consolidated `model.invoke` on stream completion
(see `_emit_consolidated` in `mistral_adapter.py`), not one event per chunk.
