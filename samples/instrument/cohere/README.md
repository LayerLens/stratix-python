# Cohere adapter sample

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

This sample demonstrates the LayerLens Cohere provider adapter wrapping a
real `cohere.Client` (Cohere SDK v5+). The single chat call in `main.py` is
intercepted and turned into telemetry events shipped to atlas-app.

## What you'll see

Running `python -m samples.instrument.cohere.main` makes one call to
`client.chat(...)` against `command-r-plus` (the Cohere **v1 Chat** endpoint
— `client.chat`, not `client.v2.chat`) and produces these events:

- `model.invoke` (L3) — the request and response, with captured parameters
  (model, preamble routed through metadata, etc.), token usage, and latency.
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  pricing table for the requested Cohere model. For models not in the table,
  `api_cost_usd` is `None` and `pricing_unavailable` is `True`.
- `tool.call` (L5a, only if the chat response includes tool calls) — one
  event per tool call. The default prompt ("What is 2 + 2?") does not
  trigger tools, so this event is not emitted in the canonical run.

Events are batched and POSTed to
`$LAYERLENS_STRATIX_BASE_URL/telemetry/spans` with `X-API-Key` auth. If no key
is present the sink runs anonymously and the platform may reject the events
depending on org policy.

## Install

```bash
pip install 'layerlens[providers-cohere]'
```

The `providers-cohere` extra installs the `cohere` Python SDK (v5+). The
default `pip install layerlens` does NOT pull `cohere` — that's the
lazy-import guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export COHERE_API_KEY=...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
python -m samples.instrument.cohere.main
```

Console output prints:

- `Response: <cohere's reply text>` — from `response.text`.
- `Tokens — input: <n>, output: <n>` — from
  `response.meta.billed_units.input_tokens` / `output_tokens` (only when
  `billed_units` is present on the response).
- `Telemetry shipped. Check the LayerLens dashboard adapter health page.`

## Verify telemetry landed

After the sample exits, check the LayerLens dashboard adapter health page —
the `cohere` adapter row should show a recent `last_seen` timestamp and a
non-zero invocation count.

## What this demonstrates

Traced from `main.py` and
`src/layerlens/instrument/adapters/providers/cohere_adapter.py`:

- **Cohere v1 Chat endpoint** — the sample exercises `client.chat(...)`
  (positional Cohere v1 surface). The adapter ALSO wraps `client.v2.chat`
  (Cohere v2) and `client.embed` via method substitution when those
  attributes exist on the connected client, but the sample only invokes v1
  chat directly.
- **Standard `CaptureConfig`** — created via `CaptureConfig.standard()`,
  gating which layers the adapter emits.
- **HTTP sink batching** — `HttpEventSink` with `max_batch=10`,
  `flush_interval_s=1.0`, ships events to `/telemetry/spans`.
- **Cohere-specific response shape** — `response.text` (not OpenAI's
  `choices[0].message.content`), and `response.meta.billed_units` for token
  accounting. The adapter normalizes these into the canonical
  `NormalizedTokenUsage` shape before emission.
- **Pricing fallback** — Cohere pricing reuses the canonical `PRICING`
  table; Cohere-on-Bedrock uses `BEDROCK_PRICING`; unknown models flag
  `pricing_unavailable: true`.

## v1 vs v2

The Cohere SDK exposes both `client.chat` (v1, used by this sample) and
`client.v2.chat` (v2). The adapter wraps both surfaces so callers can switch
between them without losing instrumentation. To run the sample against v2,
swap `client.chat(...)` for `client.v2.chat(...)` and use the v2 request
shape (`messages=[...]` instead of `message=...`).
