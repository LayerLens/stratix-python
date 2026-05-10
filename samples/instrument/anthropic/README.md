# Anthropic adapter sample

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

This sample demonstrates the LayerLens Anthropic provider adapter wrapping a
real `anthropic.Anthropic` client. The single chat call in `main.py` is
intercepted and turned into telemetry events shipped to atlas-app.

## What you'll see

Running `python -m samples.instrument.anthropic.main` makes one call to
`client.messages.create` against `claude-haiku-4-5-20251001` and produces these
events:

- `model.invoke` (L3) — the request and response, with parameters (model,
  max_tokens, system), normalized token usage (`input_tokens` →
  `prompt_tokens`, `output_tokens` → `completion_tokens`), and latency.
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  pricing table for the requested Claude model.
- `tool.call` (L5a, only if the model returns `tool_use` content blocks) —
  one event per tool call. The default prompt ("What is 2 + 2?") does not
  trigger tools, so this event is not emitted in the canonical run.

Events are batched and POSTed to
`$LAYERLENS_STRATIX_BASE_URL/telemetry/spans` with `X-API-Key` auth. If no key
is present the sink runs anonymously and the platform may reject the events
depending on org policy.

## Install

```bash
pip install 'layerlens[providers-anthropic]'
```

The `providers-anthropic` extra installs the `anthropic` Python SDK. The
default `pip install layerlens` does NOT pull `anthropic` — that's the
lazy-import guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
python -m samples.instrument.anthropic.main
```

Console output prints:

- `Response: <claude's reply text>` — extracted from `response.content` text
  blocks (only blocks with `type == "text"`).
- `Tokens — input: <n>, output: <n>` — from `response.usage.input_tokens` /
  `output_tokens`.
- `Telemetry shipped. Check the LayerLens dashboard adapter health page.`

## Verify telemetry landed

After the sample exits, check the LayerLens dashboard adapter health page —
the `anthropic` adapter row should show a recent `last_seen` timestamp and a
non-zero invocation count.

## What this demonstrates

Traced from `main.py` and
`src/layerlens/instrument/adapters/providers/anthropic_adapter.py`:

- **Messages API instrumentation** — `connect_client` wraps
  `client.messages.create` (and `client.messages.stream` if present) via
  method substitution; originals are restored on `disconnect()`.
- **Standard `CaptureConfig`** — created via `CaptureConfig.standard()`,
  gating which layers the adapter emits.
- **HTTP sink batching** — `HttpEventSink` with `max_batch=10`,
  `flush_interval_s=1.0`, ships events to `/telemetry/spans`.
- **Claude model targeting** — the sample pins
  `claude-haiku-4-5-20251001` to exercise the current generation of Claude
  Haiku; pricing for this model flows through `_emit_cost_record`.
- **Provider-specific fields** — Anthropic's `system` kwarg is captured into
  parameters; `response.content` is iterated as typed blocks (text vs.
  tool_use) rather than a flat string.

## Streaming

To run the sample against a streaming response, switch
`client.messages.create(...)` to `client.messages.stream(...)` and iterate the
context manager. The adapter's stream wrapper accumulates content + tool-call
deltas and emits a single consolidated `model.invoke` on stream completion,
not one per chunk.
