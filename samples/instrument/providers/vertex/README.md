# Vertex AI adapter sample

> The platform telemetry endpoint (`/api/v1/telemetry/spans`) is not
> live yet. It lands in atlas-app M1.B (integrations + telemetry-ingest
> Go packages). Until then, the sink will log
> `layerlens.sink.batch_dropped` after the third consecutive failure
> and events are not persisted. The adapter and SDK side are fully
> functional — you can run this sample today against any HTTP server
> that accepts JSON POSTs.

This sample demonstrates the LayerLens Vertex AI provider adapter
wrapping a real (or stubbed) `GenerativeModel` client. Every
`generate_content` call is intercepted and turned into telemetry events
shipped to atlas-app.

The adapter supports all three model families that share Vertex's
`generate_content` surface:

- Gemini (e.g. `gemini-2.5-pro`, `gemini-1.5-flash`)
- Anthropic on Vertex (`publishers/anthropic/models/claude-opus-4-6`)
- Llama on Vertex (`publishers/meta/models/llama-3.3-70b-instruct-maas`)

## What you'll see

Running `python -m samples.instrument.providers.vertex.main` produces
two events for a single completion:

- `model.invoke` (L3) — request and response with parameters, tokens,
  finish reason, vendor classification, GCP project / location, and
  credential source (`service_account_json`, `application_default`,
  or `unknown`).
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  Vertex pricing table for the requested model (per-1K-token rates with
  Google's 75% cached-token discount applied to Gemini models).
- `tool.call` (L5a, only if the model returned function calls) — one
  event per function call, with the parsed `args` dict.

Events are batched and POSTed to
`$LAYERLENS_STRATIX_BASE_URL/telemetry/spans` with `X-API-Key` auth.

## Install

```bash
pip install 'layerlens[providers-vertex]'
```

The `providers-vertex` extra installs `google-cloud-aiplatform>=1.50,<2`
and `google-auth>=2.23,<3`. The default `pip install layerlens` does
NOT pull either — that's the lazy-import guarantee tested by
`tests/instrument/test_lazy_imports.py`.

## Run modes

### Mock mode (no credentials, no Google SDK)

Always succeeds. Useful as an end-to-end smoke test of the adapter
wiring on a fresh checkout:

```bash
LAYERLENS_VERTEX_SAMPLE_MODE=mock \
  python -m samples.instrument.providers.vertex.main
```

The sample swaps in a stubbed `GenerativeModel` whose
`generate_content` returns a synthetic `GenerateContentResponse`
shaped exactly like the real one (parts, candidates, usage_metadata).
The adapter cannot tell the difference — every event is emitted as if
a real call ran.

### Live mode (Service Account JSON)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/sa-key.json
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
export GOOGLE_CLOUD_REGION=us-central1
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
python -m samples.instrument.providers.vertex.main
```

The Vertex SDK auto-detects the SA-JSON path. The adapter reads the
same env var to record the credential source on each `model.invoke`
event so you can audit which workloads use which key.

### Live mode (Application Default Credentials)

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
python -m samples.instrument.providers.vertex.main
```

When `GOOGLE_APPLICATION_CREDENTIALS` is unset, the SDK falls back to
ADC. The adapter records `credential_source: application_default` on
every event.

### Run a non-Gemini model

```bash
export LAYERLENS_VERTEX_MODEL=publishers/anthropic/models/claude-opus-4-6
python -m samples.instrument.providers.vertex.main
```

The adapter strips the `publishers/<vendor>/models/` prefix before
pricing lookup, so the resulting `cost.record` event uses the canonical
`claude-opus-4-6` rate from `VERTEX_PRICING`.

## Verify telemetry landed

After the sample exits, check the LayerLens dashboard adapter health
page — the `vertex` adapter row should show a recent `last_seen`
timestamp and a non-zero invocation count.

## Streaming

To run the sample against a streaming response, modify the
`client.generate_content` call to add `stream=True` and iterate the
result. The adapter's stream wrapper accumulates chunk-level usage and
emits exactly one consolidated `model.invoke` plus one `cost.record`
on stream completion — not one per chunk.
