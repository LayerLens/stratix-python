# Azure OpenAI adapter sample

> The platform telemetry endpoint (`/api/v1/telemetry/spans`) is not live
> yet — it lands in atlas-app M1.B. Until then, the sink will log
> `layerlens.sink.batch_dropped` after the third consecutive failure and
> events are not persisted. The adapter and SDK side are fully
> functional — you can run this sample today against any HTTP server
> that accepts JSON POSTs.

This sample demonstrates the LayerLens Azure OpenAI provider adapter
wrapping a real `AzureOpenAI` client. Every chat completion (and
embedding) call is intercepted and turned into telemetry events shipped
to atlas-app. The adapter additionally captures Azure-specific
metadata — the sanitized endpoint, the API version, and the deployment
name — and attaches it to every event.

## What you'll see

Running the sample produces three events for a single chat completion:

- `model.invoke` (L3) — the request and response, with parameters,
  tokens, latency, the Azure `azure_endpoint`, the `api_version`, and
  the deployment as `model`.
- `cost.record` (cross-cutting) — the API cost in USD computed from
  the **Azure** pricing table (`AZURE_PRICING`), which differs from
  the public OpenAI pricing.
- `tool.call` (L5a, only if the model returned function calls) — one
  event per tool call.

## Install

```bash
pip install 'layerlens[providers-azure-openai]'
```

The `providers-azure-openai` extra installs `openai>=1.30,<2`. The
default `pip install layerlens` does NOT pull `openai` — that's the
lazy-import guarantee enforced by `tests/instrument/test_lazy_imports.py`.

## Run against real Azure

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-08-01-preview
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-prod   # NOT the base model name
export LAYERLENS_STRATIX_API_KEY=ll-...      # optional
python -m samples.instrument.azure_openai.main
```

`AZURE_OPENAI_DEPLOYMENT` is the **deployment name** you configured in
the Azure portal (Azure OpenAI Studio → Deployments). It is what you
pass as `model=` to `client.chat.completions.create` — Azure routes the
deployment name to the underlying base model (e.g. `gpt-4o-2024-08-06`).

## Run with the offline mock fixture

For CI and smoke tests where you don't want to provision an Azure
resource, set `LAYERLENS_SAMPLE_MOCK=1`. The sample installs an
`httpx.MockTransport` on the AzureOpenAI client only — no global
patching, no `respx`, no leakage into other HTTP clients in the
process (so the real LayerLens HttpEventSink still fires normally):

```bash
pip install 'layerlens[providers-azure-openai]'
PYTHONPATH=. LAYERLENS_SAMPLE_MOCK=1 python samples/instrument/azure_openai/main.py
```

The mock returns a deterministic chat completion (`"2 + 2 = 4"`, 14
prompt tokens / 6 completion tokens) so the adapter pipeline can be
exercised end-to-end without a live Azure endpoint or any network call.
The mock honours `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_VERSION`
if set; otherwise it defaults to a placeholder Azure URL.

## Endpoint sanitization

The adapter strips query strings from the captured `azure_endpoint`
before any event is emitted. If the URL ever contains an `api-key=...`
query (some legacy SDK paths attach it), that secret will NOT be sent
to the LayerLens telemetry pipeline.

## Pricing

`cost.record` uses the `AZURE_PRICING` table in
`layerlens.instrument.adapters.providers._base.pricing`. Azure prices
are typically 10% above public OpenAI rates and are checked into the
canonical pricing manifest (hash-matched with `ateam` in CI).

If a deployment maps to a base model that is not in `AZURE_PRICING`,
the `cost.record` event still fires with `api_cost_usd: null` and
`pricing_unavailable: true`.

## Streaming

To run the sample against a streaming response, modify the
`client.chat.completions.create` call to add
`stream=True, stream_options={"include_usage": True}` and iterate the
stream. Streaming wrapping is shared with the OpenAI adapter via the
`openai` SDK — a single consolidated `model.invoke` is emitted on
stream completion, not one per chunk.
