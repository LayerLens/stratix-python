# Azure OpenAI provider adapter

`layerlens.instrument.adapters.providers.azure_openai_adapter.AzureOpenAIAdapter`
wraps the `openai` Python SDK's `AzureOpenAI` client to instrument every
chat completion, embedding, and (via the shared OpenAI streaming wrapper)
streaming call against an Azure OpenAI Service resource. It emits the
same events as the OpenAI adapter plus Azure-specific metadata
(`azure_endpoint`, `api_version`, deployment as `model`) and bills against
the Azure pricing table rather than the public OpenAI rates.

Ported from `ateam/stratix/sdk/python/adapters/llm_providers/azure_openai_adapter.py`.

## Install

```bash
pip install 'layerlens[providers-azure-openai]'
```

Pulls `openai>=1.30,<2`. The default `pip install layerlens` does NOT
pull the `openai` SDK — that's the lazy-import guarantee enforced by
`tests/instrument/test_lazy_imports.py`.

## Quick start

```python
from openai import AzureOpenAI
from layerlens.instrument.adapters.providers.azure_openai_adapter import AzureOpenAIAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="azure_openai")
adapter = AzureOpenAIAdapter()
adapter.add_sink(sink)
adapter.connect()

client = AzureOpenAI(
    api_key="...",
    api_version="2024-08-01-preview",
    azure_endpoint="https://my-resource.openai.azure.com/",
)
adapter.connect_client(client)

response = client.chat.completions.create(
    model="gpt-4o-prod",  # Azure: this is the DEPLOYMENT name
    messages=[{"role": "user", "content": "Hello"}],
)

adapter.disconnect()
sink.close()
```

## Azure deployment + endpoint setup

Azure OpenAI requires three coordinates that have no analogue on the
public OpenAI API. Configure them once in the Azure portal and pass
them to the `AzureOpenAI` client.

### 1. Resource and endpoint

In the [Azure portal](https://portal.azure.com), create a resource of
type **Azure OpenAI** in a region that has the model you want (e.g.
`eastus`, `swedencentral`, `westeurope` for `gpt-4o`). Once created:

- **Endpoint** — copy from the resource's _Keys and Endpoint_ blade.
  Format: `https://<resource-name>.openai.azure.com/`. Pass as
  `azure_endpoint=...` to the SDK and as `AZURE_OPENAI_ENDPOINT` to the
  sample. The adapter strips any query string before surfacing it on
  events, so a leaked `api-key=...` query param will NEVER reach the
  LayerLens telemetry pipeline.
- **API key** — also on the _Keys and Endpoint_ blade. Pass as
  `api_key=...` to the SDK and as `AZURE_OPENAI_API_KEY` to the
  sample. The adapter does not handle this — the OpenAI SDK injects it
  as the `api-key` HTTP header.

The adapter never logs, persists, or transmits the api-key. It only
captures the endpoint URL (sanitized).

### 2. API version

Azure pins each request to a specific REST API version (e.g.
`2024-08-01-preview`, `2024-10-21`, `2024-12-01-preview`). Pass it as
`api_version=...` to the SDK constructor. The adapter reads it back via
`client._api_version` (and falls back to `client._custom_query["api-version"]`
for older SDK paths) and surfaces it on every `model.invoke` and
`cost.record` event so you can correlate behaviour changes to API
version bumps.

The latest stable API versions are listed at
[Azure OpenAI REST API reference — versioning](https://learn.microsoft.com/azure/ai-services/openai/reference#api-specs).

### 3. Deployment

Azure decouples the **deployment name** from the **base model**. In
the portal under _Azure OpenAI Studio → Deployments_, you create a
deployment that maps a chosen base model (e.g. `gpt-4o-2024-08-06`) to
a name of your choice (e.g. `gpt-4o-prod`, `gpt-4o-eu-west`).

When you call `client.chat.completions.create(model=..., ...)`, the
`model=` argument MUST be the deployment name, not the base model name:

```python
client.chat.completions.create(
    model="gpt-4o-prod",   # deployment name from the Azure portal
    messages=[...],
)
```

The adapter records the deployment as `model` on every event. The
underlying base model returned in the response (e.g.
`gpt-4o-2024-08-06`) is recorded separately under
`response_model` so you can audit which base served each call.

### Authentication alternatives

The sample uses the static API key for simplicity. In production,
prefer **Azure AD / managed identity** auth via `azure-identity`:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)
client = AzureOpenAI(
    azure_ad_token_provider=token_provider,
    api_version="2024-08-01-preview",
    azure_endpoint="https://my-resource.openai.azure.com/",
)
```

The adapter is auth-agnostic — both flows wrap the same client object.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `model.invoke` | L3 | Every chat completion or embedding call (success or failure). |
| `cost.record` | cross-cutting | Every successful call with token usage in the response. |
| `tool.call` | L5a | One per tool call returned in a chat response. |
| `policy.violation` | cross-cutting | When the SDK raises (rate limit, invalid input, content filter, etc.). |

The `model.invoke` payload includes:

- `provider="azure_openai"`, `model` (the deployment),
  `parameters` (temperature, max_tokens, top_p, ...)
- `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `cached_tokens` (when present), `reasoning_tokens` (o1/o3 deployments)
- `latency_ms`, `response_id`, `system_fingerprint`,
  `response_model` (the underlying base model returned by Azure)
- `azure_endpoint` (sanitized), `api_version`
- `messages` (input) and `output_message` — captured only when
  `CaptureConfig.capture_content` is True (the default).

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Production-light: only L1 + protocol discovery + lifecycle.
adapter = AzureOpenAIAdapter(capture_config=CaptureConfig.minimal())

# Recommended: L1 + L3 + L4a + L5a + L6.
adapter = AzureOpenAIAdapter(capture_config=CaptureConfig.standard())

# Everything (development / debugging).
adapter = AzureOpenAIAdapter(capture_config=CaptureConfig.full())

# Hand-rolled: redact prompt/response content but keep tokens + costs.
adapter = AzureOpenAIAdapter(
    capture_config=CaptureConfig(
        l3_model_metadata=True,
        capture_content=False,
    ),
)
```

## Cost calculation

Costs are computed from the canonical `AZURE_PRICING` table in
`layerlens.instrument.adapters.providers._base.pricing`. Azure rates
are typically 10% above the public OpenAI rates. The table hash is
matched against `ateam` in CI to prevent drift; if the source
`ateam/stratix/sdk/python/adapters/llm_providers/pricing.py` adds a new
Azure entry, the same row must land in this repo's table in the same
PR.

If a deployment maps to a base model that is not in `AZURE_PRICING`,
the `cost.record` event still fires with `api_cost_usd: null` and
`pricing_unavailable: true`.

## Endpoint sanitization

The adapter passes the captured endpoint through `urlparse` /
`urlunparse` and discards the query string and fragment **before** any
event is emitted. If the SDK ever attaches the api-key as a query
parameter (some legacy auth paths do), the secret will not reach the
LayerLens telemetry pipeline.

```python
# What the client gives us
"https://my-resource.openai.azure.com/?api-key=SECRET"

# What the adapter emits
"https://my-resource.openai.azure.com/"
```

## BYOK

The adapter does NOT manage your Azure OpenAI api-key, AAD token, or
managed identity. Pass them to the `AzureOpenAI` client as you would
normally. The platform-side BYOK store (`atlas-app` `byok_credentials`
table, encrypted in AWS Secrets Manager) is for orgs that want their
key managed centrally; in that flow the SDK fetches the key at startup
and passes it to the `AzureOpenAI` constructor. See
`docs/adapters/byok.md` for setup.

## Restoring originals

`adapter.disconnect()` restores the original
`client.chat.completions.create` and `client.embeddings.create`
methods. After disconnect, the client behaves exactly as before
`connect_client` was called.

## Circuit breaker

If `_stratix.emit()` fails 10 times in a row (transport down, server
5xx storm), the circuit opens and events are silently dropped for 60s.
After the cooldown a single attempt is made; success resumes normal
flow. This protects the user's program from a flaky telemetry pipeline.
