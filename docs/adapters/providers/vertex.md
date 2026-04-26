# Vertex AI provider adapter (multi-vendor, M3)

`layerlens.instrument.adapters.providers.vertex.VertexAdapter` wraps
`GenerativeModel.generate_content` from the
`google-cloud-aiplatform` SDK (or the legacy `google.generativeai`
package). Unlike the older
`layerlens.instrument.adapters.providers.google_vertex_adapter.GoogleVertexAdapter`
— which is Gemini-only — this adapter supports the three model
families that share Vertex's `generate_content` surface.

## When to use this vs `google_vertex_adapter`

| Use case | Use |
|---|---|
| Gemini-only workload, existing integration | Either; `GoogleVertexAdapter` is the legacy import path. |
| Anthropic Claude on Vertex | **`VertexAdapter`** (this package). |
| Llama on Vertex | **`VertexAdapter`** (this package). |
| Mixed Gemini + non-Gemini in one app | **`VertexAdapter`** (this package). |
| New code, no existing import contract | **`VertexAdapter`**. |

`VertexAdapter` will be the long-term home; `GoogleVertexAdapter` stays
in the package for backwards compatibility.

## Install

```bash
pip install 'layerlens[providers-vertex]'
```

Pulls `google-cloud-aiplatform>=1.50,<2` and `google-auth>=2.23,<3`.

## Quick start

```python
from vertexai.generative_models import GenerativeModel
from layerlens.instrument.adapters.providers.vertex import VertexAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

import vertexai

vertexai.init(project="my-gcp-project", location="us-central1")

sink = HttpEventSink(adapter_name="vertex")
adapter = VertexAdapter()
adapter.add_sink(sink)
adapter.connect()

model = GenerativeModel("gemini-2.5-pro")
adapter.connect_client(model)

response = model.generate_content("Why is the sky blue?")
```

## Vertex-specific behaviour

- **Publisher prefix stripping**: `publishers/anthropic/models/claude-opus-4-6`
  is normalized to `claude-opus-4-6` for pricing lookup; `models/gemini-1.5-pro`
  is normalized to `gemini-1.5-pro`.
- **Vendor classification**: every `model.invoke` event includes a
  `vendor` field — one of `google`, `anthropic`, or `meta` — derived
  from the model identifier.
- **Function calls**: extracted from
  `candidates[0].content.parts[].function_call` and emitted as
  `tool.call` events with the parsed `args` dict.
- **`thoughts_token_count`**: when a model returns reasoning tokens
  (Gemini 2.5 thinking budgets), they populate
  `model.invoke.reasoning_tokens`.
- **`finish_reason`**: enum value name is captured (e.g. `"STOP"`,
  `"MAX_TOKENS"`, `"SAFETY"`).
- **GCP context on every event**: `gcp_project`, `gcp_location`, and
  `credential_source` are stamped on `model.invoke` so traces are
  correlatable with billing dashboards and IAM audit logs.

## Authentication

The adapter does **not** mint Google credentials itself; it defers to
the standard Google credential chain so the adapter behaves identically
to a non-instrumented Vertex client. Two credential sources are
supported.

### 1. Service Account JSON (recommended for prod workloads)

Create a service account with the `roles/aiplatform.user` role, download
the JSON key, and set:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/sa-key.json
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
export GOOGLE_CLOUD_REGION=us-central1   # optional, defaults to us-central1
```

The Vertex SDK picks up the path automatically. The adapter records
`credential_source: service_account_json` on every `model.invoke`
event so you can audit which workloads are using which key. If the
path is set but the file is missing, the adapter records
`credential_source: unknown` (the SDK will raise a clearer error
when the call is actually made).

> **Never commit SA-JSON files to git.** Use a secret manager
> (GCP Secret Manager, AWS Secrets Manager, HashiCorp Vault) and
> mount the file at runtime, or move to short-lived ADC via
> [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
> for production.

### 2. Application Default Credentials (ADC) — fallback

When `GOOGLE_APPLICATION_CREDENTIALS` is **not** set, the SDK falls
back to ADC, which resolves in this order:

1. `gcloud auth application-default login` user credentials
   (`~/.config/gcloud/application_default_credentials.json`).
2. The metadata-server identity attached to the running workload
   (GCE VM, GKE pod with Workload Identity, Cloud Run, Cloud
   Functions, App Engine).

```bash
gcloud auth application-default login    # local dev
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
python your_script.py
```

The adapter records `credential_source: application_default` for ADC
runs. Inside GKE / Cloud Run with Workload Identity bound to a service
account, this is the recommended pattern — no SA-JSON key file ever
touches the workload.

### Project / region resolution

The adapter reads `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, and
`GCP_PROJECT` (in that order) for the project id, and
`GOOGLE_CLOUD_REGION`, `GOOGLE_CLOUD_LOCATION`, `VERTEX_LOCATION` for
the region. Both flow into `model.invoke.gcp_project` and
`model.invoke.gcp_location` when set. The Vertex SDK itself uses
whatever you pass to `vertexai.init()`; the adapter's read is for trace
metadata only.

## Streaming

`generate_content(stream=True)` is wrapped — the adapter accumulates
chunk-level usage and emits exactly one consolidated `model.invoke`
plus one `cost.record` on stream completion. Function calls in
streamed responses follow the same accumulation pattern.

## Cost calculation

`VERTEX_PRICING` (in
`layerlens.instrument.adapters.providers.vertex.pricing`) is the
authoritative table for Vertex-routed invocations. Gemini models get
the 75% cached-token discount per the canonical pricing rules; Claude
on Vertex gets Anthropic's 90% cached-token discount; Llama on Vertex
has no cache discount. If a model is missing from the table the
`cost.record` event sets `api_cost_usd: null` and
`pricing_unavailable: true` rather than silently emitting a zero
cost — fix it by adding the entry to `VERTEX_PRICING` and adjusting
the hash-check baseline in `ateam`.

## Backwards-compatible aliases

For code written against the legacy `STRATIX_*` naming, the
`vertex/__init__.py` exposes:

```python
from layerlens.instrument.adapters.providers.vertex import (
    LayerLensVertexAdapter,        # alias for VertexAdapter
    STRATIX_VERTEX_ADAPTER_CLASS,  # alias for ADAPTER_CLASS
    STRATIX_VERTEX_PRICING,        # alias for VERTEX_PRICING
)
```

All three point at the same objects — these are aliases, not
separate types.

## See also

- Sample: `samples/instrument/providers/vertex/main.py`
- Tests: `tests/instrument/adapters/providers/test_vertex.py`
- Source port: `ateam/stratix/sdk/python/adapters/llm_providers/google_vertex_adapter.py`
