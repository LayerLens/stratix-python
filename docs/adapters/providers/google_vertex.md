# Google Vertex AI provider adapter

Instruments [Google Vertex AI](https://cloud.google.com/vertex-ai) /
[google-generativeai](https://github.com/google-gemini/generative-ai-python)
`GenerativeModel` calls via monkey-patching. Captures Gemini token usage
(including `reasoning_tokens` from extended thinking), function calls,
streaming responses, `finish_reason`, and `response_id`.

## Install

```bash
pip install layerlens[providers-vertex]
```

Pulls `google-cloud-aiplatform>=1.38`. The `google-vertex` extra is kept as
an alias for prior installs.

## Authentication

The Vertex SDK authenticates with Google Cloud through one of two paths.
The adapter doesn't manage auth itself — set up the SDK as you would
normally, then wrap the `GenerativeModel` instance.

### Option A — Service Account JSON

Best for CI, containers, and any environment where ADC isn't available.

1. In the GCP console, create a service account with the
   `roles/aiplatform.user` IAM role.
2. Download the key as JSON.
3. Set `GOOGLE_APPLICATION_CREDENTIALS` to the file path:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

The Vertex SDK picks this up automatically.

### Option B — Application Default Credentials (ADC)

Best for local dev on a machine with `gcloud` installed.

```bash
gcloud auth application-default login
gcloud config set project your-project-id
```

ADC is also what runs by default inside Google Cloud (Cloud Run, GKE, GCE)
without any extra setup — the workload identity attached to the resource
provides credentials.

## Usage

```python
import vertexai
from vertexai.generative_models import GenerativeModel
from layerlens.instrument.adapters.providers import GoogleVertexProvider

vertexai.init(project="your-project-id", location="us-central1")
model = GenerativeModel("gemini-2.5-pro")

provider = GoogleVertexProvider()
provider.connect(model)             # monkey-patches generate_content + _async

response = model.generate_content("Hello!")
print(response.text)
```

`provider.connect(model)` captures the model id from `model_name` (stripping
the `models/` prefix when present) so cost-record events resolve against
the canonical pricing manifest entry.

## Event surface

- `model.invoke` for every `generate_content` / `generate_content_async` call.
  Payload includes `model`, `usage` (with `reasoning_tokens` when Gemini
  returns `thoughts_token_count`), `finish_reason` (enum name), and
  `response_id` when the SDK exposes one.
- `tool.call` per function call surfaced in `candidates[0].content.parts`.
- `cost.record` for each invoke whose response carries usage data;
  Gemini pricing is vendored in `pricing.py`.
- Streaming: the adapter wraps the iterator and emits a single aggregated
  `model.invoke` when the stream ends (via `_AggregatedVertexResponse`).

## Supported model families

Gemini natively. The same `GenerativeModel` surface is used for
Anthropic-on-Vertex and Llama-on-Vertex models — the adapter wraps the SDK
call boundary so any model id valid for `GenerativeModel` flows through it.
Pricing entries for non-Gemini families can be added to `PRICING` /
`pricing_table` overrides as needed.

## Sample

[`samples/instrument/google_vertex/example.py`](../../../samples/instrument/google_vertex/example.py)

## Compat

- `google-cloud-aiplatform>=1.38`
- Python 3.9+
