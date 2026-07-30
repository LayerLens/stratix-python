# Azure OpenAI provider adapter

Instruments [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
calls via monkey-patching the OpenAI Python SDK's `AzureOpenAI` client. The
adapter subclasses `OpenAIProvider` — the Azure SDK exposes the same
`chat.completions.create` / `responses.create` / `embeddings.create` surface,
so extraction and patch targets are reused, with Azure-specific response
metadata (`azure_api_version`, `azure_deployment`, scrubbed `azure_endpoint`)
and Azure pricing layered on top.

## Install

```bash
pip install layerlens[azure]
```

Pulls the OpenAI Python SDK (`AzureOpenAI` lives in the `openai` package).

## Authentication

The adapter does not manage Azure auth — set up the `AzureOpenAI` client as you
normally would, then wrap it. Azure OpenAI requires three pieces beyond a plain
OpenAI client:

1. **`azure_endpoint`** — your resource endpoint, e.g.
   `https://<resource>.openai.azure.com`. Set it explicitly or via
   `AZURE_OPENAI_ENDPOINT`.
2. **API version** — passed as `api_version`, commonly sourced from
   `OPENAI_API_VERSION` (e.g. `2024-10-21`).
3. **Deployment name** — on Azure, the `model` argument is the *deployment*
   name you created in the Azure portal, not the base model id.

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<key>"
export OPENAI_API_VERSION="2024-10-21"
```

On `connect()` the adapter reads the client's base URL, strips any query string
(so `api-key` is never logged), and surfaces it as `azure_endpoint` on every
`model.invoke` event.

## Usage

```python
import os
from openai import AzureOpenAI
from layerlens.instrument.adapters.providers import AzureOpenAIProvider

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("OPENAI_API_VERSION", "2024-10-21"),
)

provider = AzureOpenAIProvider()
provider.connect(client)   # patches chat.completions / responses / embeddings

resp = client.chat.completions.create(
    model="<your-deployment-name>",   # Azure deployment, not the base model id
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
```

## Event surface

- `model.invoke` for every `chat.completions.create`, `responses.create`, and
  `embeddings.create` call. Payload includes `model`, `usage`, `finish_reason`,
  and Azure metadata (`azure_api_version`, `azure_deployment` when the SDK
  attaches them; `azure_endpoint` from the scrubbed resource URL).
- `tool.call` per function/tool call surfaced in the response.
- `cost.record` for each invoke whose response carries usage data, priced
  against `AZURE_PRICING`.
- Streaming (`stream=True`) is handled by the shared OpenAI streaming path —
  chunks are aggregated and emitted as a single `model.invoke` when the stream
  ends.

## Pricing

Azure list prices differ from OpenAI's, so the adapter uses a dedicated
`AZURE_PRICING` table (`providers/pricing.py`). Rates are per 1K tokens:

| Model           | Input    | Output   |
|-----------------|----------|----------|
| `gpt-4o`        | 0.00275  | 0.011    |
| `gpt-4o-mini`   | 0.000165 | 0.00066  |
| `gpt-4-turbo`   | 0.011    | 0.033    |
| `gpt-4`         | 0.033    | 0.066    |
| `gpt-35-turbo`  | 0.00055  | 0.00165  |

Cost is keyed on the model/deployment id; deployments mapped to a base model
not in the table resolve to no cost.

## Sample

[`samples/instrument/azure_openai/example.py`](../../../samples/instrument/azure_openai/example.py)
and [`samples/adapters/providers/azure_openai.py`](../../../samples/adapters/providers/azure_openai.py)

## Compat

- OpenAI Python SDK with `AzureOpenAI` (v1.x)
- Python 3.9+
