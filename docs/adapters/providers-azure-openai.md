# Azure OpenAI provider adapter

`layerlens.instrument.adapters.providers.azure_openai_adapter.AzureOpenAIAdapter`
uses the same `openai` SDK as the OpenAI adapter but captures Azure-specific
metadata (deployment, endpoint, region, api-version) and uses the Azure
pricing table.

## Install

```bash
pip install 'layerlens[providers-azure-openai]'
```

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

client.chat.completions.create(model="my-deployment", messages=[...])
```

## Azure-specific behavior

- **Endpoint sanitization**: query strings are stripped from the captured
  `azure_endpoint` to prevent token leakage if the URL ever contains an
  `api-key` query param.
- **Pricing**: cost calculations use `AZURE_PRICING` (different rates than
  OpenAI's public API).
- **api-version**: read from `client._api_version` or the `api-version` key of
  `client._custom_query` and surfaced in every `model.invoke`.

## Events emitted

Same set as OpenAI: `model.invoke`, `cost.record`, `tool.call`, `policy.violation`.
