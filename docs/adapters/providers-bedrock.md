# AWS Bedrock provider adapter

`layerlens.instrument.adapters.providers.bedrock_adapter.AWSBedrockAdapter`
wraps the `bedrock-runtime` boto3 client. Bedrock is a multi-provider
front: Anthropic, Meta, Cohere, Amazon Titan, AI21, and Mistral models all
flow through the same client interface but with different request and
response body shapes. The adapter detects the provider family from
`modelId` and parses tokens, content, and stop reasons accordingly.

## Install

```bash
pip install 'layerlens[providers-bedrock]'
```

Pulls `boto3>=1.34`.

## Quick start

```python
import boto3
from layerlens.instrument.adapters.providers.bedrock_adapter import AWSBedrockAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="aws_bedrock")
adapter = AWSBedrockAdapter()
adapter.add_sink(sink)
adapter.connect()

client = boto3.client("bedrock-runtime", region_name="us-east-1")
adapter.connect_client(client)

# Either invoke_model or converse — both wrapped.
client.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
)
```

## Wrapped methods

- `invoke_model` — body is JSON, parsed per provider family. Response body is
  wrapped in a `_RereadableBody` so the caller's downstream `body.read()`
  still works.
- `converse` — unified Anthropic-style envelope. Token extraction is uniform.
- `invoke_model_with_response_stream` — emits `model.invoke` immediately with
  `streaming=true`; content extraction during stream consumption is deferred
  to a future PR.
- `converse_stream` — same.

## Provider-family token extraction

| `modelId` prefix | Family | Token fields |
|---|---|---|
| `anthropic.` | anthropic | `usage.input_tokens` / `usage.output_tokens` |
| `meta.` | meta | `prompt_token_count` / `generation_token_count` |
| `cohere.` | cohere | `meta.billed_units.input_tokens` / `output_tokens` |
| `amazon.` | amazon | (no usage in body; tokens come from `Converse` API) |
| `ai21.` | ai21 | (handled via `Converse` API) |
| `mistral.` | mistral | `prompt_tokens` / `completion_tokens` |

## Cost calculation

Uses the `BEDROCK_PRICING` table (separate from OpenAI/Azure tables).
