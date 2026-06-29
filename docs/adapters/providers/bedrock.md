# AWS Bedrock provider adapter

Instruments [AWS Bedrock](https://aws.amazon.com/bedrock/) by wrapping the
`boto3` `bedrock-runtime` client's `invoke_model`, `converse`, and their
streaming variants. The `modelId` prefix selects a family-specific
token/output parser, so a single adapter covers Anthropic, Meta, Cohere,
Amazon, and Mistral models. Non-streaming responses are fully parsed
(messages, output, usage, stop reason); the body is re-materialized so the
caller's single-read `StreamingBody` still works.

## Install

```bash
pip install layerlens[bedrock]
```

Pulls `boto3` for the `bedrock-runtime` client.

## IAM permissions

The IAM principal running your client needs Bedrock runtime invoke
permissions. Grant the actions you actually call:

- `bedrock:InvokeModel` — required for `invoke_model` / `converse`.
- `bedrock:InvokeModelWithResponseStream` — required for
  `invoke_model_with_response_stream` and `converse_stream`.

Cross-region inference profiles (`us.`, `eu.`, `apac.`, `us-gov.` prefixes)
may require the action scoped to the inference-profile ARN as well as the
underlying foundation model. The adapter does not manage credentials — it
wraps whatever `boto3` client you hand it; standard AWS credential resolution
(`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, instance/role credentials) applies.

## Usage

```python
import os, json, boto3
from layerlens.instrument.adapters.providers import BedrockProvider

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

provider = BedrockProvider()
provider.connect(client)   # wraps invoke_model / converse / *_stream

resp = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hi"}],
    }),
)
body = json.loads(resp["body"].read())
print(body.get("content"))
```

## Event surface

- `model.invoke` for each call, named by method: `aws_bedrock.invoke_model`,
  `aws_bedrock.converse`, `aws_bedrock.invoke_model_with_response_stream`, and
  `aws_bedrock.converse_stream`. Payload includes `model`, parsed `messages` /
  `output_message`, `usage`, `family`, `stop_reason`, `response_id` (from the
  AWS `ResponseMetadata.RequestId`), and OTel GenAI semantic-convention
  attributes. Streaming calls emit a `streaming=True` `model.invoke` without
  aggregated body/usage — the `StreamingBody` is single-read and is not
  buffered.
- `cost.record` for each invoke whose response carries usage data, priced
  against `BEDROCK_PRICING`.
- `agent.error` if the underlying boto3 call raises (the error is emitted and
  the exception re-raised).

## Supported model families

The `modelId` prefix maps to a family-specific parser (region/inference-profile
prefixes and full ARNs are unwrapped first):

- `anthropic.*` — Claude
- `meta.*` — Llama
- `cohere.*` — Command
- `amazon.*` — Titan and Nova (Converse-shaped)
- `mistral.*` — Mistral
- `ai21.*` — recognized as a prefix during family classification

## Pricing

Cost is resolved from `BEDROCK_PRICING` (`providers/pricing.py`), with priced
entries for Anthropic Claude 3 (Sonnet/Opus/Haiku), Meta Llama 3.1
(70B/8B), Cohere Command R / R+, and Amazon Nova
(Micro/Lite/Pro/Premier). Region-prefixed inference-profile ids (e.g.
`us.amazon.nova-lite-v1:0`) normalize to the bare model id before lookup.

Be aware: the Mistral and AI21 families are **parsed but not yet priced** —
they have no entries in `BEDROCK_PRICING`, so their `cost.record` cost
resolves to `None`. Token usage is still captured for those families.

## Sample

[`samples/instrument/bedrock/example.py`](../../../samples/instrument/bedrock/example.py)
and [`samples/adapters/providers/bedrock_invoke.py`](../../../samples/adapters/providers/bedrock_invoke.py)

## Compat

- `boto3` (`bedrock-runtime` client)
- Python 3.9+
