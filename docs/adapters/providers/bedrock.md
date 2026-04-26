# AWS Bedrock provider adapter

`layerlens.instrument.adapters.providers.bedrock.AWSBedrockAdapter` wraps the
`bedrock-runtime` boto3 client. Bedrock is a multi-provider front: Anthropic,
Meta, Cohere, Amazon Titan, AI21, and Mistral models all flow through the
same client interface but with different request and response body shapes.
The adapter detects the provider family from `modelId` and dispatches token,
content, finish-reason, and response-id extraction accordingly.

> The legacy single-file import path
> `layerlens.instrument.adapters.providers.bedrock_adapter` continues to work
> via a thin re-export shim. New code should import from
> `layerlens.instrument.adapters.providers.bedrock` directly.

## Install

```bash
pip install 'layerlens[providers-bedrock]'
```

This pulls `boto3>=1.34`. The default `pip install layerlens` does NOT
import or require boto3 — verified by `tests/instrument/test_lazy_imports.py`
and `tests/instrument/test_default_install.py`.

## Authentication

The adapter does not touch `boto3.Session` — it wraps whichever
`bedrock-runtime` client you pass in, so any auth method boto3 already
supports works:

| Method | When to use | How |
|---|---|---|
| **IAM role** | EC2 / ECS / EKS / Lambda runtime | No env vars needed; `boto3.client("bedrock-runtime")` picks up the instance / task / pod role automatically. Preferred for production. |
| **Static keys** | Local dev, CI, machines without an instance role | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars (and optional `AWS_SESSION_TOKEN` for temporary creds). |
| **SSO profile** | Workforce identities via IAM Identity Center | `aws sso login --profile my-prof` then either `AWS_PROFILE=my-prof` or `boto3.Session(profile_name="my-prof").client("bedrock-runtime")`. |
| **Cross-account assume-role** | Multi-account deployments | Use `boto3.client("sts").assume_role(...)` and feed the temporary credentials into `boto3.client("bedrock-runtime", aws_access_key_id=..., aws_secret_access_key=..., aws_session_token=...)`. |

### Required IAM permissions

Minimum policy for the adapter's wrapped methods:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.*",
        "arn:aws:bedrock:*::foundation-model/meta.*",
        "arn:aws:bedrock:*::foundation-model/cohere.*",
        "arn:aws:bedrock:*::foundation-model/amazon.*",
        "arn:aws:bedrock:*::foundation-model/ai21.*",
        "arn:aws:bedrock:*::foundation-model/mistral.*"
      ]
    }
  ]
}
```

Scope `Resource` down to only the model families you actually call. For
provisioned-throughput model deployments use the deployment ARN
(`arn:aws:bedrock:REGION:ACCOUNT:provisioned-model/MODEL_ID`) instead of the
foundation-model ARN.

## Quick start

```python
import boto3
from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="aws_bedrock")
adapter = AWSBedrockAdapter()
adapter.add_sink(sink)
adapter.connect()

client = boto3.client("bedrock-runtime", region_name="us-east-1")
adapter.connect_client(client)

# Either invoke_model or converse — both are wrapped automatically.
client.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
)
```

## Wrapped methods

| Method | Body extraction | Notes |
|---|---|---|
| `invoke_model` | Per-family JSON body shape | Response body is wrapped in a `RereadableBody` so the caller's downstream `body.read()` still works. |
| `invoke_model_with_response_stream` | Request only | Emits `model.invoke` immediately with `streaming=true`; content extraction during stream consumption is deferred to a future PR. |
| `converse` | Unified Converse envelope | Token extraction is uniform across families. Captures `stopReason` and `ResponseMetadata.RequestId`. |
| `converse_stream` | Request only | Same streaming semantics as above. |

## Supported model families

Bedrock's `modelId` prefix selects the dispatch branch:

| `modelId` prefix | Family | Token fields | Output extraction |
|---|---|---|---|
| `anthropic.` | anthropic | `usage.input_tokens` / `usage.output_tokens` | `content[*].text` |
| `meta.` | meta | `prompt_token_count` / `generation_token_count` | `generation` |
| `cohere.` | cohere | `meta.billed_units.input_tokens` / `output_tokens` | `generations[0].text` |
| `amazon.` | amazon | `inputTextTokenCount` / `results[0].tokenCount` | `results[0].outputText` |
| `ai21.` | ai21 | `usage.prompt_tokens` / `usage.completion_tokens` | `choices[0].message.content` (Jamba) or `completions[0].data.text` (J2) |
| `mistral.` | mistral | `prompt_tokens` / `completion_tokens` | `generation` |
| anything else | unknown | Generic `inputTokenCount` / `outputTokenCount` fallback | `generation` / `completion` / `outputText` fallback |

The Converse API exposes a uniform `usage.inputTokens` / `usage.outputTokens`
envelope across all families, so converse-path token extraction does not need
the per-family branch.

## Cost calculation

Uses the `BEDROCK_PRICING` table (separate from `PRICING` and `AZURE_PRICING`).
Cost is emitted on the `cost.record` event in USD. If the `modelId` is not
present in the table, the event still fires but `api_cost_usd` is `None` and
`pricing_unavailable` is `True`. Update
`src/layerlens/instrument/adapters/providers/_base/pricing.py` to add new model
entries — both the source-of-truth in `ateam` and the `stratix-python` mirror
must move together (CI hash-checks the two for drift).

Currently priced model IDs:

| modelId | $/1k input | $/1k output |
|---|---|---|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | 0.003 | 0.015 |
| `anthropic.claude-3-opus-20240229-v1:0` | 0.015 | 0.075 |
| `anthropic.claude-3-haiku-20240307-v1:0` | 0.00025 | 0.00125 |
| `meta.llama3-1-70b-instruct-v1:0` | 0.00099 | 0.00099 |
| `meta.llama3-1-8b-instruct-v1:0` | 0.00022 | 0.00022 |
| `cohere.command-r-plus-v1:0` | 0.003 | 0.015 |
| `cohere.command-r-v1:0` | 0.0005 | 0.0015 |

## Sample

A runnable, fully offline sample lives at
`samples/instrument/providers/bedrock/`. It uses
`botocore.stub.Stubber` so it requires no AWS credentials. Set
`LAYERLENS_BEDROCK_LIVE=1` plus standard AWS env vars to drive the live API
instead.
