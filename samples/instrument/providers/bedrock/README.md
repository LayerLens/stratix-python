# AWS Bedrock adapter sample

This sample demonstrates the LayerLens AWS Bedrock provider adapter wrapping a
real `boto3 bedrock-runtime` client. Every `Converse` or `invoke_model` call is
intercepted and turned into telemetry events.

## Two run modes

The sample defaults to **offline mode** so it can be run on any machine
without AWS credentials. A `botocore.stub.Stubber` is attached to a real
`boto3` client and the adapter wraps that stubbed client. Both the `Converse`
and `invoke_model` paths are exercised, and the captured events are printed to
stdout.

To run against the **live Bedrock API** instead, set
`LAYERLENS_BEDROCK_LIVE=1` and provide AWS credentials by any of the methods
boto3 already supports — IAM role, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`,
or `AWS_PROFILE` for an SSO-configured profile. The script will issue a single
low-cost `Converse` call against `anthropic.claude-3-5-sonnet-20241022-v2:0` in
`AWS_REGION` (default `us-east-1`).

## What you'll see

Each call produces two events:

- `model.invoke` (L3) — the request and response, with the detected
  `provider_family`, parameters, tokens, latency, and the assistant output
  message.
- `cost.record` (cross-cutting) — the API cost in USD computed from the
  `BEDROCK_PRICING` table for the model's `modelId`.

Streaming methods (`invoke_model_with_response_stream`, `converse_stream`)
emit only the request-side `model.invoke` because the content is not known
until the caller fully consumes the iterator.

## Install

```bash
pip install 'layerlens[providers-bedrock]'
```

This pulls `boto3>=1.34`. The default `pip install layerlens` does NOT pull
`boto3` — that's the lazy-import guarantee tested by
`tests/instrument/test_lazy_imports.py`.

## Run (offline, no AWS account needed)

```bash
python -m samples.instrument.providers.bedrock.main
```

You'll see the captured `model.invoke` and `cost.record` events streamed to
stdout for both the Converse and invoke_model paths.

## Run (live, against AWS Bedrock)

```bash
export LAYERLENS_BEDROCK_LIVE=1
export AWS_REGION=us-east-1
# Plus one of:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# OR
export AWS_PROFILE=my-sso-profile
# OR
# (rely on the IAM role of the EC2 / ECS / Lambda runtime)

python -m samples.instrument.providers.bedrock.main
```

Make sure your IAM principal has `bedrock:InvokeModel` (or
`bedrock:InvokeModelWithResponseStream`) and `bedrock:Converse` allowed for
the `anthropic.claude-3-5-sonnet-20241022-v2:0` model in the chosen region.
