"""Runnable sample: AWS Bedrock + LayerLens instrumentation (LAY-3452).

Run with::

    pip install layerlens[bedrock]
    python samples/instrument/bedrock/example.py

See ``docs/adapters/providers/bedrock.md`` for the required IAM permissions
and the supported model families (Anthropic / Meta / Cohere / Amazon / Mistral).
"""

from __future__ import annotations

import os
import sys
import json


def main() -> int:
    try:
        from layerlens.instrument.adapters.providers import BedrockProvider
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Bedrock deps with: pip install layerlens[bedrock]")
        return 0

    print("BedrockProvider available.")
    try:
        import boto3  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install boto3: pip install layerlens[bedrock]")
        return 0

    region = os.environ.get("AWS_REGION", "us-east-1")
    print(f"Wiring against Bedrock runtime in {region}")
    print("(requires IAM permission bedrock:InvokeModel — see the doc)")
    print()
    print("    client = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION'))")
    print("    provider = BedrockProvider()")
    print("    provider.connect(client)   # wraps invoke_model / converse / *_stream")
    print("    resp = client.invoke_model(")
    print("        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',")
    print("        body=json.dumps({...}),")
    print("    )")

    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        print("\n[live call skipped] configure AWS credentials (AWS_PROFILE or keys) to run for real.")
        return 0
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        provider = BedrockProvider()
        provider.connect(client)
        model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                }
            ),
        )
        body = json.loads(resp["body"].read())
        print(f"Bedrock responded: {body.get('content')}")
    except Exception as exc:  # noqa: BLE001 -- sample shouldn't hard-fail
        print(f"[bedrock call skipped] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
