"""Sample: AWS Bedrock adapter — invoke_model + converse on a Claude model."""

from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.bedrock import (
    instrument_bedrock,
    uninstrument_bedrock,
)


def main() -> None:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        print("Install the Bedrock extra: pip install 'layerlens[bedrock]'")
        return

    if not any(os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE")):
        print("Configure AWS credentials (AWS_ACCESS_KEY_ID or AWS_PROFILE) to run against Bedrock.")
        return

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    instrument_bedrock(client)
    try:
        with capture_events("bedrock_invoke"):
            resp = client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 60,
                        "messages": [{"role": "user", "content": "Name a planet."}],
                    }
                ),
            )
            print("reply raw bytes:", resp["body"].read()[:200])
    finally:
        uninstrument_bedrock()


if __name__ == "__main__":
    main()
