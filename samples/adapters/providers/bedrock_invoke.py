"""Sample: AWS Bedrock adapter — invoke_model + converse on Amazon Nova.

Uses ``amazon.nova-micro-v1:0`` (an active, on-demand model) for BOTH surfaces
the adapter instruments: the low-level ``invoke_model`` (Nova request schema) and
the unified ``converse`` API. A richer, industry-realistic Nova + Converse
tool-loop ships as the tracked Family-B samples ``energy_grid_forecast`` /
``energy_grid_tooluse``.
"""

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

MODEL_ID = "amazon.nova-micro-v1:0"


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
            # 1) Low-level invoke_model (Amazon Nova request schema).
            resp = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(
                    {
                        "messages": [{"role": "user", "content": [{"text": "Name a planet."}]}],
                        "inferenceConfig": {"maxTokens": 60},
                    }
                ),
            )
            print("invoke_model raw bytes:", resp["body"].read()[:200])

            # 2) The unified converse API (same model).
            converse = client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "Name a different planet."}]}],
                inferenceConfig={"maxTokens": 60},
            )
            print("converse reply:", converse["output"]["message"]["content"][0]["text"])
    finally:
        uninstrument_bedrock()


if __name__ == "__main__":
    main()
