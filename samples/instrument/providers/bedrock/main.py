"""Sample: instrument an AWS Bedrock client with the LayerLens adapter.

By default the script runs **fully offline** — it builds a real boto3
``bedrock-runtime`` client, attaches a ``botocore.stub.Stubber``, and
drives both the ``Converse`` and ``invoke_model`` paths through the
adapter. Telemetry is captured in-process and printed to stdout. No AWS
calls are made, no AWS credentials are required.

To exercise the adapter against the live Bedrock API instead, set
``LAYERLENS_BEDROCK_LIVE=1`` and provide standard AWS credentials (IAM
role, ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``, or a
configured SSO profile via ``AWS_PROFILE``). The script will then issue
a single low-cost ``Converse`` call against
``anthropic.claude-3-5-sonnet-20241022-v2:0`` in the region named by
``AWS_REGION`` (default ``us-east-1``) and print the response.

Run::

    pip install 'layerlens[providers-bedrock]'
    python -m samples.instrument.providers.bedrock.main
"""

from __future__ import annotations

import io
import os
import sys
import json
from typing import Any, Dict, List


def _print_event(event_type: str, payload: Dict[str, Any]) -> None:
    print(f"[{event_type}] {json.dumps(payload, default=str, indent=2)}")


class _StdoutLayerLens:
    """Tiny telemetry sink that prints every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            _print_event(args[0], args[1])


def _run_offline() -> int:
    try:
        import boto3
        from botocore.stub import Stubber
        from botocore.response import StreamingBody
    except ImportError:
        print(
            "boto3 not installed. Install with:\n"
            "    pip install 'layerlens[providers-bedrock]'",
            file=sys.stderr,
        )
        return 2

    from layerlens.instrument.adapters._base import CaptureConfig
    from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter

    sink = _StdoutLayerLens()
    adapter = AWSBedrockAdapter(layerlens=sink, capture_config=CaptureConfig.full())
    adapter.connect()

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        # Stubber requires a configured client but never sends bytes.
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    adapter.connect_client(client)

    # ---- Converse path -----------------------------------------------
    print("\n=== Converse (anthropic.claude-3-5-sonnet) ===")
    converse_response: Dict[str, Any] = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "2 + 2 = 4."}],
            }
        },
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
        "stopReason": "end_turn",
        # boto3 >= 1.40 enforces ``metrics`` on Converse responses.
        "metrics": {"latencyMs": 42},
    }
    converse_params: Dict[str, Any] = {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": [{"text": "What is 2 + 2?"}]}],
    }
    with Stubber(client) as stubber:
        stubber.add_response("converse", converse_response, converse_params)
        result = client.converse(**converse_params)
        stubber.assert_no_pending_responses()
    answer = result["output"]["message"]["content"][0]["text"]
    print(f"\nAssistant: {answer}\n")

    # ---- invoke_model path (Anthropic Messages body shape) ------------
    print("\n=== invoke_model (anthropic.claude-3-haiku) ===")
    invoke_payload = {
        "id": "msg_demo",
        "stop_reason": "end_turn",
        "content": [{"text": "Roses are red."}],
        "usage": {"input_tokens": 9, "output_tokens": 5},
    }
    raw = json.dumps(invoke_payload).encode("utf-8")
    invoke_response = {
        "body": StreamingBody(io.BytesIO(raw), len(raw)),
        "contentType": "application/json",
    }
    invoke_params = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": json.dumps(
            {
                "messages": [{"role": "user", "content": "Compose a haiku."}],
                "max_tokens": 50,
                "anthropic_version": "bedrock-2023-05-31",
            }
        ),
        "contentType": "application/json",
        "accept": "application/json",
    }
    with Stubber(client) as stubber:
        stubber.add_response("invoke_model", invoke_response, invoke_params)
        response = client.invoke_model(**invoke_params)
        stubber.assert_no_pending_responses()
    body = json.loads(response["body"].read())
    print(f"\nResponse body: {body['content'][0]['text']}\n")

    adapter.disconnect()
    print(f"\nTotal events captured: {len(sink.events)}")
    return 0


def _run_live() -> int:
    try:
        import boto3
    except ImportError:
        print(
            "boto3 not installed. Install with:\n"
            "    pip install 'layerlens[providers-bedrock]'",
            file=sys.stderr,
        )
        return 2

    from layerlens.instrument.adapters._base import CaptureConfig
    from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter

    region = os.environ.get("AWS_REGION", "us-east-1")
    sink = _StdoutLayerLens()
    adapter = AWSBedrockAdapter(layerlens=sink, capture_config=CaptureConfig.standard())
    adapter.connect()

    client = boto3.client("bedrock-runtime", region_name=region)
    adapter.connect_client(client)

    try:
        response = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Say hi in five words."}]}],
            inferenceConfig={"maxTokens": 50},
        )
    except Exception as exc:  # pragma: no cover - live path
        print(f"Bedrock call failed: {exc}", file=sys.stderr)
        adapter.disconnect()
        return 1

    answer = response["output"]["message"]["content"][0]["text"]
    print(f"\nAssistant: {answer}\n")
    adapter.disconnect()
    return 0


def main() -> int:
    if os.environ.get("LAYERLENS_BEDROCK_LIVE"):
        return _run_live()
    return _run_offline()


if __name__ == "__main__":
    raise SystemExit(main())
