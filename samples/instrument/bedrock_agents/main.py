"""Sample: instrument an AWS Bedrock Agent invocation with LayerLens.

Builds a ``bedrock-agent-runtime`` boto3 client, registers the LayerLens
event hooks via ``BedrockAgentsAdapter.instrument_client``, and runs a
single ``invoke_agent`` call. Emits ``agent.input`` + ``model.invoke`` +
``tool.call`` + ``agent.output`` events that ship to atlas-app via
``HttpEventSink``.

This sample requires a live Bedrock Agent ID. If you don't have one,
the sample exits with a clear error.

Required environment:

* ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` (or another standard
  boto3 credential source — IAM role, profile, etc.).
* ``AWS_REGION`` — the AWS region your agent lives in.
* ``BEDROCK_AGENT_ID`` — your Bedrock Agent ID (e.g. ``ABCDEFGHIJ``).
* ``BEDROCK_AGENT_ALIAS_ID`` — agent alias to invoke (default
  ``TSTALIASID``).
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[bedrock-agents]'
    python -m samples.instrument.bedrock_agents.main
"""

from __future__ import annotations

import os
import sys
import uuid

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter


def main() -> int:
    agent_id = os.environ.get("BEDROCK_AGENT_ID")
    if not agent_id:
        print("BEDROCK_AGENT_ID is not set; cannot run sample.", file=sys.stderr)
        return 2

    region = os.environ.get("AWS_REGION", "us-east-1")
    alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")

    try:
        import boto3
    except ImportError:
        print(
            "boto3 not installed. Install with:\n"
            "    pip install 'layerlens[bedrock-agents]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="bedrock_agents",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = BedrockAgentsAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = boto3.client("bedrock-agent-runtime", region_name=region)
    adapter.instrument_client(client)

    try:
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=str(uuid.uuid4()),
            inputText="What is 2 + 2?",
        )
        # Drain the streamed response — trace events fire as we iterate.
        chunks: list[bytes] = []
        for event in response["completion"]:
            if "chunk" in event:
                chunks.append(event["chunk"]["bytes"])
        text = b"".join(chunks).decode("utf-8", errors="replace")
        print(f"Response: {text}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
