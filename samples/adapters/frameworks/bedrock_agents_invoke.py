"""Sample: AWS Bedrock Agents ``invoke_agent`` instrumented with layerlens (single agent).

The adapter hooks the ``bedrock-agent-runtime`` client and observes the
``completion`` EventStream as it is drained, emitting the model invocation,
output, and any trace steps. Requires a provisioned Bedrock Agent plus AWS
credentials.
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter


def main() -> None:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[bedrock]' boto3")
        return

    if not (os.environ.get("BEDROCK_AGENT_ID") and os.environ.get("BEDROCK_AGENT_ALIAS_ID")):
        print("Set BEDROCK_AGENT_ID, BEDROCK_AGENT_ALIAS_ID and AWS credentials to run Bedrock Agents.")
        return

    rt = boto3.client("bedrock-agent-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    adapter = BedrockAgentsAdapter(None)
    adapter.connect(target=rt)
    try:
        with capture_events("bedrock_agents_invoke"):
            response = rt.invoke_agent(
                agentId=os.environ["BEDROCK_AGENT_ID"],
                agentAliasId=os.environ["BEDROCK_AGENT_ALIAS_ID"],
                sessionId="ll-sample-" + uuid.uuid4().hex[:12],
                inputText="What is 2+2? Reply with only the number.",
                enableTrace=True,
            )
            # Drain the completion stream exactly as a customer would — this is
            # what drives the adapter's emission and flush.
            chunks = list(response["completion"])
            print("completion chunks:", len(chunks))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
