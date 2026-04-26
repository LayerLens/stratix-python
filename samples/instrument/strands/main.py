"""Sample: instrument an AWS Strands agent with the LayerLens adapter.

Builds a one-shot Strands ``Agent`` backed by a Bedrock model, wraps it via
``StrandsAdapter.instrument_agent``, and runs a single call. Each call emits
``agent.input`` + ``model.invoke`` + ``agent.output`` events that ship to
atlas-app via ``HttpEventSink``.

Required environment:

* ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` (or another standard
  boto3 credential source — IAM role, profile, etc.).
* ``AWS_REGION`` — the AWS region (Strands defaults to us-west-2; set
  this to wherever your Bedrock model access is enabled).
* ``BEDROCK_MODEL_ID`` — Bedrock model ID for Strands to use; defaults to
  ``us.anthropic.claude-3-5-sonnet-20241022-v2:0`` if unset.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[strands]'
    python -m samples.instrument.strands.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter


def main() -> int:
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get(
        "AWS_PROFILE"
    ):
        print(
            "AWS credentials are not set (need AWS_ACCESS_KEY_ID or AWS_PROFILE); "
            "cannot run sample.",
            file=sys.stderr,
        )
        return 2

    try:
        from strands import Agent
    except ImportError:
        print(
            "strands-agents not installed. Install with:\n"
            "    pip install 'layerlens[strands]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="strands",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = StrandsAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    agent = Agent(model=model_id, system_prompt="Reply with the digit only.")

    try:
        adapter.instrument_agent(agent)
        response = agent("What is 2 + 2?")
        print(f"Response: {response}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
