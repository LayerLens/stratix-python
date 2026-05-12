"""Sample: instrument the OpenAI Agents SDK with the LayerLens adapter.

Registers the LayerLens trace processor with the SDK via
``OpenAIAgentsAdapter.instrument_runner``, then runs a one-turn agent via
``Runner.run_sync``. Each span the SDK produces (agent, model, tool,
handoff) emits a LayerLens event that ships to atlas-app via
``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by the underlying OpenAI client.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[openai-agents]' openai-agents
    python -m samples.instrument.openai_agents.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from agents import Agent, Runner
    except ImportError:
        print(
            "openai-agents not installed. Install with:\n"
            "    pip install 'layerlens[openai-agents]' openai-agents",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="openai_agents",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = OpenAIAgentsAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()
    adapter.instrument_runner(None)  # global trace processor

    agent = Agent(
        name="answerer",
        instructions="Reply with the digit only.",
        model="gpt-4o-mini",
    )

    try:
        result = Runner.run_sync(agent, "What is 2 + 2?")
        print(f"Response: {result.final_output}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
