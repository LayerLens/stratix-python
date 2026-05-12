"""Sample: instrument an Agno agent with the LayerLens adapter.

Builds a one-shot Agno ``Agent`` with the OpenAI ``gpt-4o-mini`` model,
instruments it via ``AgnoAdapter.instrument_agent``, and runs a single
``agent.run()`` call. Each run emits ``agent.input`` + ``model.invoke`` +
``agent.output`` events that ship to atlas-app via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by the ``OpenAIChat`` model.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[agno,providers-openai]'
    python -m samples.instrument.agno.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
    except ImportError:
        print(
            "agno not installed. Install with:\n"
            "    pip install 'layerlens[agno,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="agno",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = AgnoAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", max_tokens=20),
        instructions="Reply with the digit only.",
    )

    try:
        adapter.instrument_agent(agent)
        response = agent.run("What is 2 + 2?")
        content = getattr(response, "content", str(response))
        print(f"Response: {content}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
