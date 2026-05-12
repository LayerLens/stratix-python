"""Sample: instrument a PydanticAI agent with the LayerLens adapter.

Builds a one-shot ``Agent``, wraps it with
``PydanticAIAdapter.instrument_agent``, and runs ``agent.run_sync``. Each
run emits ``agent.input`` + ``model.invoke`` + ``agent.output`` events that
ship to atlas-app via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by the ``"openai:gpt-4o-mini"`` model spec.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[pydantic-ai,providers-openai]'
    python -m samples.instrument.pydantic_ai.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from pydantic_ai import Agent
    except ImportError:
        print(
            "pydantic-ai not installed. Install with:\n"
            "    pip install 'layerlens[pydantic-ai,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="pydantic_ai",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = PydanticAIAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Reply with the digit only.",
    )

    try:
        adapter.instrument_agent(agent)
        result = agent.run_sync("What is 2 + 2?")
        print(f"Response: {result.data}")
        usage = result.usage()
        if usage is not None:
            print(
                f"Tokens — request: {usage.request_tokens}, "
                f"response: {usage.response_tokens}, total: {usage.total_tokens}"
            )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
