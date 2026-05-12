"""Sample: instrument a Microsoft Agent Framework chat with LayerLens.

Builds a one-shot ``ChatCompletionAgent`` backed by an OpenAI chat
completion service, wraps it via ``MSAgentAdapter.instrument_chat``, and
runs a single ``invoke`` call. Each invocation emits ``agent.input`` +
``model.invoke`` + ``agent.output`` events that ship to atlas-app via
``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by ``OpenAIChatCompletion``.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[ms-agent-framework,providers-openai]'
    python -m samples.instrument.ms_agent_framework.main
"""

from __future__ import annotations

import os
import sys
import asyncio

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentAdapter


async def _run(agent: object) -> str:
    chunks: list[str] = []
    async for response in agent.invoke("What is 2 + 2?"):  # type: ignore[attr-defined]
        content = getattr(response, "content", None)
        if content is not None:
            chunks.append(str(content))
    return " ".join(chunks)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from semantic_kernel.agents import ChatCompletionAgent
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    except ImportError:
        print(
            "semantic-kernel agents not installed. Install with:\n"
            "    pip install 'layerlens[ms-agent-framework,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="ms_agent_framework",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = MSAgentAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(ai_model_id="gpt-4o-mini"),
        name="answerer",
        instructions="Reply with the digit only.",
    )
    adapter.instrument_chat(agent)

    try:
        text = asyncio.run(_run(agent))
        print(f"Response: {text}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
