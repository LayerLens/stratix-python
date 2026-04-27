"""Sample: instrument a Google ADK agent with the LayerLens adapter.

Builds a one-shot ``LlmAgent``, attaches the LayerLens callbacks via
``GoogleADKAdapter.instrument_agent``, and runs a single turn through the
ADK ``Runner``. Each callback fires a LayerLens event that ships to atlas-app
via ``HttpEventSink``.

Required environment:

* ``GOOGLE_API_KEY`` — used by the Gemini model when running against
  Google AI Studio. (For Vertex AI, set ``GOOGLE_GENAI_USE_VERTEXAI=true``
  and provide ADC.)
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[google-adk]'
    python -m samples.instrument.google_adk.main
"""

from __future__ import annotations

import os
import sys
import asyncio

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter


async def _run_agent(runner: object, session_id: str, user_id: str) -> str:
    from google.genai import types  # type: ignore[import-untyped,unused-ignore]

    new_message = types.Content(
        role="user",
        parts=[types.Part(text="What is 2 + 2?")],
    )

    chunks: list[str] = []
    # ``run_async`` is the recommended async API on the ADK Runner.
    async for event in runner.run_async(  # type: ignore[attr-defined]
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        content = getattr(event, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get(
        "GOOGLE_GENAI_USE_VERTEXAI"
    ) != "true":
        print(
            "Neither GOOGLE_API_KEY nor GOOGLE_GENAI_USE_VERTEXAI is set; "
            "cannot run sample.",
            file=sys.stderr,
        )
        return 2

    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
    except ImportError:
        print(
            "google-adk not installed. Install with:\n"
            "    pip install 'layerlens[google-adk]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="google_adk",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = GoogleADKAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    agent = LlmAgent(
        name="answerer",
        model="gemini-2.0-flash",
        instruction="Reply with the digit only.",
    )
    adapter.instrument_agent(agent)

    runner = InMemoryRunner(agent=agent, app_name="layerlens-sample")
    user_id = "sample-user"
    # Create a session up front so ``run_async`` has somewhere to write.
    session = asyncio.run(
        runner.session_service.create_session(
            app_name="layerlens-sample", user_id=user_id
        )
    )

    try:
        text = asyncio.run(_run_agent(runner, session.id, user_id))
        print(f"Response: {text}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
