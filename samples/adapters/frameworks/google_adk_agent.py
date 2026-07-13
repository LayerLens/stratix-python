"""Sample: Google ADK agent instrumented with layerlens (single agent).

The adapter is a Runner plugin (``runner=Runner(..., plugins=[adapter.plugin])``)
that captures the model call, tokens, tool calls, and any handoffs. Uses Gemini
via an API key (maps ``GEMINI_API_KEY`` -> ``GOOGLE_API_KEY`` if needed).
"""

from __future__ import annotations

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter


async def run() -> None:
    try:
        from google.genai import types  # type: ignore[import-not-found]
        from google.adk.agents import Agent  # type: ignore[import-not-found]
        from google.adk.runners import Runner  # type: ignore[import-not-found]
        from google.adk.sessions import InMemorySessionService  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install google-adk")
        return

    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to run Google ADK against Gemini.")
        return

    agent = Agent(
        name="sample_agent",
        model=os.environ.get("LL_GEMINI_MODEL", "gemini-2.5-flash"),
        instruction="Answer in one short sentence.",
    )

    adapter = GoogleADKAdapter(None)
    adapter.connect()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="layerlens-sample",
        agent=agent,
        session_service=session_service,
        plugins=[adapter.plugin],
    )
    try:
        with capture_events("google_adk_agent"):
            session = await session_service.create_session(app_name="layerlens-sample", user_id="sample")
            message = types.Content(role="user", parts=[types.Part(text="Name two oceans in a few words.")])
            async for _event in runner.run_async(user_id="sample", session_id=session.id, new_message=message):
                pass
            print("run complete")
    finally:
        adapter.disconnect()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
