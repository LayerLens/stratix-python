"""Sample: Google ADK hierarchical multi-agent team instrumented with layerlens.

A ``coordinator`` agent owns two ``sub_agents`` (researcher + writer). ADK's
hierarchical delegation lets the coordinator transfer control to a sub-agent;
the adapter captures each agent's turn plus the ``agent.handoff`` edges from the
``transfer_to_agent`` actions. Multi-agent.
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
        print("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to run the Google ADK team against Gemini.")
        return

    model = os.environ.get("LL_GEMINI_MODEL", "gemini-2.5-flash")
    researcher = Agent(
        name="researcher",
        model=model,
        instruction="Name one ocean in a few words.",
    )
    writer = Agent(
        name="writer",
        model=model,
        instruction="Turn the researcher's fact into one short sentence.",
    )
    coordinator = Agent(
        name="coordinator",
        model=model,
        instruction=(
            "You coordinate a team. Transfer to the researcher to name an ocean, "
            "then transfer to the writer to phrase the final answer."
        ),
        sub_agents=[researcher, writer],
    )

    adapter = GoogleADKAdapter(None)
    adapter.connect()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="layerlens-sample-team",
        agent=coordinator,
        session_service=session_service,
        plugins=[adapter.plugin],
    )
    try:
        with capture_events("google_adk_team"):
            session = await session_service.create_session(app_name="layerlens-sample-team", user_id="sample")
            message = types.Content(role="user", parts=[types.Part(text="Give me one sentence about an ocean.")])
            async for _event in runner.run_async(user_id="sample", session_id=session.id, new_message=message):
                pass
            print("team run complete")
    finally:
        adapter.disconnect()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
