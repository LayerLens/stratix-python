"""Sample: Agno agent run instrumented with layerlens (single agent)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter


def main() -> None:
    try:
        from agno.agent import Agent  # type: ignore[import-not-found]
        from agno.models.openai import OpenAIChat  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install agno openai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run Agno against a live LLM.")
        return

    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), markdown=False)

    adapter = AgnoAdapter(None)
    adapter.connect(target=agent)
    try:
        with capture_events("agno_agent"):
            result = agent.run("Name two oceans in a few words.")
            print("reply:", getattr(result, "content", result))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
