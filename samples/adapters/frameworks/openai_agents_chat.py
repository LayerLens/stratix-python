"""Sample: OpenAI Agents SDK adapter."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]


def main() -> None:
    try:
        from agents import Agent, Runner  # type: ignore[import-not-found]

        from layerlens.instrument.adapters.frameworks.openai_agents import (
            OpenAIAgentsAdapter,
        )
    except ImportError:
        print("Install: pip install 'layerlens[openai-agents]'")
        return

    agent = Agent(name="demo", instructions="Answer in one word.")
    adapter = OpenAIAgentsAdapter(client=agent)
    adapter.connect()
    with capture_events("openai_agents"):
        result = Runner.run_sync(agent, "What colour is grass?")
        print("reply:", result.final_output)


if __name__ == "__main__":
    main()
