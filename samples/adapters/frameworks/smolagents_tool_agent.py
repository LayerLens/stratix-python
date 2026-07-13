"""Sample: SmolAgents tool-calling agent instrumented with layerlens (single agent)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter


def main() -> None:
    try:
        from smolagents import ToolCallingAgent, OpenAIServerModel, tool  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install smolagents openai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run SmolAgents against a live LLM.")
        return

    @tool
    def ocean_count() -> int:
        """Return the number of oceans on Earth."""
        return 5

    model = OpenAIServerModel(model_id="gpt-4o-mini")
    agent = ToolCallingAgent(tools=[ocean_count], model=model, max_steps=3)

    adapter = SmolAgentsAdapter(None)
    adapter.connect(target=agent)
    try:
        with capture_events("smolagents_tool_agent"):
            result = agent.run("Use the ocean_count tool, then answer in one short sentence.")
            print("reply:", result)
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
