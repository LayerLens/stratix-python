"""Sample: PydanticAI typed agent with a tool."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter


def main() -> None:
    try:
        from pydantic_ai import Agent  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[pydantic-ai]' pydantic-ai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run PydanticAI against a live LLM.")
        return

    agent = Agent("openai:gpt-4o-mini", system_prompt="Reply in one word.")

    @agent.tool_plain
    def length(text: str) -> int:
        return len(text)

    PydanticAIAdapter(None).connect(agent)
    with capture_events("pydanticai_agent"):
        result = agent.run_sync("Colour of grass?")
        print("reply:", result.data)


if __name__ == "__main__":
    main()
