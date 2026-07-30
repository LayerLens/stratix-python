"""Sample: AutoGen two-agent conversation (autogen-agentchat >= 0.4)."""

from __future__ import annotations

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter


def main() -> None:
    try:
        from autogen_agentchat.teams import RoundRobinGroupChat  # type: ignore[import-not-found]
        from autogen_agentchat.agents import AssistantAgent  # type: ignore[import-not-found]
        from autogen_agentchat.conditions import MaxMessageTermination  # type: ignore[import-not-found]
        from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[autogen]' 'autogen-ext[openai]'")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run AutoGen against a live LLM.")
        return

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    poet = AssistantAgent(
        "poet",
        model_client=model_client,
        system_message="Answer in a single short line.",
    )
    critic = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Improve the previous line in one short line.",
    )
    team = RoundRobinGroupChat(
        [poet, critic],
        termination_condition=MaxMessageTermination(max_messages=3),
    )

    adapter = AutoGenAdapter(None)
    adapter.connect()
    try:
        with capture_events("autogen_conversation"):
            result = asyncio.run(team.run(task="Say grass is green in one line."))
            last = result.messages[-1]
            print("last message:", getattr(last, "content", last))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
