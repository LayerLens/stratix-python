"""Sample: MS Agent Framework multi-agent group chat instrumented with layerlens.

The adapter instruments a Semantic Kernel ``AgentGroupChat`` (the multi-agent
surface the ms_agent_framework key detects). This sample runs a genuine
two-agent team — a writer and a reviewer — and captures each agent's turn plus
the handoffs between them. Multi-agent.
"""

from __future__ import annotations

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentFrameworkAdapter


async def run() -> None:
    try:
        from semantic_kernel import Kernel  # type: ignore[import-not-found]
        from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent  # type: ignore[import-not-found]
        from semantic_kernel.agents.strategies import DefaultTerminationStrategy  # type: ignore[import-not-found]
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[semantic-kernel]' semantic-kernel")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run the MS Agent Framework group chat against a live LLM.")
        return

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", service_id="chat"))
    writer = ChatCompletionAgent(
        kernel=kernel,
        name="writer",
        instructions="Answer the question in one short sentence.",
    )
    reviewer = ChatCompletionAgent(
        kernel=kernel,
        name="reviewer",
        instructions="Reply 'approved' if the sentence is correct, else suggest a one-line fix.",
    )
    chat = AgentGroupChat(
        agents=[writer, reviewer],
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=2),
    )

    adapter = MSAgentFrameworkAdapter(None)
    adapter.connect()
    adapter.instrument_chat(chat)
    try:
        with capture_events("ms_agent_framework_chat"):
            await chat.add_chat_message(message="Name two oceans in a few words.")
            async for message in chat.invoke():
                print(f"{getattr(message, 'name', '?')}: {getattr(message, 'content', message)}")
    finally:
        adapter.disconnect()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
