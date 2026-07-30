"""Sample: Semantic Kernel multi-agent team instrumented with layerlens.

Unlike ``semantic_kernel_planner.py`` (a single prompt function), this drives a
two-agent ``AgentGroupChat``. The ``SemanticKernelAdapter`` auto-detects the
group chat on ``connect()`` and wraps its ``invoke``/``invoke_stream`` so each
agent turn and the handoffs between agents are captured. Multi-agent.
"""

from __future__ import annotations

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter


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
        print("Set OPENAI_API_KEY to run the Semantic Kernel team against a live LLM.")
        return

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", service_id="chat"))
    proposer = ChatCompletionAgent(
        kernel=kernel,
        name="proposer",
        instructions="Propose a one-line answer to the question.",
    )
    critic = ChatCompletionAgent(
        kernel=kernel,
        name="critic",
        instructions="Improve the proposed line in one short line.",
    )
    chat = AgentGroupChat(
        agents=[proposer, critic],
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=2),
    )

    adapter = SemanticKernelAdapter(None)
    adapter.connect(chat)
    try:
        with capture_events("semantic_kernel_team"):
            await chat.add_chat_message(message="Name two oceans in a few words.")
            async for message in chat.invoke():
                print(f"{getattr(message, 'name', '?')}: {getattr(message, 'content', message)}")
    finally:
        adapter.disconnect()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
