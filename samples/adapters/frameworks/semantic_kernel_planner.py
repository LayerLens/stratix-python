"""Sample: Semantic Kernel prompt function invocation."""

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
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[semantic-kernel]' semantic-kernel")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run semantic_kernel against a live LLM.")
        return

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o-mini"))
    fn = kernel.add_function(
        plugin_name="demo",
        function_name="greet",
        prompt="Reply in one word: what colour is grass?",
    )

    SemanticKernelAdapter(None).connect(kernel)
    with capture_events("semantic_kernel_planner"):
        result = await kernel.invoke(fn)
        print("reply:", result)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
