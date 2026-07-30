"""Sample: AWS Strands agent instrumented with layerlens (single agent).

The Strands adapter is a native ``HookProvider``: it is passed to the agent via
``hooks=[adapter]`` and captures the model invocation, tokens, and output. Uses
the OpenAI model backend so it runs with only ``OPENAI_API_KEY`` (no AWS creds).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter


def main() -> None:
    try:
        from strands import Agent  # type: ignore[import-not-found]
        from strands.models.openai import OpenAIModel  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'strands-agents[openai]'")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run Strands against a live LLM.")
        return

    adapter = StrandsAdapter(None)
    adapter.connect()
    try:
        model = OpenAIModel(model_id="gpt-4o-mini", params={"max_tokens": 64})
        agent = Agent(model=model, hooks=[adapter], callback_handler=None)
        with capture_events("strands_agent"):
            result = agent("Name two oceans in a few words.")
            print("reply:", result)
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
