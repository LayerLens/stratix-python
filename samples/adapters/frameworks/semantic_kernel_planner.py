#!/usr/bin/env python3
"""
Semantic Kernel with Plugins and STRATIX Instrumentation

Demonstrates a Semantic Kernel application with native plugins, traced
by STRATIX to capture plugin invocations and planner steps.

Requirements:
    pip install semantic-kernel

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.functions import kernel_function
except ImportError:
    sys.exit(
        "This sample requires semantic-kernel. Install with:\n"
        "  pip install semantic-kernel"
    )

from layerlens.instrument import STRATIX, emit_tool_call


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------

class MathPlugin:
    """A simple math plugin for Semantic Kernel."""

    @kernel_function(name="add", description="Add two numbers")
    def add(self, a: float, b: float) -> str:
        return str(float(a) + float(b))

    @kernel_function(name="multiply", description="Multiply two numbers")
    def multiply(self, a: float, b: float) -> str:
        return str(float(a) * float(b))

    @kernel_function(name="factorial", description="Compute factorial of n")
    def factorial(self, n: int) -> str:
        n = int(n)
        if n < 0:
            return "Error: negative input"
        result = 1
        for i in range(2, n + 1):
            result *= i
        return str(result)


class TextPlugin:
    """A text manipulation plugin."""

    @kernel_function(name="summarize_prompt", description="Create a summarization prompt")
    def summarize_prompt(self, text: str) -> str:
        return f"Please summarize the following in 2-3 sentences:\n\n{text}"

    @kernel_function(name="word_count", description="Count words in text")
    def word_count(self, text: str) -> str:
        return str(len(text.split()))


async def run_kernel(args):
    """Build and execute the Semantic Kernel pipeline."""
    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="semantic_kernel",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(args.query)

    # Build kernel
    kernel = sk.Kernel()
    service = OpenAIChatCompletion(
        service_id="chat",
        ai_model_id=args.model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    kernel.add_service(service)
    kernel.add_plugin(MathPlugin(), plugin_name="math")
    kernel.add_plugin(TextPlugin(), plugin_name="text")

    # Configure function-calling behavior
    settings = kernel.get_prompt_execution_settings_from_service_id("chat")
    settings.function_choice_behavior = sk.FunctionChoiceBehavior.Auto(
        filters={"included_plugins": ["math", "text"]}
    )

    print(f"Query: {args.query}\n")

    # Invoke the kernel with auto function calling
    result = await kernel.invoke_prompt(
        args.query,
        settings=settings,
    )

    output = str(result)
    print(f"Result: {output}\n")
    stratix.emit_output(output)

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Kernel with STRATIX tracing")
    parser.add_argument("--query", default="What is 7 factorial, and then multiply the result by 3?")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="semantic-kernel-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    asyncio.run(run_kernel(args))


if __name__ == "__main__":
    main()
