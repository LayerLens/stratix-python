#!/usr/bin/env python3
"""
BrowserUse Web Browsing Agent with STRATIX Instrumentation

Demonstrates a BrowserUse agent that navigates the web to complete tasks,
with STRATIX tracing of navigation steps, page interactions, and results.

Requirements:
    pip install browser-use langchain-openai

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

try:
    from browser_use import Agent as BrowserAgent
    from langchain_openai import ChatOpenAI
except ImportError:
    sys.exit(
        "This sample requires browser-use. Install with:\n"
        "  pip install browser-use langchain-openai"
    )

from layerlens.instrument import STRATIX


async def run_browser_agent(args):
    """Execute the browser agent with STRATIX tracing."""
    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="langchain",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(args.task)

    # Configure LLM
    llm = ChatOpenAI(model=args.model, temperature=0)

    # Create browser agent
    agent = BrowserAgent(
        task=args.task,
        llm=llm,
        max_actions_per_step=3,
    )

    print(f"Task: {args.task}\n")
    print("Running browser agent (this may take a moment)...\n")

    # Run the agent
    result = await agent.run(max_steps=args.max_steps)

    # Extract and display results
    output = str(result)
    print(f"Result:\n{output[:600]}{'...' if len(output) > 600 else ''}\n")

    # Show action history if available
    history = getattr(agent, "history", None)
    if history:
        print(f"Actions taken ({len(history)}):")
        for i, action in enumerate(history, 1):
            action_str = str(action)[:100]
            print(f"  {i}. {action_str}")
        print()

    stratix.emit_output(output[:1000])

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="BrowserUse web agent with STRATIX tracing")
    parser.add_argument(
        "--task",
        default="Go to the Python Package Index (pypi.org) and find the latest version of the 'requests' library.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--agent-id", default="browseruse-agent-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    asyncio.run(run_browser_agent(args))


if __name__ == "__main__":
    main()
