#!/usr/bin/env python3
"""
AutoGen Multi-Agent Conversation with STRATIX Instrumentation

Demonstrates a multi-agent conversation where agents collaborate to solve
a task, with STRATIX tracing of message exchanges and tool usage.

Requirements:
    pip install pyautogen

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    sys.exit(
        "This sample requires autogen. Install with:\n"
        "  pip install pyautogen"
    )

from layerlens.instrument import STRATIX, emit_handoff


def main():
    parser = argparse.ArgumentParser(description="AutoGen multi-agent conversation with STRATIX tracing")
    parser.add_argument("--task", default="Write a Python function to compute the Fibonacci sequence and test it.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="autogen-conversation-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="autogen",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(args.task)

    # LLM config
    llm_config = {
        "config_list": [{"model": args.model, "api_key": os.environ["OPENAI_API_KEY"]}],
        "temperature": 0,
    }

    # Define agents
    coder = AssistantAgent(
        name="Coder",
        system_message=(
            "You are a senior Python developer. Write clean, well-documented code. "
            "When you produce code, wrap it in ```python blocks."
        ),
        llm_config=llm_config,
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        system_message=(
            "You are a code reviewer. Review the code for correctness, style, and "
            "edge cases. Suggest improvements if needed. Say APPROVED when satisfied."
        ),
        llm_config=llm_config,
    )

    executor = UserProxyAgent(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config={"work_dir": "/tmp/autogen_work", "use_docker": False},
        system_message="Execute code and report results. Terminate when the task is complete.",
    )

    # Set up group chat
    group_chat = GroupChat(
        agents=[coder, reviewer, executor],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    print(f"Task: {args.task}\n")
    print("Starting multi-agent conversation...\n")

    # Run the conversation
    executor.initiate_chat(manager, message=args.task)

    # Collect final output
    final_messages = [
        msg.get("content", "") for msg in group_chat.messages if msg.get("content")
    ]
    output = final_messages[-1] if final_messages else "(no output)"
    print(f"\nFinal output:\n{output[:500]}{'...' if len(output) > 500 else ''}\n")
    stratix.emit_output(output[:1000])

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


if __name__ == "__main__":
    main()
