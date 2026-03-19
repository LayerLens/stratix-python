#!/usr/bin/env python3
"""
CrewAI Multi-Agent Crew with STRATIX Instrumentation

Demonstrates a multi-agent crew with task delegation, traced by STRATIX
to capture agent handoffs, tool usage, and task completion.

Requirements:
    pip install crewai crewai-tools

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from crewai import Agent, Task, Crew, Process
except ImportError:
    sys.exit(
        "This sample requires crewai. Install with:\n"
        "  pip install crewai crewai-tools"
    )

from layerlens.instrument import STRATIX, emit_handoff, emit_tool_call


def main():
    parser = argparse.ArgumentParser(description="CrewAI multi-agent crew with STRATIX tracing")
    parser.add_argument("--topic", default="the future of AI agent observability")
    parser.add_argument("--agent-id", default="crewai-crew-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="crewai",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(f"Research and write about: {args.topic}")

    # Define agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Produce comprehensive research findings on {args.topic}",
        backstory=(
            "You are a seasoned research analyst with deep expertise in AI "
            "and technology trends. You excel at finding key insights."
        ),
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Transform research into a compelling, well-structured article",
        backstory=(
            "You are a skilled technical writer who makes complex topics "
            "accessible. You produce clear, engaging content."
        ),
        verbose=True,
        allow_delegation=False,
    )

    editor = Agent(
        role="Editor-in-Chief",
        goal="Ensure the final article is polished, accurate, and publication-ready",
        backstory=(
            "You are a meticulous editor with years of experience in tech "
            "publishing. You catch errors and improve clarity."
        ),
        verbose=True,
        allow_delegation=True,
    )

    # Define tasks
    research_task = Task(
        description=f"Research {args.topic} thoroughly. Identify key trends, challenges, and opportunities.",
        expected_output="A detailed research brief with at least 5 key findings.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a 300-word article based on the research findings.",
        expected_output="A well-structured article with introduction, body, and conclusion.",
        agent=writer,
    )

    editing_task = Task(
        description="Review and polish the article for publication.",
        expected_output="A final, publication-ready article with any corrections applied.",
        agent=editor,
    )

    # Assemble and run crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )

    print(f"Topic: {args.topic}\n")
    print("Running crew (this may take a minute)...\n")
    result = crew.kickoff()

    output = str(result)
    print(f"Result:\n{output[:500]}{'...' if len(output) > 500 else ''}\n")
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
