#!/usr/bin/env python3
"""
PydanticAI Type-Safe Agent with STRATIX Instrumentation

Demonstrates a PydanticAI agent with typed tool functions and structured
output, with STRATIX tracing of tool invocations and agent reasoning.

Requirements:
    pip install pydantic-ai

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass

try:
    from pydantic_ai import Agent, RunContext
except ImportError:
    sys.exit(
        "This sample requires pydantic-ai. Install with:\n"
        "  pip install pydantic-ai"
    )

from pydantic import BaseModel, Field
from layerlens.instrument import STRATIX


# ---------------------------------------------------------------------------
# Dependencies and result models
# ---------------------------------------------------------------------------

@dataclass
class ProjectDeps:
    """Dependencies injected into the agent."""
    project_name: str
    team_size: int
    budget_usd: float
    tech_stack: list[str]


class ProjectPlan(BaseModel):
    """Structured project plan output."""
    milestones: list[str] = Field(description="Key project milestones")
    estimated_weeks: int = Field(description="Total estimated weeks")
    risk_level: str = Field(description="Risk assessment: low, medium, or high")
    recommended_tools: list[str] = Field(description="Tools and services to adopt")
    summary: str = Field(description="Executive summary of the plan")


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

project_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ProjectDeps,
    result_type=ProjectPlan,
    system_prompt=(
        "You are a senior technical project manager. Given project details, "
        "create a realistic project plan with milestones and risk assessment."
    ),
)


@project_agent.tool
async def get_team_capacity(ctx: RunContext[ProjectDeps]) -> str:
    """Calculate available team capacity in person-hours per week."""
    hours = ctx.deps.team_size * 35  # 35 productive hours per person
    return f"Team capacity: {hours} person-hours/week ({ctx.deps.team_size} engineers)"


@project_agent.tool
async def check_budget_feasibility(ctx: RunContext[ProjectDeps], estimated_cost: float) -> str:
    """Check whether estimated cost fits within budget."""
    remaining = ctx.deps.budget_usd - estimated_cost
    if remaining >= 0:
        return f"Budget OK. Estimated: ${estimated_cost:,.0f}, Remaining: ${remaining:,.0f}"
    return f"OVER BUDGET by ${abs(remaining):,.0f}. Estimated: ${estimated_cost:,.0f}, Budget: ${ctx.deps.budget_usd:,.0f}"


@project_agent.tool
async def list_tech_stack(ctx: RunContext[ProjectDeps]) -> str:
    """Return the project's current technology stack."""
    return f"Current stack: {', '.join(ctx.deps.tech_stack)}"


@project_agent.tool
async def estimate_complexity(ctx: RunContext[ProjectDeps], feature_description: str) -> str:
    """Estimate implementation complexity for a feature."""
    word_count = len(feature_description.split())
    if word_count > 30:
        return "Complexity: HIGH - detailed feature with many requirements"
    elif word_count > 15:
        return "Complexity: MEDIUM - moderate feature scope"
    return "Complexity: LOW - straightforward implementation"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_agent(args):
    """Execute the PydanticAI agent with STRATIX tracing."""
    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="pydantic_ai",
    )
    ctx = stratix.start_trial()

    prompt = (
        f"Create a project plan for: {args.project}. "
        f"Check team capacity, verify the budget can handle approximately "
        f"${args.budget * 0.8:,.0f} in costs, and assess the tech stack."
    )
    stratix.emit_input(prompt)

    deps = ProjectDeps(
        project_name=args.project,
        team_size=args.team_size,
        budget_usd=args.budget,
        tech_stack=args.stack.split(","),
    )

    print(f"Project: {args.project}")
    print(f"Team: {args.team_size} engineers | Budget: ${args.budget:,.0f}")
    print(f"Stack: {args.stack}\n")

    # Run agent
    result = await project_agent.run(prompt, deps=deps)
    plan = result.data

    print(f"--- Project Plan ---")
    print(f"Summary: {plan.summary}\n")
    print(f"Estimated duration: {plan.estimated_weeks} weeks")
    print(f"Risk level: {plan.risk_level}\n")
    print("Milestones:")
    for i, ms in enumerate(plan.milestones, 1):
        print(f"  {i}. {ms}")
    print(f"\nRecommended tools: {', '.join(plan.recommended_tools)}")

    stratix.emit_output(plan.summary)

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="PydanticAI agent with STRATIX tracing")
    parser.add_argument("--project", default="AI Agent Observability Platform MVP")
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--budget", type=float, default=150000.0)
    parser.add_argument("--stack", default="Python,FastAPI,React,PostgreSQL,LangGraph")
    parser.add_argument("--agent-id", default="pydanticai-agent-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
