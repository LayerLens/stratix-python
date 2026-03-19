#!/usr/bin/env python3
"""
LangGraph Stateful Multi-Step Agent with STRATIX Instrumentation

Demonstrates a tool-calling agent built with LangGraph, traced end-to-end
by STRATIX including node execution, edge transitions, and tool calls.

Requirements:
    pip install langgraph langchain-openai

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Annotated, TypedDict

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
except ImportError:
    sys.exit(
        "This sample requires langgraph + langchain-openai. Install with:\n"
        "  pip install langgraph langchain-openai"
    )

from layerlens.instrument import STRATIX
from layerlens.instrument.adapters.langgraph import STRATIXLangGraphAdapter


# ---------------------------------------------------------------------------
# State & Tools
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "san francisco": "Foggy, 58F",
        "new york": "Sunny, 72F",
        "london": "Rainy, 55F",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    },
]

TOOL_DISPATCH = {"get_weather": get_weather, "calculate": calculate}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(model: str) -> StateGraph:
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            fn = TOOL_DISPATCH.get(call["name"])
            output = fn(**call["args"]) if fn else f"Unknown tool: {call['name']}"
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="LangGraph agent with STRATIX tracing")
    parser.add_argument("--query", default="What's the weather in San Francisco and what is 42 * 17?")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="langgraph-agent-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="langgraph",
    )
    adapter = STRATIXLangGraphAdapter(stratix=stratix)
    adapter.connect()
    ctx = stratix.start_trial()

    # Build and wrap graph
    graph = build_graph(args.model)
    traced_graph = adapter.wrap_graph(graph)

    # Run
    stratix.emit_input(args.query)
    print(f"Query: {args.query}\n")
    result = traced_graph.invoke({"messages": [HumanMessage(content=args.query)]})

    final_msg = result["messages"][-1].content
    print(f"Answer: {final_msg}\n")
    stratix.emit_output(final_msg)

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
