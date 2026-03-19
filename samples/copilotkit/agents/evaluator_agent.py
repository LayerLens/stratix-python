"""LangGraph CoAgent — Evaluation Lifecycle with Human-in-the-Loop.

Orchestrates the full evaluation workflow inside CopilotKit:
  parse_intent -> select_judge -> confirm_with_user -> run_evaluation -> present_results

Uses STRATIX instrumentation to capture the agent's own trace while it evaluates
other agents' traces.

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
    OPENAI_API_KEY             - For the LLM powering the agent
"""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from layerlens import Stratix
from layerlens.instrument import STRATIX

# ── State schema ──────────────────────────────────────────────────────────────

class EvaluatorState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    judge_id: str
    dataset_id: str
    confirmed: bool
    eval_id: str
    results: dict[str, Any]
    status: str


# ── Clients ───────────────────────────────────────────────────────────────────

ll_client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY", ""))
llm = ChatOpenAI(model="gpt-4o", temperature=0)
stratix = STRATIX(policy_ref="copilotkit-eval-v1@1.0.0", agent_id="evaluator_coagent", framework="langgraph")


# ── Node implementations ─────────────────────────────────────────────────────

def parse_intent(state: EvaluatorState) -> dict:
    """Extract the user's evaluation intent from the conversation."""
    last_msg = state["messages"][-1].content if state["messages"] else ""

    with stratix.context():
        response = llm.invoke([
            {"role": "system", "content": (
                "You are an evaluation assistant. Extract the user's intent. "
                "Identify the judge name/ID and dataset name/ID if mentioned. "
                "Respond as JSON: {\"intent\": \"...\", \"judge_id\": \"...\", \"dataset_id\": \"...\"}"
            )},
            {"role": "user", "content": last_msg},
        ])

    import json
    try:
        parsed = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        parsed = {"intent": last_msg, "judge_id": "", "dataset_id": ""}

    return {
        "intent": parsed.get("intent", last_msg),
        "judge_id": parsed.get("judge_id", state.get("judge_id", "")),
        "dataset_id": parsed.get("dataset_id", state.get("dataset_id", "")),
        "status": "parsed",
    }


def select_judge(state: EvaluatorState) -> dict:
    """If no judge was specified, list available judges and recommend one."""
    if state.get("judge_id"):
        return {"status": "judge_selected"}

    judges = ll_client.judges.list()
    if not judges:
        return {
            "messages": [AIMessage(content="No judges found. Please create one first.")],
            "status": "error",
        }

    judge_list = "\n".join(f"- **{j.name}** (`{j.id}`): {j.description}" for j in judges[:5])
    return {
        "messages": [AIMessage(content=f"Available judges:\n{judge_list}\n\nWhich judge should I use?")],
        "status": "awaiting_judge",
    }


def confirm_with_user(state: EvaluatorState) -> dict:
    """Present the evaluation plan and ask for confirmation."""
    summary = (
        f"I'll run an evaluation with:\n"
        f"- **Judge**: `{state['judge_id']}`\n"
        f"- **Dataset**: `{state['dataset_id']}`\n\n"
        f"Shall I proceed? (yes/no)"
    )
    return {
        "messages": [AIMessage(content=summary)],
        "status": "awaiting_confirmation",
    }


def run_evaluation(state: EvaluatorState) -> dict:
    """Submit the evaluation to LayerLens and poll for results."""
    with stratix.context():
        evaluation = ll_client.evaluations.create(
            judge_id=state["judge_id"],
            dataset_id=state["dataset_id"],
        )

        if not evaluation:
            return {
                "messages": [AIMessage(content="Failed to create evaluation. Check your judge and dataset IDs.")],
                "status": "error",
            }

        # Poll (simplified — in production use async)
        import time
        for _ in range(30):
            updated = ll_client.evaluations.get(evaluation.id)
            if updated and updated.status in ("completed", "failed"):
                evaluation = updated
                break
            time.sleep(10)

        results = ll_client.evaluations.get_results(evaluation.id) if evaluation.status == "completed" else None

    return {
        "eval_id": evaluation.id,
        "results": results.to_dict() if results else {},
        "status": evaluation.status,
    }


def present_results(state: EvaluatorState) -> dict:
    """Format and present the evaluation results to the user."""
    if state["status"] == "failed":
        return {"messages": [AIMessage(content=f"Evaluation `{state['eval_id']}` failed. Check the dashboard for details.")]}

    results = state.get("results", {})
    scores = results.get("results", [])
    if not scores:
        return {"messages": [AIMessage(content=f"Evaluation `{state['eval_id']}` completed but returned no scored results.")]}

    avg = sum(r.get("score", 0) for r in scores) / len(scores)
    passing = sum(1 for r in scores if r.get("score", 0) >= 0.7)

    summary = (
        f"## Evaluation Results\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Eval ID | `{state['eval_id']}` |\n"
        f"| Samples | {len(scores)} |\n"
        f"| Average Score | {avg:.2%} |\n"
        f"| Pass Rate | {passing}/{len(scores)} ({passing/len(scores):.1%}) |\n"
    )
    return {"messages": [AIMessage(content=summary)], "status": "done"}


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_parse(state: EvaluatorState) -> str:
    if state.get("judge_id") and state.get("dataset_id"):
        return "confirm_with_user"
    return "select_judge"


def route_after_confirm(state: EvaluatorState) -> str:
    last_msg = state["messages"][-1].content.lower() if state["messages"] else ""
    if any(word in last_msg for word in ("yes", "proceed", "go", "confirm")):
        return "run_evaluation"
    return END


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(EvaluatorState)
builder.add_node("parse_intent", parse_intent)
builder.add_node("select_judge", select_judge)
builder.add_node("confirm_with_user", confirm_with_user)
builder.add_node("run_evaluation", run_evaluation)
builder.add_node("present_results", present_results)

builder.set_entry_point("parse_intent")
builder.add_conditional_edges("parse_intent", route_after_parse)
builder.add_edge("select_judge", "confirm_with_user")
builder.add_conditional_edges("confirm_with_user", route_after_confirm)
builder.add_edge("run_evaluation", "present_results")
builder.add_edge("present_results", END)

graph = builder.compile(interrupt_before=["confirm_with_user"])
