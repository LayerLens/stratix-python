"""Customer support with handoffs — LangGraph multi-agent graph.

VENDOR FORK. Upstream: LangChain / LangGraph "Build customer support with
handoffs" tutorial
(https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support,
source in github.com/langchain-ai/langgraph docs). The upstream sample routes a
customer between specialized support agents, handing the conversation off from a
triage agent to a domain specialist and finally to a closing agent.

This fork keeps that shape but expresses it as an explicit ``langgraph.graph.StateGraph``
with distinctly *named agent nodes* — ``triage`` -> ``<department>_specialist`` ->
``closer`` — so that agent-to-agent transitions are real graph node transitions.
That is exactly what the LayerLens LangGraph adapter detects and reports as
``agent.handoff`` events (see run_instrumented.py). See the ``UPSTREAM`` file for
full provenance and the exact list of changes vs upstream.

Model wiring (``build_model``): the upstream fork routed the model through
OpenRouter. This SDK reference keeps that as the primary path but adds an OpenAI
fallback (the SAME ``langchain_openai.ChatOpenAI`` class, a different base_url/key),
selected by ``VENDOR_MODEL_BACKEND`` — because no OpenRouter credential is
available in the SDK's test environment. The chosen model must support tool calling.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Model — OpenRouter (upstream) with an OpenAI fallback (this SDK reference)
# ---------------------------------------------------------------------------


def build_model(temperature: float = 0.0) -> ChatOpenAI:
    """Return a ChatOpenAI for the graph's agents.

    ``VENDOR_MODEL_BACKEND`` selects the backend:

    * ``openrouter`` (upstream default) — ``OPENROUTER_API_KEY`` +
      ``OPENROUTER_MODEL`` (default ``openai/gpt-4o-mini``), base_url the
      OpenRouter gateway.
    * ``openai`` — ``OPENAI_API_KEY`` + ``OPENAI_MODEL`` (default ``gpt-4o-mini``),
      the OpenAI endpoint. Used in the SDK environment (no OpenRouter credential).

    Both use ``langchain_openai.ChatOpenAI``; only the base_url/key/model differ.
    The chosen model must support tool calling.
    """
    backend = os.environ.get("VENDOR_MODEL_BACKEND", "openrouter").strip().lower()

    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VENDOR_MODEL_BACKEND=openrouter requires OPENROUTER_API_KEY "
                "(or set VENDOR_MODEL_BACKEND=openai to use OPENAI_API_KEY)."
            )
        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://layerlens.ai",
                "X-Title": "LayerLens vendor sample: customer-support-with-handoffs",
            },
        )

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VENDOR_MODEL_BACKEND=openai requires OPENAI_API_KEY."
            )
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
        )

    raise RuntimeError(
        f"unknown VENDOR_MODEL_BACKEND={backend!r} (use 'openrouter' or 'openai')"
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class SupportState(TypedDict, total=False):
    """State threaded through the support graph.

    ``messages`` uses LangGraph's ``add_messages`` reducer so that each node
    *appends* to the conversation rather than replacing it — required so each
    specialist reads the original customer turn, and so the CopilotKit / AG-UI
    runtime (in the full-stack upstream repo) can accumulate agent turns.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    category: str  # billing | technical | returns | unknown
    resolution: str
    handoff_log: List[str]


DEPARTMENTS = ("billing", "technical", "returns")


# ---------------------------------------------------------------------------
# Department tools (the specialists call these)
# ---------------------------------------------------------------------------


@tool
def route_to_department(department: str, reason: str) -> str:
    """Route the customer to a specialist department.

    Args:
        department: One of "billing", "technical", or "returns".
        reason: Short reason for the routing decision.
    """
    return f"routed to {department}: {reason}"


@tool
def lookup_invoice(account_id: str) -> str:
    """Look up the most recent invoice for a billing account."""
    return (
        f"invoice INV-2043 for account {account_id}: subscription $29.00 charged "
        "twice on 2026-07-01 (duplicate confirmed)"
    )


@tool
def issue_credit(account_id: str, amount_usd: float) -> str:
    """Issue an account credit to resolve an overcharge."""
    return f"credit of ${amount_usd:.2f} applied to account {account_id} (ref CR-8891)"


@tool
def run_diagnostic(device: str) -> str:
    """Run a remote diagnostic against the customer's device."""
    return f"diagnostic for {device}: firmware 3.1.2 out of date; wifi radio OK"


@tool
def create_ticket(summary: str) -> str:
    """Open an engineering support ticket."""
    return f"ticket TICK-5567 opened: {summary}"


@tool
def check_return_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for return."""
    return f"order {order_id}: within 30-day window, eligible for return"


@tool
def start_return(order_id: str) -> str:
    """Start a return / RMA for an eligible order."""
    return f"RMA RMA-3120 created for order {order_id}; prepaid label emailed"


DEPARTMENT_TOOLS: Dict[str, List[Any]] = {
    "billing": [lookup_invoice, issue_credit],
    "technical": [run_diagnostic, create_ticket],
    "returns": [check_return_eligibility, start_return],
}

DEPARTMENT_SYSTEM: Dict[str, str] = {
    "billing": (
        "You are a billing support specialist. Resolve overcharges, refunds, and "
        "invoice questions. Use your tools to look up invoices and issue credits, "
        "then explain the resolution to the customer."
    ),
    "technical": (
        "You are a technical support specialist. Diagnose device and software "
        "problems. Use your tools to run diagnostics and open tickets, then explain "
        "the next steps to the customer."
    ),
    "returns": (
        "You are a returns specialist. Handle returns, exchanges, and RMAs. Use your "
        "tools to check eligibility and start returns, then explain the process."
    ),
}


# ---------------------------------------------------------------------------
# Helper: one bounded tool-using turn (LLM -> execute tools -> LLM)
# ---------------------------------------------------------------------------


def _run_tools(tool_calls: List[Dict[str, Any]], tools: List[Any]) -> List[ToolMessage]:
    by_name = {t.name: t for t in tools}
    out: List[ToolMessage] = []
    for call in tool_calls:
        name = call["name"]
        tool_obj = by_name.get(name)
        if tool_obj is None:
            out.append(ToolMessage(content=f"unknown tool {name}", tool_call_id=call["id"]))
            continue
        result = tool_obj.invoke(call["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return out


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def triage_agent(state: SupportState) -> SupportState:
    """Classify the customer's request and choose a department."""
    model = build_model()
    system = SystemMessage(
        content=(
            "You are the front-line triage agent for customer support. Read the "
            "customer's message and route them to exactly one department by calling "
            "route_to_department with department in {billing, technical, returns}. "
            "billing = charges/refunds/invoices; technical = device/software problems; "
            "returns = returns/exchanges/RMAs."
        )
    )
    convo = [system, *state.get("messages", [])]
    response = model.bind_tools([route_to_department], tool_choice="any").invoke(convo)

    category = "unknown"
    if response.tool_calls:
        dept = str(response.tool_calls[0]["args"].get("department", "")).lower().strip()
        if dept in DEPARTMENTS:
            category = dept

    log = list(state.get("handoff_log", []))
    log.append(f"triage -> {category}")
    # triage deliberately does NOT emit ``response`` into ``messages``: its model
    # turn is an internal routing tool-call never executed as a tool — surfacing it
    # would leave an orphan AI tool-call the next LLM turn rejects with a 400.
    return {
        "category": category,
        "handoff_log": log,
    }


def _make_specialist(department: str):
    def specialist(state: SupportState) -> SupportState:
        model = build_model()
        tools = DEPARTMENT_TOOLS[department]
        system = SystemMessage(content=DEPARTMENT_SYSTEM[department])
        # Only carry the human turns forward; the triage tool-call/AI message is
        # internal routing chatter and confuses a fresh specialist prompt.
        human_turns = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        convo: List[BaseMessage] = [system, *human_turns]

        first = model.bind_tools(tools).invoke(convo)
        convo.append(first)

        resolution_text = first.content or ""
        if first.tool_calls:
            tool_messages = _run_tools(first.tool_calls, tools)
            convo.extend(tool_messages)
            final = model.bind_tools(tools).invoke(convo)
            convo.append(final)
            resolution_text = final.content or resolution_text

        log = list(state.get("handoff_log", []))
        log.append(f"{department}_specialist resolved")
        return {
            "messages": [AIMessage(content=str(resolution_text))],
            "resolution": str(resolution_text),
            "handoff_log": log,
        }

    specialist.__name__ = f"{department}_specialist"
    return specialist


def closer(state: SupportState) -> SupportState:
    """Confirm the resolution and close the conversation."""
    model = build_model()
    system = SystemMessage(
        content=(
            "You are the closing agent. In one or two sentences, confirm the "
            "resolution to the customer warmly and ask if there is anything else."
        )
    )
    resolution = state.get("resolution", "your issue has been handled")
    convo = [system, HumanMessage(content=f"Resolution so far: {resolution}")]
    response = model.invoke(convo)

    log = list(state.get("handoff_log", []))
    log.append("closer -> done")
    return {"messages": [response], "handoff_log": log}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _route_after_triage(state: SupportState) -> str:
    category = state.get("category", "unknown")
    if category in DEPARTMENTS:
        return f"{category}_specialist"
    # Unknown category: send to billing as a safe default human-style fallback.
    return "billing_specialist"


def build_graph(*, checkpointer: Any = None) -> Any:
    """Build and compile the customer-support handoff graph.

    ``triage -> (billing|technical|returns)_specialist -> closer``. The console
    scripts compile without a checkpointer; the full-stack AG-UI server (upstream
    repo) passes an ``InMemorySaver``.
    """
    graph = StateGraph(SupportState)

    graph.add_node("triage", triage_agent)
    graph.add_node("billing_specialist", _make_specialist("billing"))
    graph.add_node("technical_specialist", _make_specialist("technical"))
    graph.add_node("returns_specialist", _make_specialist("returns"))
    graph.add_node("closer", closer)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        {
            "billing_specialist": "billing_specialist",
            "technical_specialist": "technical_specialist",
            "returns_specialist": "returns_specialist",
        },
    )
    for dept in DEPARTMENTS:
        graph.add_edge(f"{dept}_specialist", "closer")
    graph.add_edge("closer", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # Uninstrumented smoke run (no LayerLens). Use run_instrumented.py for tracing.
    compiled = build_graph()
    result = compiled.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I was charged twice for my subscription this month "
                    "and I'd like the duplicate refunded to account ACC-77."
                )
            ]
        }
    )
    print("handoff_log:", result.get("handoff_log"))
    print("category:", result.get("category"))
    for msg in result["messages"]:
        msg.pretty_print()
