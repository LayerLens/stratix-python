"""Trace generation templates."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel


class TraceCategory(str, Enum):
    LLM = "llm"
    AGENT = "agent"
    MULTI_AGENT = "multi-agent"
    RAG = "rag"
    TOOL_CALLING = "tool-calling"
    OTEL = "otel"


class TemplateParameter(BaseModel):
    name: str
    type: str = Field(description="one of: string|int|float|bool|list|dict")
    required: bool = False
    default: Any = None
    description: Optional[str] = None
    choices: Optional[List[Any]] = None


class TraceTemplate(BaseModel):
    id: str
    category: TraceCategory
    title: str
    description: Optional[str] = None
    parameters: List[TemplateParameter] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    min_traces: int = 1
    max_traces: int = 1000
    provider_hint: Optional[str] = None


def _p(name: str, type: str, **kw: Any) -> TemplateParameter:
    return TemplateParameter(name=name, type=type, **kw)


TEMPLATE_LIBRARY: Dict[str, TraceTemplate] = {
    "llm.chat.basic": TraceTemplate(
        id="llm.chat.basic",
        category=TraceCategory.LLM,
        title="LLM chat invocation",
        description="Single-turn chat completion with usage, latency and cost.",
        parameters=[
            _p("model", "string", default="gpt-4o-mini"),
            _p("prompt_tokens_avg", "int", default=300),
            _p("completion_tokens_avg", "int", default=120),
        ],
        defaults={"model": "gpt-4o-mini", "prompt_tokens_avg": 300, "completion_tokens_avg": 120},
        provider_hint="stochastic",
    ),
    "agent.tool_calling": TraceTemplate(
        id="agent.tool_calling",
        category=TraceCategory.TOOL_CALLING,
        title="Agent with tool calls",
        description="Single agent that fans out to 1-5 tools per run.",
        parameters=[
            _p("model", "string", default="gpt-4o"),
            _p("tools_per_run_max", "int", default=5),
        ],
        defaults={"model": "gpt-4o", "tools_per_run_max": 5},
        provider_hint="stochastic",
    ),
    "rag.retrieval": TraceTemplate(
        id="rag.retrieval",
        category=TraceCategory.RAG,
        title="RAG retrieval + generation",
        description="Retrieve k docs then generate a grounded answer.",
        parameters=[
            _p("model", "string", default="gpt-4o-mini"),
            _p("top_k", "int", default=5),
        ],
        defaults={"model": "gpt-4o-mini", "top_k": 5},
        provider_hint="stochastic",
    ),
    "multi_agent.handoff": TraceTemplate(
        id="multi_agent.handoff",
        category=TraceCategory.MULTI_AGENT,
        title="Multi-agent handoff",
        description="Planner → executor handoff chain.",
        parameters=[
            _p("agents", "int", default=3),
        ],
        defaults={"agents": 3},
        provider_hint="stochastic",
    ),
}
