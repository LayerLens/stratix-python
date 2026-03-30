"""
Agent Evaluation -- Shared Pydantic Schemas
============================================

Common payload envelope used by all six agent evaluation demos.

Every demo wraps its domain-specific payload inside ``AgentEvalRequest``
and receives structured scores via ``AgentEvalResponse``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class EvaluatorConfig(BaseModel):
    """Configuration for a LayerLens evaluator instance."""

    evaluator_id: str = Field(description="References a LayerLens evaluator definition")
    judge_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model used as judge",
    )
    scoring_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions to score (e.g. ['accuracy', 'clarity'])",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension or aggregate pass thresholds",
    )


class EvalSubject(BaseModel):
    """Identifies the entity being evaluated."""

    agent_id: str | None = Field(default=None, description="Agent identifier")
    model_id: str | None = Field(default=None, description="LLM backend being evaluated")
    skill_id: str | None = Field(default=None, description="Skill registry identifier")
    task_id: str | None = Field(default=None, description="Task battery item ID")


# ---------------------------------------------------------------------------
# Request / Response envelopes
# ---------------------------------------------------------------------------


class AgentEvalRequest(BaseModel):
    """
    Common request envelope for all six agent evaluation demos.

    Carries the demo-specific payload alongside evaluator configuration
    and subject metadata.
    """

    demo_id: str = Field(description="Demo identifier (e.g. 'cage-match')")
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evaluation run ID",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )
    evaluator_config: EvaluatorConfig
    subject: EvalSubject
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Demo-specific content (raw output, traces, etc.)",
    )


class AgentEvalResponse(BaseModel):
    """
    Common response envelope returned by all six agent evaluation demos.

    Contains scored results, a human-readable rationale, and an optional
    verdict classification.
    """

    run_id: str
    evaluator_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    aggregate_score: float = 0.0
    verdict: str | None = Field(
        default=None,
        description=(
            "Classification: PASS | FAIL | SAFE | SUSPICIOUS | DANGEROUS "
            "| ALIGNED | DRIFT | VIOLATION"
        ),
    )
    rationale: str = ""
    evidence: list[str] | None = None
    recommendations: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Demo-specific payload models
# ---------------------------------------------------------------------------


class ModelOutput(BaseModel):
    """Output from a single model in the Cage Match demo."""

    model_id: str
    raw_output: str
    latency_ms: int = 0
    token_count: int = 0


class SkillAuditPayload(BaseModel):
    """Payload for the Skill Auditor demo."""

    skill_id: str
    skill_md_content: str = ""
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = Field(default_factory=list)
    honeypot_trigger_log: list[dict[str, Any]] = Field(default_factory=list)


class CodeGatePayload(BaseModel):
    """Payload for the Code Gate demo."""

    task_description: str
    code_diff: str = ""
    reviewer_comments: str = ""
    test_results: str = ""
    iteration_count: int = 1


class SoulProbePayload(BaseModel):
    """Payload for the Soul Red-Team demo."""

    soul_spec: str
    probe_id: str
    probe_category: str
    agent_response: str
    model_id: str


class ContentFeedPostPayload(BaseModel):
    """Payload for the Content Feed Observer demo."""

    post_id: str
    agent_id: str
    community: str = "general"
    content: str = ""
    karma_tier: str = "standard"


class BenchmarkTaskPayload(BaseModel):
    """Payload for the Heartbeat Benchmark demo."""

    task_battery_version: str
    model_id: str
    task_id: str
    raw_output: str
    golden_answer: str = ""
    latency_ms: int = 0
    scoring_method: str = "semantic_similarity"
    weight: float = 1.0
