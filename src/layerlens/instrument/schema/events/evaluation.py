"""
STRATIX Evaluation Events

Cross-Cutting Event: Evaluation Result

Emitted by the Evaluator after computing a dimension score or final evaluation
score. Enables evaluation results to flow through the STRATIX event pipeline and
be exported via OTel with gen_ai.evaluation.* attributes.

{
    "event_type": "evaluation.result",
    "trace_id": "uuid",
    "evaluation_id": "uuid",
    "evaluation": {
        "dimension": "factual_accuracy",
        "score": 0.85,
        "label": "pass",
        "explanation": "...",
        "grader_id": "factual_accuracy_judge_v2",
        "threshold": 0.5
    },
    "is_passing": true
}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationInfo(BaseModel):
    """Evaluation result information."""
    dimension: str = Field(
        description="Dimension being evaluated (e.g., 'factual_accuracy', 'safety', 'final')"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evaluation score (0.0-1.0)"
    )
    label: str | None = Field(
        default=None,
        description="Quality label (e.g., 'pass', 'fail', 'partial')"
    )
    explanation: str | None = Field(
        default=None,
        description="Human-readable explanation of the evaluation result"
    )
    grader_id: str | None = Field(
        default=None,
        description="Identifier of the grader that produced this result"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Passing threshold for this dimension"
    )


class EvaluationResultEvent(BaseModel):
    """
    Cross-Cutting Event: Evaluation Result

    Represents an evaluation score for a trace, emitted per-dimension
    and as a final composite score.

    NORMATIVE:
    - Emit one event per dimension evaluated
    - Emit one final event with dimension="final" for the composite score
    - Include trace_id for correlation with the evaluated trace
    """
    event_type: str = Field(
        default="evaluation.result",
        description="Event type identifier"
    )
    trace_id: str = Field(
        description="The trace being evaluated"
    )
    evaluation_id: str | None = Field(
        default=None,
        description="Evaluation run identifier"
    )
    evaluation: EvaluationInfo = Field(
        description="Evaluation result details"
    )
    is_passing: bool = Field(
        description="Whether the score meets the threshold"
    )

    @classmethod
    def create(
        cls,
        trace_id: str,
        dimension: str,
        score: float,
        evaluation_id: str | None = None,
        label: str | None = None,
        explanation: str | None = None,
        grader_id: str | None = None,
        threshold: float = 0.5,
        is_passing: bool | None = None,
    ) -> EvaluationResultEvent:
        """
        Create an evaluation result event.

        Args:
            trace_id: The trace being evaluated
            dimension: Dimension name (e.g., 'factual_accuracy', 'final')
            score: Evaluation score (0.0-1.0)
            evaluation_id: Evaluation run identifier
            label: Quality label
            explanation: Human-readable explanation
            grader_id: Grader identifier
            threshold: Passing threshold
            is_passing: Override for passing status (defaults to score >= threshold)

        Returns:
            EvaluationResultEvent instance
        """
        passing = is_passing if is_passing is not None else (score >= threshold)
        return cls(
            trace_id=trace_id,
            evaluation_id=evaluation_id,
            evaluation=EvaluationInfo(
                dimension=dimension,
                score=score,
                label=label or ("pass" if passing else "fail"),
                explanation=explanation,
                grader_id=grader_id,
                threshold=threshold,
            ),
            is_passing=passing,
        )
