"""
STRATIX Feedback Events

Defines event types for feedback collection (Epic 2).

Event Types:
- feedback.explicit: Human ratings, thumbs, comments
- feedback.implicit: Behavioral signals (retry, abandonment, etc.)
- feedback.annotation: Expert annotation queue results
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ExplicitFeedbackEvent(BaseModel):
    """
    Feedback Event: Explicit Feedback

    Captures deliberate human feedback on a trace or individual span.
    This is the primary mechanism for thumbs up/down ratings, numeric
    scores, and free-text comments from end users or reviewers.

    NORMATIVE:
    - At least one of rating, thumbs, or comment MUST be provided
    - rating interpretation depends on context: 0.0-1.0 for normalized
      scores, 1-5 for Likert scales
    """
    event_type: str = Field(
        default="feedback.explicit",
        description="Event type identifier",
    )
    trace_id: str = Field(
        description="The trace receiving feedback (UUID)",
    )
    span_id: str | None = Field(
        default=None,
        description="Optional span-level targeting",
    )
    rating: float | None = Field(
        default=None,
        description="Numeric rating (0.0-1.0 or 1-5 scale)",
    )
    thumbs: str | None = Field(
        default=None,
        description="Thumbs rating: 'up' | 'down'",
    )
    comment: str | None = Field(
        default=None,
        description="Free-text feedback",
    )
    user_id: str | None = Field(
        default=None,
        description="Who provided the feedback",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorical tags (e.g., ['helpful', 'accurate'])",
    )

    @model_validator(mode="after")
    def validate_at_least_one_signal(self) -> ExplicitFeedbackEvent:
        """At least one of rating, thumbs, or comment must be provided."""
        if self.rating is None and self.thumbs is None and self.comment is None:
            raise ValueError(
                "At least one of rating, thumbs, or comment must be provided"
            )
        return self

    @classmethod
    def create(
        cls,
        trace_id: str,
        rating: float | None = None,
        thumbs: str | None = None,
        comment: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> ExplicitFeedbackEvent:
        """
        Create an explicit feedback event.

        Args:
            trace_id: The trace receiving feedback
            rating: Numeric rating
            thumbs: Thumbs up/down
            comment: Free-text feedback
            span_id: Optional span-level targeting
            user_id: Who provided the feedback
            tags: Categorical tags

        Returns:
            ExplicitFeedbackEvent instance
        """
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            rating=rating,
            thumbs=thumbs,
            comment=comment,
            user_id=user_id,
            tags=tags or [],
        )


class ImplicitFeedbackEvent(BaseModel):
    """
    Feedback Event: Implicit Feedback

    Captures behavioral signals that indicate user satisfaction without
    explicit feedback. These signals are inferred from user actions
    during or after the agent interaction.

    NORMATIVE:
    - signal_type must be one of: retry, abandonment, conversion,
      escalation, correction
    - signal_data structure varies by signal_type
    """
    event_type: str = Field(
        default="feedback.implicit",
        description="Event type identifier",
    )
    trace_id: str = Field(
        description="The trace being observed (UUID)",
    )
    signal_type: str = Field(
        description=(
            "Signal type: 'retry' | 'abandonment' | 'conversion' "
            "| 'escalation' | 'correction'"
        ),
    )
    signal_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Signal-specific data (varies by signal_type)",
    )
    inferred_satisfaction: float | None = Field(
        default=None,
        description="0.0 (dissatisfied) to 1.0 (satisfied)",
    )
    session_id: str | None = Field(
        default=None,
        description="Session context for multi-turn interactions",
    )

    @classmethod
    def create(
        cls,
        trace_id: str,
        signal_type: str,
        signal_data: dict[str, Any] | None = None,
        inferred_satisfaction: float | None = None,
        session_id: str | None = None,
    ) -> ImplicitFeedbackEvent:
        """
        Create an implicit feedback event.

        Args:
            trace_id: The trace being observed
            signal_type: Type of behavioral signal
            signal_data: Signal-specific data
            inferred_satisfaction: Inferred satisfaction score
            session_id: Session context

        Returns:
            ImplicitFeedbackEvent instance
        """
        return cls(
            trace_id=trace_id,
            signal_type=signal_type,
            signal_data=signal_data or {},
            inferred_satisfaction=inferred_satisfaction,
            session_id=session_id,
        )


class AnnotationFeedbackEvent(BaseModel):
    """
    Feedback Event: Annotation Feedback

    Captures structured annotations from human reviewers working through
    annotation queues. This is the mechanism for expert-level quality
    review, typically performed asynchronously after the agent interaction.

    NORMATIVE:
    - annotator_id and queue_id are always required
    - label values are defined per annotation queue configuration;
      built-in labels are "pass" and "fail"
    """
    event_type: str = Field(
        default="feedback.annotation",
        description="Event type identifier",
    )
    trace_id: str = Field(
        description="The trace being annotated (UUID)",
    )
    span_id: str | None = Field(
        default=None,
        description="Optional span-level targeting",
    )
    annotator_id: str = Field(
        description="Human reviewer identifier",
    )
    queue_id: str = Field(
        description="Annotation queue this belongs to",
    )
    label: str = Field(
        description="Annotation label: 'pass' | 'fail' | custom label",
    )
    score: float | None = Field(
        default=None,
        description="Numeric score if applicable",
    )
    comment: str | None = Field(
        default=None,
        description="Reviewer notes",
    )
    annotation_time_ms: float | None = Field(
        default=None,
        description="Time spent annotating (milliseconds)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Queue-specific metadata",
    )

    @classmethod
    def create(
        cls,
        trace_id: str,
        annotator_id: str,
        queue_id: str,
        label: str,
        span_id: str | None = None,
        score: float | None = None,
        comment: str | None = None,
        annotation_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnnotationFeedbackEvent:
        """
        Create an annotation feedback event.

        Args:
            trace_id: The trace being annotated
            annotator_id: Human reviewer identifier
            queue_id: Annotation queue identifier
            label: Annotation label
            span_id: Optional span-level targeting
            score: Numeric score
            comment: Reviewer notes
            annotation_time_ms: Time spent annotating
            metadata: Queue-specific metadata

        Returns:
            AnnotationFeedbackEvent instance
        """
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            annotator_id=annotator_id,
            queue_id=queue_id,
            label=label,
            score=score,
            comment=comment,
            annotation_time_ms=annotation_time_ms,
            metadata=metadata or {},
        )
