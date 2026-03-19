"""Tests for EvaluationResultEvent model."""

import pytest

from layerlens.instrument.schema.events.evaluation import (
    EvaluationInfo,
    EvaluationResultEvent,
)


class TestEvaluationInfo:
    """Tests for EvaluationInfo model."""

    def test_create_basic(self):
        """Test creating a basic evaluation info."""
        info = EvaluationInfo(
            dimension="factual_accuracy",
            score=0.85,
        )
        assert info.dimension == "factual_accuracy"
        assert info.score == 0.85
        assert info.label is None
        assert info.explanation is None
        assert info.grader_id is None
        assert info.threshold == 0.5

    def test_create_full(self):
        """Test creating evaluation info with all fields."""
        info = EvaluationInfo(
            dimension="safety",
            score=0.95,
            label="pass",
            explanation="Content is safe and appropriate",
            grader_id="safety_judge_v2",
            threshold=0.8,
        )
        assert info.dimension == "safety"
        assert info.score == 0.95
        assert info.label == "pass"
        assert info.explanation == "Content is safe and appropriate"
        assert info.grader_id == "safety_judge_v2"
        assert info.threshold == 0.8

    def test_score_validation_min(self):
        """Test that score must be >= 0.0."""
        with pytest.raises(ValueError):
            EvaluationInfo(dimension="test", score=-0.1)

    def test_score_validation_max(self):
        """Test that score must be <= 1.0."""
        with pytest.raises(ValueError):
            EvaluationInfo(dimension="test", score=1.1)

    def test_score_boundary_zero(self):
        """Test that score=0.0 is valid."""
        info = EvaluationInfo(dimension="test", score=0.0)
        assert info.score == 0.0

    def test_score_boundary_one(self):
        """Test that score=1.0 is valid."""
        info = EvaluationInfo(dimension="test", score=1.0)
        assert info.score == 1.0

    def test_threshold_validation_min(self):
        """Test that threshold must be >= 0.0."""
        with pytest.raises(ValueError):
            EvaluationInfo(dimension="test", score=0.5, threshold=-0.1)

    def test_threshold_validation_max(self):
        """Test that threshold must be <= 1.0."""
        with pytest.raises(ValueError):
            EvaluationInfo(dimension="test", score=0.5, threshold=1.1)


class TestEvaluationResultEvent:
    """Tests for EvaluationResultEvent model."""

    def test_create_passing(self):
        """Test creating a passing evaluation event."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="factual_accuracy",
            score=0.85,
        )
        assert event.event_type == "evaluation.result"
        assert event.trace_id == "trace-123"
        assert event.evaluation.dimension == "factual_accuracy"
        assert event.evaluation.score == 0.85
        assert event.is_passing is True
        assert event.evaluation.label == "pass"

    def test_create_failing(self):
        """Test creating a failing evaluation event."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="relevance",
            score=0.3,
        )
        assert event.is_passing is False
        assert event.evaluation.label == "fail"

    def test_create_with_custom_threshold(self):
        """Test that custom threshold affects passing status."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="safety",
            score=0.7,
            threshold=0.8,
        )
        assert event.is_passing is False
        assert event.evaluation.threshold == 0.8

    def test_create_at_threshold_boundary(self):
        """Test that score exactly at threshold is passing."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="test",
            score=0.5,
            threshold=0.5,
        )
        assert event.is_passing is True

    def test_create_with_override_passing(self):
        """Test that is_passing override works."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="test",
            score=0.3,
            is_passing=True,  # Override: force passing despite low score
        )
        assert event.is_passing is True

    def test_create_with_all_fields(self):
        """Test creating event with all optional fields."""
        event = EvaluationResultEvent.create(
            trace_id="trace-456",
            dimension="helpfulness",
            score=0.92,
            evaluation_id="eval-789",
            label="excellent",
            explanation="Very helpful and comprehensive response",
            grader_id="helpfulness_judge_v1",
            threshold=0.7,
        )
        assert event.trace_id == "trace-456"
        assert event.evaluation_id == "eval-789"
        assert event.evaluation.dimension == "helpfulness"
        assert event.evaluation.score == 0.92
        assert event.evaluation.label == "excellent"
        assert event.evaluation.explanation == "Very helpful and comprehensive response"
        assert event.evaluation.grader_id == "helpfulness_judge_v1"
        assert event.evaluation.threshold == 0.7
        assert event.is_passing is True

    def test_create_final_dimension(self):
        """Test creating a final composite score event."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="final",
            score=0.78,
            grader_id="composite_evaluator",
        )
        assert event.evaluation.dimension == "final"
        assert event.evaluation.grader_id == "composite_evaluator"

    def test_default_event_type(self):
        """Test that event_type defaults to evaluation.result."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="test",
            score=0.5,
        )
        assert event.event_type == "evaluation.result"

    def test_default_label_pass(self):
        """Test that default label is 'pass' when score >= threshold."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="test",
            score=0.8,
        )
        assert event.evaluation.label == "pass"

    def test_default_label_fail(self):
        """Test that default label is 'fail' when score < threshold."""
        event = EvaluationResultEvent.create(
            trace_id="trace-123",
            dimension="test",
            score=0.2,
        )
        assert event.evaluation.label == "fail"
