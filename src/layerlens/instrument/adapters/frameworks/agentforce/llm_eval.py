"""
Agentforce LLM Evaluation Scenarios

Provides evaluation capabilities beyond agent tracing:
- Einstein completions evaluation (grading LLM responses)
- Prompt template A/B testing for Agentforce topics
- Model comparison (GPT vs Claude vs Gemini via Atlas)
- CRM outcome ground truth correlation

These scenarios use imported Agentforce session data as input
and run Stratix graders to produce evaluation scores.
"""

from __future__ import annotations

import os
import logging
from typing import Any
from dataclasses import field, dataclass

from layerlens.instrument.adapters.frameworks.agentforce.models import EvaluationResult

logger = logging.getLogger(__name__)


def _get_stratix_client() -> Any | None:
    """Lazily create a Stratix API client from environment variables."""
    api_url = os.environ.get("LAYERLENS_API_URL")
    api_key = os.environ.get("LAYERLENS_API_KEY")
    if not api_url or not api_key:
        return None
    try:
        from layerlens import Stratix as StratixClient

        return StratixClient(base_url=api_url, api_key=api_key)
    except Exception as exc:
        logger.debug("Could not create Stratix client: %s", exc)
        return None


# Default graders for Agentforce evaluation
_DEFAULT_GRADERS = ["relevance", "faithfulness", "coherence", "safety"]

# Composite score weights (aligned with Section 4.3 of integration doc)
_DEFAULT_WEIGHTS = {
    "topic_accuracy": 0.20,
    "action_correctness": 0.25,
    "response_quality": 0.20,
    "safety_compliance": 0.20,
    "crm_outcome": 0.15,
}


@dataclass
class ABTestResult:
    """Result of an A/B test between two prompt variants."""

    variant_a_scores: dict[str, float] = field(default_factory=dict)
    variant_b_scores: dict[str, float] = field(default_factory=dict)
    winner: str = ""
    significance: float = 0.0
    sample_size: int = 0


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models on the same test cases."""

    model_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    best_model: str = ""
    test_cases_evaluated: int = 0


class EinsteinEvaluator:
    """
    Evaluate Agentforce LLM responses using Stratix graders.

    Operates on imported session data (from ``AgentForceAdapter.import_sessions()``)
    and applies graders to LLM execution steps, action sequences, and agent responses.

    Usage:
        evaluator = EinsteinEvaluator(adapter=adapter)
        results = evaluator.evaluate_completions(
            session_ids=["0Xx..."],
            graders=["relevance", "faithfulness"],
        )
    """

    def __init__(
        self,
        adapter: Any = None,
        connection: Any = None,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            adapter: AgentForceAdapter instance (for session import).
            connection: SalesforceConnection (for ground truth queries).
        """
        self._adapter = adapter
        self._connection = connection
        self._client = _get_stratix_client()

    def evaluate_completions(
        self,
        session_ids: list[str],
        graders: list[str] | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate LLM completions from imported Agentforce sessions.

        Extracts LLM execution steps from the session data and runs
        the specified graders on each completion.

        Args:
            session_ids: Salesforce session IDs to evaluate.
            graders: List of grader names (defaults to relevance + faithfulness).

        Returns:
            List of EvaluationResult, one per session.
        """
        if not session_ids:
            return []

        grader_names = graders or _DEFAULT_GRADERS
        results: list[EvaluationResult] = []

        for session_id in session_ids:
            try:
                scores = self._evaluate_session(session_id, grader_names)
                composite = self._compute_composite_score(scores)
                results.append(
                    EvaluationResult(
                        session_id=session_id,
                        scores=scores,
                        composite_score=composite,
                    )
                )
            except Exception as e:
                logger.warning("Failed to evaluate session %s: %s", session_id, e)
                results.append(
                    EvaluationResult(
                        session_id=session_id,
                        errors=[str(e)],
                    )
                )

        return results

    def evaluate_topic(
        self,
        topic: str,
        graders: list[str] | None = None,
        limit: int = 100,
    ) -> list[EvaluationResult]:
        """
        Convenience method: import sessions for a topic and evaluate.

        Combines session import + grading in one call.

        Args:
            topic: Agentforce topic name to evaluate.
            graders: Grader names to run.
            limit: Maximum sessions to evaluate.

        Returns:
            List of EvaluationResult for the topic.
        """
        if not self._adapter:
            raise RuntimeError("Adapter required for evaluate_topic()")

        # Import sessions that match the topic
        events, result = self._adapter._importer.import_sessions(limit=limit)

        # Extract session IDs from imported events
        session_ids: list[str] = []
        for event in events:
            payload = event.get("payload", {})
            sid = payload.get("session_id")
            if sid and sid not in session_ids:
                session_ids.append(sid)

        return self.evaluate_completions(session_ids[:limit], graders)

    def ab_test_prompts(
        self,
        topic: str,
        variant_a: str,
        variant_b: str,
        test_cases: list[dict[str, str]] | None = None,
        graders: list[str] | None = None,
    ) -> ABTestResult:
        """
        A/B test two prompt variants for an Agentforce topic.

        Args:
            topic: The Agentforce topic being tested.
            variant_a: First prompt instruction text.
            variant_b: Second prompt instruction text.
            test_cases: List of test inputs (dicts with "input" key).
            graders: Grader names to use for scoring.

        Returns:
            ABTestResult with per-variant scores and winner.
        """
        grader_names = graders or ["relevance", "trajectory_accuracy"]
        cases = test_cases or []
        sample_size = len(cases)

        # Score each variant
        a_scores = self._score_variant(variant_a, cases, grader_names)
        b_scores = self._score_variant(variant_b, cases, grader_names)

        # Determine winner by average score across graders
        a_avg = sum(a_scores.values()) / max(len(a_scores), 1)
        b_avg = sum(b_scores.values()) / max(len(b_scores), 1)
        winner = "variant_a" if a_avg >= b_avg else "variant_b"

        return ABTestResult(
            variant_a_scores=a_scores,
            variant_b_scores=b_scores,
            winner=winner,
            significance=abs(a_avg - b_avg),
            sample_size=sample_size,
        )

    def compare_models(
        self,
        topic: str,
        models: list[str],
        test_cases: list[dict[str, str]] | None = None,
        graders: list[str] | None = None,
    ) -> ModelComparisonResult:
        """
        Compare multiple LLM models for an Agentforce topic.

        Args:
            topic: The Agentforce topic to evaluate.
            models: List of model names (e.g., ["gpt-5.3", "claude-opus-4-6"]).
            test_cases: Test inputs for evaluation.
            graders: Grader names to use.

        Returns:
            ModelComparisonResult with per-model scores and best model.
        """
        grader_names = graders or _DEFAULT_GRADERS
        cases = test_cases or []
        model_scores: dict[str, dict[str, float]] = {}

        for model in models:
            scores = self._score_model(model, topic, cases, grader_names)
            model_scores[model] = scores

        # Determine best model by highest average score
        best_model = ""
        best_avg = -1.0
        for model, scores in model_scores.items():
            avg = sum(scores.values()) / max(len(scores), 1)
            if avg > best_avg:
                best_avg = avg
                best_model = model

        return ModelComparisonResult(
            model_scores=model_scores,
            best_model=best_model,
            test_cases_evaluated=len(cases),
        )

    def correlate_outcomes(
        self,
        session_ids: list[str],
        outcome_query: str,
        evaluation_dimensions: list[str] | None = None,
    ) -> list[EvaluationResult]:
        """
        Correlate evaluation scores with CRM business outcomes.

        Args:
            session_ids: Session IDs to evaluate and correlate.
            outcome_query: SOQL query to fetch business outcomes.
            evaluation_dimensions: Grader dimensions to include.

        Returns:
            EvaluationResult list with ground_truth populated.
        """
        dimensions = evaluation_dimensions or _DEFAULT_GRADERS

        # Evaluate sessions
        results = self.evaluate_completions(session_ids, dimensions)

        # Fetch ground truth from Salesforce
        if self._connection:
            try:
                outcomes = self._connection.query(outcome_query)
                outcome_map = {r.get("CaseId", r.get("Id", "")): r for r in outcomes}
                for result in results:
                    gt = outcome_map.get(result.session_id, {})
                    if gt:
                        result.ground_truth = gt
            except Exception as e:
                logger.warning("Failed to fetch ground truth: %s", e)

        return results

    # --- Internal helpers ---

    def _evaluate_session(
        self,
        session_id: str,
        grader_names: list[str],
    ) -> dict[str, float]:
        """Run graders on a single session. Returns grader->score mapping."""
        if self._client:
            try:
                result = self._client.evaluations.create(
                    trace_id=session_id,
                    grader_ids=grader_names,
                )
                # result may be a dict or model; normalise to dict
                result_dict = result if isinstance(result, dict) else result.model_dump()
                return {g: result_dict.get("scores", {}).get(g, 0.0) for g in grader_names}
            except Exception as exc:
                logger.warning(
                    "Grader invocation failed for session %s: %s",
                    session_id,
                    exc,
                )

        logger.warning(
            "No Stratix client configured — returning 0.0 for session %s. "
            "Set LAYERLENS_API_URL and LAYERLENS_API_KEY environment variables.",
            session_id,
        )
        return dict.fromkeys(grader_names, 0.0)

    def _compute_composite_score(
        self,
        scores: dict[str, float],
    ) -> float | None:
        """Compute a weighted composite score from individual grader scores."""
        if not scores:
            return None

        total_weight = 0.0
        weighted_sum = 0.0

        # Map grader names to weight categories
        grader_to_category = {
            "topic_accuracy": "topic_accuracy",
            "tool_correctness": "action_correctness",
            "tool_adherence": "action_correctness",
            "relevance": "response_quality",
            "faithfulness": "response_quality",
            "coherence": "response_quality",
            "safety": "safety_compliance",
            "hallucination": "safety_compliance",
            "pii_detection": "safety_compliance",
        }

        for grader, score in scores.items():
            category = grader_to_category.get(grader, "response_quality")
            weight = _DEFAULT_WEIGHTS.get(category, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else None

    def _score_variant(
        self,
        prompt: str,
        test_cases: list[dict[str, str]],
        grader_names: list[str],
    ) -> dict[str, float]:
        """Score a prompt variant across test cases."""
        if not test_cases:
            logger.warning("No test cases provided for variant scoring — returning 0.0.")
            return dict.fromkeys(grader_names, 0.0)

        if self._client:
            try:
                aggregated: dict[str, float] = dict.fromkeys(grader_names, 0.0)
                for case in test_cases:
                    result = self._client.evaluations.create(
                        trace_id=case.get("trace_id", ""),
                        grader_ids=grader_names,
                        config={"prompt_override": prompt},
                    )
                    result_dict = result if isinstance(result, dict) else result.model_dump()
                    for g in grader_names:
                        aggregated[g] += result_dict.get("scores", {}).get(g, 0.0)
                n = len(test_cases)
                return {g: aggregated[g] / n for g in grader_names}
            except Exception as exc:
                logger.warning("Variant scoring failed: %s", exc)

        logger.warning(
            "No Stratix client configured — returning 0.0 for variant scoring. "
            "Set LAYERLENS_API_URL and LAYERLENS_API_KEY environment variables."
        )
        return dict.fromkeys(grader_names, 0.0)

    def _score_model(
        self,
        model: str,
        topic: str,
        test_cases: list[dict[str, str]],
        grader_names: list[str],
    ) -> dict[str, float]:
        """Score a model on test cases."""
        if not test_cases:
            logger.warning("No test cases provided for model %s — returning 0.0.", model)
            return dict.fromkeys(grader_names, 0.0)

        if self._client:
            try:
                aggregated: dict[str, float] = dict.fromkeys(grader_names, 0.0)
                for case in test_cases:
                    result = self._client.evaluations.create(
                        trace_id=case.get("trace_id", ""),
                        grader_ids=grader_names,
                        config={"model_override": model},
                    )
                    result_dict = result if isinstance(result, dict) else result.model_dump()
                    for g in grader_names:
                        aggregated[g] += result_dict.get("scores", {}).get(g, 0.0)
                n = len(test_cases)
                return {g: aggregated[g] / n for g in grader_names}
            except Exception as exc:
                logger.warning("Model %s scoring failed: %s", model, exc)

        logger.warning(
            "No Stratix client configured — returning 0.0 for model %s. "
            "Set LAYERLENS_API_URL and LAYERLENS_API_KEY environment variables.",
            model,
        )
        return dict.fromkeys(grader_names, 0.0)
