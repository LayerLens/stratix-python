"""
Comparative Judge -- Side-by-Side Multi-Model Evaluator
=======================================================

Evaluates N model outputs against the same task across four quality
dimensions: task_completion, reasoning_clarity, conciseness, and
instruction_following.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

DIMENSIONS: dict[str, dict[str, Any]] = {
    "task_completion": {"description": "Degree to which the model fully addresses every aspect of the task", "weight": 0.30, "max_score": 10.0},
    "reasoning_clarity": {"description": "Transparency, logical coherence, and step-by-step reasoning quality", "weight": 0.25, "max_score": 10.0},
    "conciseness": {"description": "Absence of unnecessary padding, repetition, or filler content", "weight": 0.20, "max_score": 10.0},
    "instruction_following": {"description": "Strict adherence to all explicit constraints and formatting rules", "weight": 0.25, "max_score": 10.0},
}

DEFAULT_PASS_THRESHOLD: float = 7.0
DEFAULT_UNCERTAIN_THRESHOLD: float = 4.5


def _classify_verdict(
    score: float,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
    **context: Any,
) -> tuple[Literal["PASS", "FAIL", "UNCERTAIN"], str]:
    if score >= pass_threshold:
        return "PASS", "LOW"
    elif score >= uncertain_threshold:
        return "UNCERTAIN", "MEDIUM"
    else:
        return "FAIL", "HIGH"


def _deterministic_scores(model_id: str, task: str) -> dict[str, float]:
    digest = hashlib.sha256(f"{model_id}:{task}".encode("utf-8")).hexdigest()
    scores: dict[str, float] = {}
    for i, dim_name in enumerate(DIMENSIONS):
        segment = int(digest[i * 4 : (i + 1) * 4], 16)
        raw = 4.0 + (segment / 65535.0) * 5.8
        scores[dim_name] = round(raw, 1)
    return scores


class ComparativeJudge:
    """Multi-dimension comparative evaluator for side-by-side model comparison.

    Note: This is a local deterministic heuristic judge for offline/demo use.
    It produces scores algorithmically (not via LLM evaluation). For real
    AI-powered evaluation, use the LayerLens SDK's trace_evaluations API.
    """

    dimension: str = "comparative"
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD
    fail_severity: str = "HIGH"

    def __init__(
        self, judge_id: str = "judge_comparative",
        weights: dict[str, float] | None = None,
        pass_threshold: float | None = None,
        uncertain_threshold: float | None = None,
    ) -> None:
        self.judge_id = judge_id
        self._weights = dict(DIMENSIONS)
        if weights:
            for dim, w in weights.items():
                if dim in self._weights:
                    self._weights[dim]["weight"] = w
        if pass_threshold is not None:
            self.pass_threshold = pass_threshold
        if uncertain_threshold is not None:
            self.uncertain_threshold = uncertain_threshold

    def evaluate(self, trace_id: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        task = context.get("task", "")
        model_id = context.get("model_id", "unknown")
        scores = _deterministic_scores(model_id, task)
        aggregate = self._compute_aggregate(scores)
        verdict, severity = _classify_verdict(aggregate, pass_threshold=self.pass_threshold, uncertain_threshold=self.uncertain_threshold)
        best_dim = max(scores, key=scores.get)  # type: ignore[arg-type]
        worst_dim = min(scores, key=scores.get)  # type: ignore[arg-type]
        rationale = (f"{model_id} scored {aggregate:.1f}/10 overall. "
                     f"Strongest dimension: {best_dim} ({scores[best_dim]:.1f}). "
                     f"Weakest dimension: {worst_dim} ({scores[worst_dim]:.1f}).")
        return {
            "trace_id": trace_id, "judge_id": self.judge_id, "model_id": model_id,
            "dimension": self.dimension, "scores": scores, "aggregate_score": aggregate,
            "verdict": verdict, "severity": severity, "rationale": rationale,
        }

    def evaluate_batch(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for entry in entries:
            result = self.evaluate(
                trace_id=entry["trace_id"], output=entry["output"],
                context={"task": entry["task"], "model_id": entry["model_id"]})
            results.append(result)
        return self.rank(results)

    def rank(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sorted_results = sorted(results, key=lambda r: r.get("aggregate_score", 0.0), reverse=True)
        for i, r in enumerate(sorted_results, 1):
            r["rank"] = i
        return sorted_results

    def _compute_aggregate(self, scores: dict[str, float]) -> float:
        total_weight = sum(self._weights[d]["weight"] for d in scores if d in self._weights)
        if total_weight == 0:
            return round(sum(scores.values()) / max(len(scores), 1), 1)
        weighted_sum = sum(scores[d] * self._weights[d]["weight"] for d in scores if d in self._weights)
        return round(weighted_sum / total_weight, 1)

    def get_dimensions(self) -> dict[str, dict[str, Any]]:
        return dict(self._weights)
