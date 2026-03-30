"""
Code Quality Judge -- Multi-Dimension Code Evaluation with Gate
================================================================

Evaluates generated code across five quality dimensions (correctness,
clarity, security, test_coverage, spec_adherence) and enforces a binary
PASS/FAIL quality gate.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

CODE_DIMENSIONS: dict[str, dict[str, Any]] = {
    "correctness": {"description": "Produces correct results for the given specification", "weight": 0.30, "max_score": 10.0},
    "clarity": {"description": "Readable, well-structured, and maintainable code", "weight": 0.15, "max_score": 10.0},
    "security": {"description": "Free of vulnerabilities, injections, and unsafe operations", "weight": 0.25, "max_score": 10.0},
    "test_coverage": {"description": "Edge cases, error paths, and core logic adequately tested", "weight": 0.15, "max_score": 10.0},
    "spec_adherence": {"description": "Implementation matches every requirement in the specification", "weight": 0.15, "max_score": 10.0},
}

DEFAULT_GATE_THRESHOLD: float = 7.5

SUGGESTION_TEMPLATES: dict[str, list[str]] = {
    "correctness": [
        "Add input validation for edge cases (null, empty, overflow).",
        "Verify return values match the specified output types.",
        "Test with boundary values from the specification.",
    ],
    "clarity": [
        "Extract complex expressions into named variables.",
        "Add docstrings to public functions and classes.",
        "Reduce nesting depth -- consider early returns.",
    ],
    "security": [
        "Sanitize all external inputs before processing.",
        "Avoid hardcoded credentials or secrets in source code.",
        "Use parameterized queries instead of string concatenation.",
    ],
    "test_coverage": [
        "Add tests for error paths and exception handling.",
        "Include at least one test per public function.",
        "Test boundary conditions (empty input, max length, negative values).",
    ],
    "spec_adherence": [
        "Cross-reference each specification requirement with implementation.",
        "Verify output format matches the spec (JSON schema, field names).",
        "Ensure all optional parameters have documented default behavior.",
    ],
}


def _classify_verdict(score: float, gate_threshold: float = DEFAULT_GATE_THRESHOLD, **context: Any) -> tuple[Literal["PASS", "FAIL"], str]:
    if score >= gate_threshold:
        return "PASS", "LOW"
    else:
        return "FAIL", "HIGH"


def _deterministic_scores(task: str, iteration: int) -> dict[str, float]:
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
    scores: dict[str, float] = {}
    for i, dim_name in enumerate(CODE_DIMENSIONS):
        segment = int(digest[i * 4 : (i + 1) * 4], 16)
        base = 4.0 + (segment / 65535.0) * 3.0
        improvement = min(1.2 * iteration, 4.5)
        dim_variance = (segment % 10) / 10.0 - 0.5
        raw = base + improvement + dim_variance
        scores[dim_name] = round(min(max(raw, 3.0), 10.0), 1)
    return scores


class CodeQualityJudge:
    """Multi-dimension code quality evaluator with binary gate enforcement.

    Note: This is a local deterministic heuristic judge for offline/demo use.
    It produces scores algorithmically (not via LLM evaluation). For real
    AI-powered evaluation, use the LayerLens SDK's trace_evaluations API.
    """

    dimension: str = "code_quality"
    pass_threshold: float = DEFAULT_GATE_THRESHOLD
    fail_severity: str = "HIGH"

    def __init__(self, judge_id: str = "judge_code_quality", gate_threshold: float | None = None, weights: dict[str, float] | None = None) -> None:
        self.judge_id = judge_id
        self._gate_threshold = gate_threshold or DEFAULT_GATE_THRESHOLD
        self._weights = dict(CODE_DIMENSIONS)
        if weights:
            for dim, w in weights.items():
                if dim in self._weights:
                    self._weights[dim]["weight"] = w
        self.pass_threshold = self._gate_threshold

    def evaluate(self, trace_id: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        task = context.get("task", "")
        iteration = context.get("iteration", 1)
        scores = _deterministic_scores(task, iteration)
        aggregate = self._compute_aggregate(scores)
        verdict, severity = _classify_verdict(aggregate, gate_threshold=self._gate_threshold)
        suggestions = self.get_suggestions(scores)
        best_dim = max(scores, key=scores.get)  # type: ignore[arg-type]
        worst_dim = min(scores, key=scores.get)  # type: ignore[arg-type]
        if verdict == "PASS":
            rationale = (f"Code PASSED the quality gate (iteration {iteration}). "
                         f"Aggregate: {aggregate:.1f} / gate: {self._gate_threshold:.1f}. "
                         f"Strongest: {best_dim} ({scores[best_dim]:.1f}). "
                         f"{len(suggestions)} minor suggestion(s).")
        else:
            gap = self._gate_threshold - aggregate
            rationale = (f"Code FAILED the quality gate (iteration {iteration}). "
                         f"Aggregate: {aggregate:.1f} / gate: {self._gate_threshold:.1f} (gap: {gap:.1f}). "
                         f"Weakest: {worst_dim} ({scores[worst_dim]:.1f}). "
                         f"{len(suggestions)} suggestion(s) for improvement.")
        return {
            "trace_id": trace_id, "judge_id": self.judge_id, "dimension": self.dimension,
            "scores": scores, "aggregate_score": aggregate, "verdict": verdict,
            "severity": severity, "suggestions": suggestions, "rationale": rationale,
            "gate_threshold": self._gate_threshold, "iteration": iteration,
            "metadata": {"dimensions_evaluated": list(self._weights.keys()),
                         "weights": {d: v["weight"] for d, v in self._weights.items()}},
        }

    def get_suggestions(self, scores: dict[str, float], suggestion_threshold: float = 7.0) -> list[str]:
        suggestions: list[str] = []
        for dim, score in scores.items():
            if score < suggestion_threshold and dim in SUGGESTION_TEMPLATES:
                templates = SUGGESTION_TEMPLATES[dim]
                idx = int(score * 10) % len(templates)
                suggestions.append(f"[{dim}] {templates[idx]}")
        return suggestions

    def _compute_aggregate(self, scores: dict[str, float]) -> float:
        total_weight = sum(self._weights[d]["weight"] for d in scores if d in self._weights)
        if total_weight == 0:
            return round(sum(scores.values()) / max(len(scores), 1), 1)
        weighted_sum = sum(scores[d] * self._weights[d]["weight"] for d in scores if d in self._weights)
        return round(weighted_sum / total_weight, 1)

    def get_dimensions(self) -> dict[str, dict[str, Any]]:
        return dict(self._weights)

    @property
    def gate_threshold(self) -> float:
        return self._gate_threshold
