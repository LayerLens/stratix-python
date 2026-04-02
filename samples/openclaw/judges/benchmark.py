"""
BenchmarkJudge -- Multi-Method Scoring Against Golden Answers
==============================================================

Evaluates model outputs against golden answers using semantic_similarity,
rubric, or exact_match scoring methods.
"""

from __future__ import annotations

import re
import math
import random
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

RUBRIC_CRITERIA: dict[str, dict[str, Any]] = {
    "accuracy": {"description": "Factual correctness relative to golden answer", "weight": 0.40},
    "completeness": {"description": "Covers all key points in the golden answer", "weight": 0.35},
    "formatting": {"description": "Proper structure, grammar, and presentation", "weight": 0.25},
}


class BenchmarkJudge:
    """Multi-method benchmark judge for golden-answer comparison.

    Note: This is a local deterministic heuristic judge for offline/demo use.
    It produces scores algorithmically (not via LLM evaluation). For real
    AI-powered evaluation, use the LayerLens SDK's trace_evaluations API.
    """

    dimension: str = "benchmark_accuracy"
    pass_threshold: float = 7.0
    fail_severity: str = "warning"
    SCORING_METHODS: set[str] = {"semantic_similarity", "rubric", "exact_match"}

    def __init__(self) -> None:
        self._method_scores: dict[str, list[float]] = {m: [] for m in self.SCORING_METHODS}

    def evaluate(self, trace_id: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        golden = context.get("golden_answer", "")
        method = context.get("scoring_method", "semantic_similarity")
        weight = context.get("weight", 1.0)
        task_id = context.get("task_id", trace_id)
        model_id = context.get("model_id", "unknown")
        if method not in self.SCORING_METHODS:
            method = "semantic_similarity"

        if method == "semantic_similarity":
            scores = self._score_semantic(output, golden)
        elif method == "rubric":
            scores = self._score_rubric(output, golden, context.get("rubric_criteria"))
        else:
            scores = self._score_exact(output, golden)

        aggregate = self._compute_aggregate(scores, method)
        verdict = self._classify_verdict(aggregate, method=method, weight=weight)
        self._method_scores[method].append(aggregate)
        rationale = self._build_rationale(scores, aggregate, verdict, method, task_id)
        return {
            "trace_id": trace_id,
            "task_id": task_id,
            "model_id": model_id,
            "scoring_method": method,
            "scores": scores,
            "aggregate_score": round(aggregate, 2),
            "verdict": verdict,
            "rationale": rationale,
            "weight": weight,
        }

    def get_method_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for method, scores in self._method_scores.items():
            if not scores:
                stats[method] = {"count": 0, "mean": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
                continue
            n = len(scores)
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n
            stats[method] = {
                "count": n,
                "mean": round(mean, 3),
                "std_dev": round(math.sqrt(variance), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
            }
        return stats

    def _classify_verdict(self, score: float, **context: Any) -> str:
        method = context.get("method", "semantic_similarity")
        if method == "exact_match":
            return "PASS" if score >= 5.0 else "FAIL"
        return "PASS" if score >= self.pass_threshold else "FAIL"

    def _score_semantic(self, output: str, golden: str) -> dict[str, float]:
        out_tokens = set(self._normalize_tokens(output))
        gold_tokens = set(self._normalize_tokens(golden))
        if not gold_tokens:
            return {"semantic_similarity": 5.0}
        intersection = out_tokens & gold_tokens
        union = out_tokens | gold_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        recall = len(intersection) / len(gold_tokens) if gold_tokens else 0.0
        raw_score = (jaccard * 0.4 + recall * 0.6) * 10.0
        seed = int(hashlib.md5((output + golden).encode()).hexdigest()[:8], 16)
        noise = random.Random(seed).uniform(-0.5, 0.5)
        score = max(0.0, min(10.0, raw_score + noise))
        return {"semantic_similarity": round(score, 2)}

    def _score_rubric(
        self, output: str, golden: str, custom_criteria: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, float]:
        criteria = custom_criteria or RUBRIC_CRITERIA
        out_tokens = set(self._normalize_tokens(output))
        gold_tokens = set(self._normalize_tokens(golden))
        recall = len(out_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0.5
        seed = int(hashlib.md5(output.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        scores: dict[str, float] = {}
        for criterion_name in criteria:
            base = recall * 10.0
            noise = rng.uniform(-1.5, 1.5)
            if criterion_name == "accuracy":
                base = recall * 10.0 + rng.uniform(-0.5, 0.5)
            elif criterion_name == "completeness":
                if len(output.split()) < len(golden.split()) * 0.3:
                    base -= 2.0
            elif criterion_name == "formatting":
                if output and output[0].isupper() and output.rstrip().endswith("."):
                    noise += 1.0
            scores[criterion_name] = round(max(0.0, min(10.0, base + noise)), 2)
        return scores

    def _score_exact(self, output: str, golden: str) -> dict[str, float]:
        norm_output = " ".join(output.lower().split())
        norm_golden = " ".join(golden.lower().split())
        return {"exact_match": 10.0 if norm_output == norm_golden else 0.0}

    def _compute_aggregate(self, scores: dict[str, float], method: str) -> float:
        if method == "rubric":
            criteria = RUBRIC_CRITERIA
            weighted_sum = 0.0
            weight_sum = 0.0
            for dim, score in scores.items():
                w = criteria.get(dim, {}).get("weight", 1.0)
                weighted_sum += score * w
                weight_sum += w
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        if scores:
            return list(scores.values())[0]
        return 0.0

    def _normalize_tokens(self, text: str) -> list[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "it",
            "its",
            "this",
            "that",
            "and",
            "or",
            "but",
            "not",
            "no",
        }
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _build_rationale(
        self, scores: dict[str, float], aggregate: float, verdict: str, method: str, task_id: str
    ) -> str:
        parts = [f"Task: {task_id}.", f"Scoring method: {method}.", f"Aggregate: {aggregate:.2f}/10."]
        if method == "rubric":
            weakest = min(scores, key=scores.get)  # type: ignore[arg-type]
            strongest = max(scores, key=scores.get)  # type: ignore[arg-type]
            parts.append(
                f"Strongest criterion: {strongest} ({scores[strongest]:.1f}), weakest: {weakest} ({scores[weakest]:.1f})."
            )
        elif method == "exact_match":
            matched = scores.get("exact_match", 0.0) >= 10.0
            parts.append("Exact match: YES." if matched else "Exact match: NO.")
        parts.append(
            "Output meets benchmark quality bar."
            if verdict == "PASS"
            else "Output does not meet benchmark threshold. Review for regressions."
        )
        return " ".join(parts)
