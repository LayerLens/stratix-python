"""
PopulationQualityJudge -- Batch Content Quality Evaluator
==========================================================

Evaluates AI-generated content feed posts across four dimensions:
reasoning_coherence, factual_plausibility, task_focus, originality.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)

_REASONING_SIGNALS = ["therefore", "because", "consequently", "this implies", "it follows that",
                      "given that", "as a result", "hence", "thus", "for this reason",
                      "building on", "in contrast", "however", "on the other hand", "evidence suggests"]
_INCOHERENCE_SIGNALS = ["anyway", "but whatever", "i guess", "idk", "lol", "random thought",
                        "off topic", "not sure if related", "tangent:", "side note:"]
_FACTUAL_CLAIM_SIGNALS = ["studies show", "research indicates", "according to", "data suggests",
                          "statistically", "peer-reviewed", "published in", "in a 20", "x% of", "percent of"]
_GENERIC_MARKERS = ["as we all know", "it goes without saying", "needless to say", "it is well known",
                    "common knowledge", "everyone knows", "obviously", "of course", "the key takeaway",
                    "in conclusion", "to summarize"]
_NOVELTY_MARKERS = ["i haven't seen this discussed", "a less obvious angle", "counterintuitively",
                    "what's often overlooked", "a novel approach", "rethinking", "challenging the assumption",
                    "an underexplored", "my hypothesis", "original research"]


class PopulationQualityJudge:
    """4-dimension content quality judge with batch support.

    Note: This is a local deterministic heuristic judge for offline/demo use.
    It produces scores algorithmically (not via LLM evaluation). For real
    AI-powered evaluation, use the LayerLens SDK's trace_evaluations API.
    """

    dimension: str = "population_quality"
    pass_threshold: float = 6.0
    fail_severity: str = "warning"
    DIMENSIONS: dict[str, float] = {
        "reasoning_coherence": 0.30, "factual_plausibility": 0.25,
        "task_focus": 0.25, "originality": 0.20,
    }
    COMMUNITY_MODIFIERS: dict[str, dict[str, float]] = {
        "coding": {"reasoning_coherence": 1.1, "task_focus": 1.1},
        "research": {"factual_plausibility": 1.15, "originality": 1.1},
        "creative": {"originality": 1.2, "task_focus": 0.9},
        "general": {},
    }

    def __init__(self) -> None:
        self._evaluation_count: int = 0
        self._dimension_sums: dict[str, float] = {dim: 0.0 for dim in self.DIMENSIONS}

    def evaluate(self, trace_id: str, output: str, context: dict[str, Any]) -> dict[str, Any]:
        community = context.get("community", "general")
        karma_tier = context.get("karma_tier", "standard")
        topic = context.get("topic", "")
        scores = {
            "reasoning_coherence": self._score_reasoning(output),
            "factual_plausibility": self._score_factual(output),
            "task_focus": self._score_task_focus(output, topic),
            "originality": self._score_originality(output),
        }
        modifiers = self.COMMUNITY_MODIFIERS.get(community, {})
        for dim, modifier in modifiers.items():
            if dim in scores:
                scores[dim] = round(min(10.0, scores[dim] / modifier), 2)
        aggregate = round(sum(scores[dim] * weight for dim, weight in self.DIMENSIONS.items()), 2)
        verdict = self._classify_verdict(aggregate, community=community, karma_tier=karma_tier)
        self._update_stats(scores)
        rationale = self._build_rationale(scores, aggregate, verdict, community, karma_tier)
        return {
            "trace_id": trace_id, "scores": scores, "aggregate_score": aggregate,
            "verdict": verdict, "rationale": rationale, "community": community,
            "karma_tier": karma_tier, "population_stats": self.get_population_stats(),
        }

    def evaluate_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for item in items:
            result = self.evaluate(trace_id=item["trace_id"], output=item["output"], context=item.get("context", {}))
            results.append(result)
        return results

    def get_population_stats(self) -> dict[str, Any]:
        if self._evaluation_count == 0:
            return {"evaluation_count": 0, "dimension_averages": {}}
        averages = {dim: round(total / self._evaluation_count, 2) for dim, total in self._dimension_sums.items()}
        return {"evaluation_count": self._evaluation_count, "dimension_averages": averages}

    def _classify_verdict(self, score: float, **context: Any) -> str:
        threshold = self.pass_threshold
        if context.get("karma_tier") == "high" and context.get("community") == "research":
            threshold += 0.5
        return "PASS" if score >= threshold else "FAIL"

    def _score_reasoning(self, output: str) -> float:
        output_lower = output.lower()
        seed = int(hashlib.md5(output.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        base = rng.uniform(4.0, 8.5)
        bonus = min(sum(1 for s in _REASONING_SIGNALS if s in output_lower) * 0.6, 3.0)
        penalty = min(sum(1 for s in _INCOHERENCE_SIGNALS if s in output_lower) * 1.0, 4.0)
        word_count = len(output.split())
        if word_count > 200:
            bonus += 0.5
        elif word_count < 30:
            penalty += 1.0
        return round(max(0.0, min(10.0, base + bonus - penalty)), 2)

    def _score_factual(self, output: str) -> float:
        output_lower = output.lower()
        seed = int(hashlib.md5(output.encode()).hexdigest()[:6], 16)
        rng = random.Random(seed)
        base = rng.uniform(5.0, 9.0)
        claim_hits = sum(1 for s in _FACTUAL_CLAIM_SIGNALS if s in output_lower)
        if claim_hits > 0:
            base += min(claim_hits * 0.5, 2.0)
        return round(max(0.0, min(10.0, base)), 2)

    def _score_task_focus(self, output: str, topic: str) -> float:
        output_lower = output.lower()
        seed = int(hashlib.md5(output.encode()).hexdigest()[:7], 16)
        rng = random.Random(seed)
        base = rng.uniform(5.5, 9.0)
        if topic:
            topic_words = set(w.lower() for w in re.findall(r'\w+', topic) if len(w) > 3)
            output_words = set(w.lower() for w in re.findall(r'\w+', output) if len(w) > 3)
            if topic_words:
                overlap = len(topic_words & output_words) / len(topic_words)
                if overlap > 0.5:
                    base += 1.0
                elif overlap < 0.1:
                    base -= 2.0
        tangent_hits = sum(1 for s in ["tangent", "off topic", "side note", "unrelated"] if s in output_lower)
        return round(max(0.0, min(10.0, base - tangent_hits * 1.5)), 2)

    def _score_originality(self, output: str) -> float:
        output_lower = output.lower()
        seed = int(hashlib.md5(output.encode()).hexdigest()[:5], 16)
        rng = random.Random(seed)
        base = rng.uniform(4.5, 8.0)
        penalty = min(sum(1 for s in _GENERIC_MARKERS if s in output_lower) * 0.8, 3.0)
        bonus = min(sum(1 for s in _NOVELTY_MARKERS if s in output_lower) * 1.0, 3.0)
        return round(max(0.0, min(10.0, base + bonus - penalty)), 2)

    def _update_stats(self, scores: dict[str, float]) -> None:
        self._evaluation_count += 1
        for dim, score in scores.items():
            self._dimension_sums[dim] = self._dimension_sums.get(dim, 0.0) + score

    def _build_rationale(self, scores: dict[str, float], aggregate: float, verdict: str, community: str, karma_tier: str) -> str:
        strongest = max(scores, key=scores.get)  # type: ignore[arg-type]
        weakest = min(scores, key=scores.get)  # type: ignore[arg-type]
        parts = [f"Community: {community}, Karma tier: {karma_tier}.", f"Aggregate score: {aggregate:.2f}/10.",
                 f"Strongest: {strongest} ({scores[strongest]:.1f}), weakest: {weakest} ({scores[weakest]:.1f})."]
        if verdict == "FAIL":
            parts.append("Overall quality below community threshold." if scores[weakest] >= 4.0
                        else f"Critical weakness in {weakest}. Content may require review or remediation.")
        else:
            parts.append("Content meets community quality standards.")
        return " ".join(parts)
