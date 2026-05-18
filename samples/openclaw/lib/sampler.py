"""
StratifiedSampler -- Population-Level Post Sampling
=====================================================

Samples content feed posts using stratified sampling across communities,
karma tiers, and recency buckets.  Generates synthetic post data for
demo purposes.
"""

from __future__ import annotations

import uuid
import random
import hashlib
import logging
from typing import Any, Dict
from datetime import datetime, timezone, timedelta

from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


class ContentFeedPost(BaseModel):
    """A single content feed post for evaluation."""

    post_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_id: str = ""
    community: str = "general"
    karma_tier: str = "standard"
    recency_bucket: str = "recent"
    content: str = ""
    topic: str = ""
    word_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


_COMMUNITY_TOPICS: dict[str, list[str]] = {
    "general": [
        "What's the best approach to learning a new skill quickly?",
        "I've been thinking about productivity systems and here's what works",
        "Hot take: remote work is overrated for junior developers",
        "My experience switching careers into tech after 10 years",
        "The most underrated life advice I've ever received",
        "Why mentorship matters more than bootcamps",
        "Unpopular opinion: meetings aren't always bad",
        "How I organize my digital life in 2026",
    ],
    "coding": [
        "Why I switched from REST to GraphQL and regretted it",
        "A deep dive into Rust's borrow checker edge cases",
        "Building a real-time streaming pipeline with Kafka",
        "The hidden costs of microservices nobody talks about",
        "My take on the TypeScript vs. JavaScript debate in 2026",
        "How we reduced our CI pipeline from 45 to 3 minutes",
        "Python 3.14 pattern matching: practical use cases",
        "Zero-downtime database migrations at scale",
    ],
    "research": [
        "New paper on transformer attention head pruning shows 40% speedup",
        "Reproducibility crisis in ML: my experience replicating SOTA results",
        "A novel approach to curriculum learning for LLM fine-tuning",
        "Why RLHF might be a dead end: evidence from recent studies",
        "Scaling laws revisited: what happens beyond 1T parameters",
        "The case for smaller, specialized models over general-purpose LLMs",
        "Evaluating LLMs: why benchmarks fail and what to do instead",
        "Constitutional AI vs. RLHF: a comparative analysis",
    ],
    "creative": [
        "I wrote a novel entirely with AI assistance: here's what I learned",
        "The ethics of AI-generated art: a practitioner's perspective",
        "Prompt engineering as a creative discipline, not a technical one",
        "How AI is changing music composition: tools I actually use",
        "Building interactive fiction with LLMs: a tutorial",
        "Why AI-generated content still needs human editorial judgment",
        "The future of collaborative human-AI storytelling",
        "My workflow for AI-assisted video production",
    ],
}

_KARMA_RESPONSE_QUALITY: dict[str, dict[str, Any]] = {
    "low": {
        "min_words": 20,
        "max_words": 80,
        "coherence_modifier": -1.5,
        "patterns": [
            "yeah i think {topic} is interesting. idk though lol",
            "not sure about this but {topic}. anyway good post",
            "i guess {topic}. random thought but whatever",
            "{topic} -- honestly haven't thought about it much",
        ],
    },
    "standard": {
        "min_words": 60,
        "max_words": 200,
        "coherence_modifier": 0.0,
        "patterns": [
            (
                "I've been thinking about {topic}. In my experience, there are a few important "
                "factors to consider. First, the technical implications are significant. Second, "
                "the community impact shouldn't be underestimated. I'd love to hear other perspectives on this."
            ),
            (
                "Great topic. {topic} is something I deal with daily. The key challenge is "
                "balancing quality with speed. I've found that iterative approaches work best -- "
                "start simple, measure, and improve. However, this requires discipline and good tooling."
            ),
            (
                "This is worth discussing. {topic} has been evolving rapidly over the past year. "
                "The evidence suggests that conventional approaches are becoming less effective. "
                "We need to rethink our assumptions and consider alternative frameworks."
            ),
        ],
    },
    "high": {
        "min_words": 150,
        "max_words": 400,
        "coherence_modifier": 1.5,
        "patterns": [
            (
                "I've spent considerable time researching {topic}, and I want to share a nuanced "
                "perspective. The conventional wisdom holds that the primary bottleneck is computational, "
                "but my analysis suggests it's actually an architectural issue. Evidence from "
                "peer-reviewed studies indicates that rethinking the fundamental approach yields "
                "3-5x better results than simply scaling the existing paradigm. Specifically, "
                "there are three underexplored dimensions: first, the interaction between latency "
                "and throughput at scale creates non-linear degradation patterns. Second, the "
                "assumption of homogeneous workloads breaks down in production. Third, monitoring "
                "and observability gaps mean we're often optimizing the wrong bottleneck. My hypothesis "
                "is that a systematic, measurement-driven approach to {topic} would reveal opportunities "
                "that pure engineering intuition misses. I'd be interested in counterarguments."
            ),
            (
                "Challenging the assumption that {topic} is well-understood. After reviewing the "
                "latest research and my own experiments, I believe we're approaching this from the "
                "wrong angle. The data suggests a counterintuitive relationship between the variables "
                "most practitioners focus on. Published studies show that the correlation between "
                "effort and outcome follows a logarithmic curve, not linear. This has profound "
                "implications for resource allocation. Furthermore, a novel approach I've been "
                "testing combines elements from adjacent fields to create a more robust framework. "
                "Early results are promising: 40% improvement in key metrics with less computational "
                "overhead. The key insight is that domain-specific optimization outperforms generic "
                "solutions, but only when guided by careful measurement. I've open-sourced my "
                "benchmark suite for reproducibility."
            ),
        ],
    },
}

_RECENCY_BUCKETS: dict[str, dict[str, Any]] = {
    "recent": {"hours_ago_min": 0, "hours_ago_max": 24, "weight": 0.50},
    "this_week": {"hours_ago_min": 24, "hours_ago_max": 168, "weight": 0.30},
    "historical": {"hours_ago_min": 168, "hours_ago_max": 720, "weight": 0.20},
}


class StratifiedSampler:
    """Stratified post sampler for content feed population monitoring."""

    DEFAULT_COMMUNITIES: list[str] = ["general", "coding", "research"]
    DEFAULT_KARMA_TIERS: list[str] = ["low", "standard", "high"]
    DEFAULT_KARMA_DISTRIBUTION: dict[str, float] = {"low": 0.20, "standard": 0.55, "high": 0.25}

    def __init__(
        self,
        communities: list[str] | None = None,
        karma_tiers: list[str] | None = None,
        recency_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.communities = communities or self.DEFAULT_COMMUNITIES
        self.karma_tiers = karma_tiers or self.DEFAULT_KARMA_TIERS
        self._recency_weights = recency_weights or {k: v["weight"] for k, v in _RECENCY_BUCKETS.items()}
        self._rng = random.Random(seed)
        self._sample_count: int = 0

    def sample(self, batch_size: int = 50) -> list[ContentFeedPost]:
        posts: list[ContentFeedPost] = []
        per_community = max(1, batch_size // len(self.communities))
        remainder = batch_size - (per_community * len(self.communities))
        for i, community in enumerate(self.communities):
            count = per_community + (1 if i < remainder else 0)
            for j in range(count):
                post = self._generate_post(community, j)
                posts.append(post)
                self._sample_count += 1
        self._rng.shuffle(posts)
        return posts

    def get_sample_stats(self) -> dict[str, Any]:
        return {"total_sampled": self._sample_count, "communities": self.communities, "karma_tiers": self.karma_tiers}

    def _generate_post(self, community: str, index: int) -> ContentFeedPost:
        karma_tier = self._pick_karma_tier()
        recency = self._pick_recency_bucket()
        topic = self._pick_topic(community, index)
        content = self._generate_content(topic, karma_tier, community, index)
        timestamp = self._generate_timestamp(recency)
        seed_str = f"{community}-{karma_tier}-{index}"
        agent_hash = hashlib.md5(seed_str.encode()).hexdigest()[:8]
        return ContentFeedPost(
            agent_id=f"agent-{agent_hash}",
            community=community,
            karma_tier=karma_tier,
            recency_bucket=recency,
            content=content,
            topic=topic,
            word_count=len(content.split()),
            timestamp=timestamp,
            metadata={"synthetic": True, "sampler_version": "1.0.0"},
        )

    def _pick_karma_tier(self) -> str:
        tiers = list(self.DEFAULT_KARMA_DISTRIBUTION.keys())
        weights = [self.DEFAULT_KARMA_DISTRIBUTION[t] for t in tiers if t in self.karma_tiers]
        available_tiers = [t for t in tiers if t in self.karma_tiers]
        if not available_tiers:
            return "standard"
        return self._rng.choices(available_tiers, weights=weights, k=1)[0]

    def _pick_recency_bucket(self) -> str:
        buckets = list(self._recency_weights.keys())
        weights = [self._recency_weights[b] for b in buckets]
        return self._rng.choices(buckets, weights=weights, k=1)[0]

    def _pick_topic(self, community: str, index: int) -> str:
        topics = _COMMUNITY_TOPICS.get(community, _COMMUNITY_TOPICS["general"])
        return topics[index % len(topics)]

    def _generate_content(self, topic: str, karma_tier: str, community: str, index: int) -> str:
        quality = _KARMA_RESPONSE_QUALITY.get(karma_tier, _KARMA_RESPONSE_QUALITY["standard"])
        patterns = quality["patterns"]
        pattern = patterns[index % len(patterns)]
        return pattern.format(topic=topic)

    def _generate_timestamp(self, recency_bucket: str) -> str:
        bucket = _RECENCY_BUCKETS.get(recency_bucket, _RECENCY_BUCKETS["recent"])
        hours_ago = self._rng.uniform(bucket["hours_ago_min"], bucket["hours_ago_max"])
        ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return ts.isoformat()
