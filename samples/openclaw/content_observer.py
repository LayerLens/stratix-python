"""
Content Observer -- OpenClaw Population-Level Content Quality Monitor
=======================================================================

Monitors the quality of AI-generated content produced by OpenClaw-powered
agents by sampling posts across communities and karma tiers, evaluating
them through the PopulationQualityJudge, and generating an intelligence
report.

Heritage: Originally developed as the "Moltbook Observer" for monitoring
content quality on Moltbook (later Moltbot), an AI-powered social
platform.  Ported to use OpenClaw agents for content generation and
LayerLens for evaluation.  The sampling and scoring logic descends from
the population-level quality monitoring system built for Moltbook's
content feed.

Usage::

    python -m samples.openclaw.content_observer \\
        --communities general,coding,research --batch-size 50
"""

from __future__ import annotations

import uuid
import logging
import argparse
from typing import Any

from ._runner import DemoRunner
from .lib.sampler import StratifiedSampler
from .judges.population_quality import PopulationQualityJudge

logger = logging.getLogger(__name__)

DEFAULT_COMMUNITIES = "general,coding,research"
DEFAULT_BATCH_SIZE = 50


class ContentObserverRunner(DemoRunner):
    """CLI-driven orchestrator for the OpenClaw Content Observer demo."""

    demo_id = "content-observer"
    demo_name = "Content Observer"
    description = (
        "OpenClaw population-level content quality monitor: sample "
        "AI-generated posts from platform communities, evaluate quality "
        "across 4 dimensions, and generate an intelligence report."
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = super().build_parser()
        parser.add_argument("--communities", default=DEFAULT_COMMUNITIES, help="Comma-separated community list.")
        parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of posts to sample.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
        return parser

    async def run(self) -> dict[str, Any]:
        communities = [c.strip() for c in self.args.communities.split(",") if c.strip()]
        batch_size = self.args.batch_size

        sampler = StratifiedSampler(communities=communities, seed=self.args.seed)
        judge = PopulationQualityJudge()

        posts = sampler.sample(batch_size=batch_size)
        logger.info("Sampled %d posts", len(posts))

        # Generate content via OpenClaw when available, enriching sampled post stubs
        for post in posts:
            execution = self.execute_with_openclaw(
                task=f"Write a {post.community} community post about: {post.topic}",
                model="claude-sonnet-4-20250514",
                agent_name=f"content-{post.community}",
            )
            if execution["output"] != post.content:  # Real OpenClaw response
                post.content = execution["output"]

        eval_items = [
            {
                "trace_id": p.post_id,
                "output": p.content,
                "context": {
                    "community": p.community,
                    "karma_tier": p.karma_tier,
                    "agent_id": p.agent_id,
                    "topic": p.topic,
                },
            }
            for p in posts
        ]

        results = judge.evaluate_batch(eval_items)
        report = self._build_report(posts, results, communities, judge)

        if not self.args.json:
            self._print_report(report)

        # SDK trace upload and real evaluation on sampled content
        run_id = str(uuid.uuid4())
        sdk_judge_id = self.create_judge(
            name="Content Quality Observer",
            evaluation_goal="Evaluate AI-generated content for coherence, helpfulness, safety, and community guideline adherence.",
        )
        sdk_results: list[dict[str, Any]] = []
        # Evaluate a sample of posts (up to 5) to avoid excessive API calls
        sample_posts = posts[:5] if len(posts) > 5 else posts
        sample_results = results[:5] if len(results) > 5 else results
        for post, result in zip(sample_posts, sample_results):
            trace_id = self.upload_trace(
                input_text=f"Content feed post from {post.community} (karma: {post.karma_tier})",
                output_text=post.content,
                metadata={
                    "demo": self.demo_id,
                    "community": post.community,
                    "karma_tier": post.karma_tier,
                    "post_id": post.post_id,
                    "source": "openclaw",
                },
            )
            if trace_id:
                logger.info("Trace uploaded for post %s: %s", post.post_id, trace_id)
                sdk_result = self.evaluate_trace(trace_id, sdk_judge_id)
                if sdk_result:
                    sdk_results.append({"post_id": post.post_id, "community": post.community, **sdk_result})

        if sdk_results and not self.args.json:
            print(f"\n  --- SDK Evaluation (sampled {len(sdk_results)} posts) ---")
            for sr in sdk_results:
                status = "PASS" if sr["passed"] else "FAIL"
                print(f"    {sr['post_id']:<14} {sr['community']:<12} score={sr['score']:>5.2f}  [{status}]")

        return {
            "run_id": run_id,
            "communities": communities,
            "batch_size": batch_size,
            "sdk_results": sdk_results,
            **report,
        }

    def _build_report(
        self,
        posts: list,
        results: list[dict[str, Any]],
        communities: list[str],
        judge: PopulationQualityJudge | None = None,
    ) -> dict[str, Any]:
        total = len(results)
        pass_count = sum(1 for r in results if r["verdict"] == "PASS")
        pass_rate = (pass_count / total * 100) if total else 0.0
        mean_score = (sum(r["aggregate_score"] for r in results) / total) if total else 0.0

        community_stats: dict[str, dict[str, Any]] = {}
        for post, result in zip(posts, results):
            c = post.community
            if c not in community_stats:
                community_stats[c] = {"count": 0, "total_score": 0.0, "pass_count": 0}
            community_stats[c]["count"] += 1
            community_stats[c]["total_score"] += result["aggregate_score"]
            if result["verdict"] == "PASS":
                community_stats[c]["pass_count"] += 1
        community_breakdown = {
            c: {
                "count": s["count"],
                "avg_score": round(s["total_score"] / s["count"], 2) if s["count"] else 0.0,
                "pass_rate": round(s["pass_count"] / s["count"] * 100, 1) if s["count"] else 0.0,
            }
            for c, s in community_stats.items()
        }

        tier_stats: dict[str, dict[str, Any]] = {}
        for post, result in zip(posts, results):
            t = post.karma_tier
            if t not in tier_stats:
                tier_stats[t] = {"count": 0, "total_score": 0.0, "pass_count": 0}
            tier_stats[t]["count"] += 1
            tier_stats[t]["total_score"] += result["aggregate_score"]
            if result["verdict"] == "PASS":
                tier_stats[t]["pass_count"] += 1
        tier_breakdown = {
            t: {
                "count": s["count"],
                "avg_score": round(s["total_score"] / s["count"], 2) if s["count"] else 0.0,
                "pass_rate": round(s["pass_count"] / s["count"] * 100, 1) if s["count"] else 0.0,
            }
            for t, s in tier_stats.items()
        }

        dim_totals: dict[str, float] = {}
        for r in results:
            for dim, score in r["scores"].items():
                dim_totals[dim] = dim_totals.get(dim, 0.0) + score
        dim_averages = {dim: round(ts / total, 2) for dim, ts in dim_totals.items()} if total else {}

        flagged = [
            {
                "post_id": p.post_id,
                "community": p.community,
                "karma_tier": p.karma_tier,
                "score": r["aggregate_score"],
                "rationale": r["rationale"],
            }
            for p, r in zip(posts, results)
            if r["verdict"] == "FAIL"
        ]

        return {
            "total_posts": total,
            "pass_count": pass_count,
            "fail_count": total - pass_count,
            "pass_rate": round(pass_rate, 1),
            "mean_score": round(mean_score, 2),
            "community_breakdown": community_breakdown,
            "tier_breakdown": tier_breakdown,
            "dimension_averages": dim_averages,
            "flagged_posts": flagged,
            "population_stats": judge.get_population_stats() if judge else {},
        }

    def _print_report(self, report: dict[str, Any]) -> None:
        communities = list(report["community_breakdown"].keys())
        print(f"\n{'=' * 60}")
        print("  CONTENT FEED INTELLIGENCE REPORT")
        print(f"{'=' * 60}")
        print(f"  Communities: {', '.join(communities)}")
        print(f"  Sample Size: {report['total_posts']} posts")
        print(f"  Overall Pass Rate: {report['pass_rate']}%")
        print(f"  Mean Quality Score: {report['mean_score']:.2f} / 10.0")
        print(f"{'-' * 60}")
        print("  Per-Community Breakdown:")
        for comm, stats in sorted(report["community_breakdown"].items()):
            print(
                f"    {comm:<14} {stats['count']:>3} posts  avg={stats['avg_score']:.2f}  pass={stats['pass_rate']:.1f}%"
            )
        print(f"{'-' * 60}")
        print("  Per-Karma-Tier Breakdown:")
        for tier in ["low", "standard", "high"]:
            if tier in report["tier_breakdown"]:
                stats = report["tier_breakdown"][tier]
                print(
                    f"    {tier:<14} {stats['count']:>3} posts  avg={stats['avg_score']:.2f}  pass={stats['pass_rate']:.1f}%"
                )
        print(f"{'-' * 60}")
        if report["dimension_averages"]:
            print("  Dimension Averages:")
            for dim, avg in sorted(report["dimension_averages"].items()):
                bar = "#" * int(avg) + "." * (10 - int(avg))
                print(f"    {dim:<28} {avg:>5.2f}  [{bar}]")
        print(f"{'=' * 60}")
        flagged = report["flagged_posts"]
        if flagged:
            print(f"\n  Flagged Posts ({len(flagged)} FAIL):")
            for fp in flagged[:10]:
                print(
                    f"    {fp['post_id']:<14} {fp['community']:<12} tier={fp['karma_tier']:<10} score={fp['score']:.2f}"
                )
            if len(flagged) > 10:
                print(f"    ... and {len(flagged) - 10} more")
            print()


def main() -> None:
    """CLI entrypoint for the Content Observer demo."""
    ContentObserverRunner().execute()


if __name__ == "__main__":
    main()
