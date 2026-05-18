"""
Cage Match -- OpenClaw Model-vs-Model Comparative Evaluation
=============================================================

Which LLM backend should my OpenClaw agent use for this skill?

Dispatches a user-supplied task to multiple OpenClaw agents backed by
different LLM models, evaluates every response through the
ComparativeJudge across four quality dimensions, and publishes a
ranked leaderboard.  Uses the LayerLens SDK for trace upload and
real evaluation alongside local judge scoring.

Usage::

    python -m samples.openclaw.cage_match \\
        --task "Explain quantum entanglement to a 10-year-old" \\
        --models claude-sonnet-4-20250514,gpt-4o,deepseek-v3
"""

from __future__ import annotations

import uuid
import logging
import argparse
from typing import Any

from ._runner import DemoRunner, _print_scores
from .lib.schemas import (
    ModelOutput,
)
from .lib.notifier import Notifier
from .judges.comparative import ComparativeJudge

logger = logging.getLogger(__name__)

DEFAULT_MODELS = "claude-sonnet-4-20250514,gpt-4o,deepseek-v3"
DEFAULT_TASK = (
    "Explain the difference between supervised and unsupervised "
    "machine learning. Include one real-world example of each."
)


class CageMatchRunner(DemoRunner):
    """CLI-driven orchestrator for the OpenClaw Cage Match demo."""

    demo_id = "cage-match"
    demo_name = "Cage Match"
    description = (
        "OpenClaw Model-vs-Model comparative evaluation: dispatch a task "
        "to N OpenClaw agents with different LLM backends, judge outputs "
        "side-by-side, and publish a ranked leaderboard."
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = super().build_parser()
        parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model IDs.")
        parser.add_argument("--task", default=DEFAULT_TASK, help="Task prompt for all models.")
        parser.add_argument("--threshold", type=float, default=7.0, help="Pass threshold (default: 7.0).")
        parser.add_argument("--notify", default="stdout://", help="Notification channel URI.")
        return parser

    async def run(self) -> dict[str, Any]:
        models = [m.strip() for m in self.args.models.split(",") if m.strip()]
        task = self.args.task
        judge = ComparativeJudge(judge_id="judge_cage_match", pass_threshold=self.args.threshold)
        notifier = Notifier(channels=[self.args.notify])

        logger.info("Cage Match: %d OpenClaw agents competing -- %s", len(models), ", ".join(models))

        run_id = str(uuid.uuid4())
        entries: list[dict[str, Any]] = []
        model_outputs: list[ModelOutput] = []

        for model_id in models:
            # Execute via OpenClaw (or simulated fallback)
            execution = self.execute_with_openclaw(
                task=task,
                model=model_id,
                agent_name=f"cage-match-{model_id}",
            )
            output = ModelOutput(
                model_id=model_id,
                raw_output=execution["output"],
                latency_ms=execution["duration_ms"],
                token_count=len(execution["output"].split()),
            )
            model_outputs.append(output)
            entries.append(
                {
                    "trace_id": str(uuid.uuid4()),
                    "output": execution["output"],
                    "model_id": model_id,
                    "task": task,
                }
            )
            logger.info("  %s: %d tokens, %d ms", model_id, output.token_count, output.latency_ms)

        ranked_results = judge.evaluate_batch(entries)

        if not self.args.json:
            print(f"\n{'=' * 60}")
            print("  CAGE MATCH LEADERBOARD")
            print(f"  Task: {task[:70]}{'...' if len(task) > 70 else ''}")
            for result in ranked_results:
                rank = result["rank"]
                medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
                print(f"{'=' * 60}")
                print(f"  #{rank} ({medal}) -- {result['model_id']}")
                _print_scores(result["scores"], result["aggregate_score"], verdict=result["verdict"])

        leaderboard = [
            {
                "model_id": r["model_id"],
                "aggregate_score": r["aggregate_score"],
                "verdict": r["verdict"],
                "rank": r["rank"],
            }
            for r in ranked_results
        ]
        notifier.publish_leaderboard(title="Cage Match: Final Rankings", entries=leaderboard)

        # SDK trace upload and real evaluation
        winner = ranked_results[0] if ranked_results else None
        sdk_judge_id = self.create_judge(
            name="Comparative Quality",
            evaluation_goal="Evaluate response quality across task completion, reasoning clarity, conciseness, and instruction following.",
        )
        sdk_results: list[dict[str, Any]] = []
        try:
            for entry in entries:
                trace_id = self.upload_trace(
                    input_text=task,
                    output_text=entry["output"],
                    metadata={"demo": self.demo_id, "model_id": entry["model_id"], "source": "openclaw"},
                )
                if trace_id:
                    logger.info("Trace uploaded for %s: %s", entry["model_id"], trace_id)
                    sdk_result = self.evaluate_trace(trace_id, sdk_judge_id)
                    if sdk_result:
                        sdk_results.append({"model_id": entry["model_id"], **sdk_result})
                        logger.info(
                            "SDK evaluation for %s: score=%.2f passed=%s",
                            entry["model_id"],
                            sdk_result["score"],
                            sdk_result["passed"],
                        )

            if sdk_results and not self.args.json:
                print(f"\n{'=' * 60}")
                print("  SDK EVALUATION RESULTS")
                print(f"{'=' * 60}")
                for sr in sdk_results:
                    status = "PASS" if sr["passed"] else "FAIL"
                    print(f"  {sr['model_id']:<30} score={sr['score']:>5.2f}  [{status}]")
                    if sr.get("reasoning"):
                        print(
                            f"    Reasoning: {sr['reasoning'][:100]}{'...' if len(str(sr.get('reasoning', ''))) > 100 else ''}"
                        )
                print(f"{'=' * 60}\n")
        finally:
            if sdk_judge_id and self.client:
                try:
                    self.client.judges.delete(sdk_judge_id)
                except Exception:
                    pass

        return {
            "run_id": run_id,
            "task": task,
            "models": models,
            "ranked_results": ranked_results,
            "leaderboard": leaderboard,
            "winner": winner["model_id"] if winner else None,
            "sdk_results": sdk_results,
        }


def main() -> None:
    """CLI entrypoint for the Cage Match demo."""
    CageMatchRunner().execute()


if __name__ == "__main__":
    main()
