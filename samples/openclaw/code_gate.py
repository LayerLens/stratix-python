"""
Code Gate -- OpenClaw Code Generation with Quality Gate Enforcement
=====================================================================

Is the code my OpenClaw agent produces safe to execute?

Has an OpenClaw agent generate code via a Coder -> Reviewer -> Tester ->
Judge pipeline, iteratively refines it based on Judge feedback, and
enforces a binary quality gate (PASS or FAIL) before the code is
considered safe to run.  Uses the LayerLens SDK for trace upload and
real evaluation alongside local judge scoring.

Usage::

    python -m samples.openclaw.code_gate \\
        --task "Implement a function to merge two sorted lists" \\
        --threshold 7.5
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import Any

from ._runner import DemoRunner, _print_scores
from .judges.code_quality import CodeQualityJudge
from .lib.code_pipeline import CodePipeline

logger = logging.getLogger(__name__)

DEFAULT_TASK = (
    "Implement a Python function called 'merge_sorted' that takes two sorted "
    "lists of integers and returns a single sorted list containing all elements "
    "from both inputs. Handle edge cases: empty lists, duplicate values, and "
    "lists of different lengths."
)
DEFAULT_THRESHOLD = 7.5
DEFAULT_MAX_ITERATIONS = 3


class CodeGateRunner(DemoRunner):
    """CLI-driven orchestrator for the OpenClaw Code Gate demo."""

    demo_id = "code-gate"
    demo_name = "Code Gate"
    description = (
        "OpenClaw code generation with quality gate enforcement: have an "
        "OpenClaw agent run a Coder -> Reviewer -> Tester -> Judge pipeline "
        "and enforce a binary PASS/FAIL gate on generated code."
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = super().build_parser()
        parser.add_argument("--task", default=DEFAULT_TASK, help="Task specification for code generation.")
        parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Gate threshold (default: 7.5).")
        parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="Max pipeline iterations (default: 3).")
        return parser

    async def run(self) -> dict[str, Any]:
        task = self.args.task
        threshold = self.args.threshold
        max_iterations = self.args.max_iterations

        if threshold < 0.0 or threshold > 10.0:
            logger.error("Threshold must be between 0.0 and 10.0")
            sys.exit(1)

        # Execute the task via OpenClaw to get the initial code
        openclaw_result = self.execute_with_openclaw(
            task=f"Generate Python code: {task}",
            agent_name="openclaw-coder",
        )
        logger.info("OpenClaw code generation: %d ms", openclaw_result["duration_ms"])

        judge = CodeQualityJudge(judge_id="judge_code_gate", gate_threshold=threshold)
        pipeline = CodePipeline(judge=judge, max_iterations=max_iterations)

        logger.info("Code Gate: task=%s", task[:80])
        pipeline_result = pipeline.execute(task)

        if not self.args.json:
            self._print_gate_decision(pipeline_result)

        # SDK trace upload and real evaluation
        run_id = str(uuid.uuid4())
        final_eval = pipeline_result.get("final_evaluation", {})
        trace_id = self.upload_trace(
            input_text=task,
            output_text=final_eval.get("rationale", ""),
            metadata={"demo": self.demo_id, "verdict": pipeline_result["final_verdict"],
                       "source": "openclaw"})
        if trace_id:
            logger.info("Trace uploaded: %s", trace_id)

        sdk_result = None
        sdk_judge_id = self.create_judge(
            name="Code Quality Gate",
            evaluation_goal="Evaluate generated code for correctness, clarity, security, test coverage, and spec adherence.",
        )
        if trace_id and sdk_judge_id:
            sdk_result = self.evaluate_trace(trace_id, sdk_judge_id)
            if sdk_result:
                logger.info("SDK evaluation: score=%.2f passed=%s", sdk_result["score"], sdk_result["passed"])

        if sdk_result and not self.args.json:
            sdk_status = "PASS" if sdk_result["passed"] else "FAIL"
            print(f"\n{'=' * 60}")
            print("  SDK EVALUATION RESULT")
            print(f"{'=' * 60}")
            print(f"  SDK Verdict: {sdk_status}  (score={sdk_result['score']:.2f})")
            if sdk_result.get("reasoning"):
                print(f"  Reasoning: {sdk_result['reasoning'][:200]}")
            print(f"{'=' * 60}\n")

        return {
            "run_id": run_id, "task": task, "gate_threshold": threshold,
            "max_iterations": max_iterations,
            "total_iterations": pipeline_result["total_iterations"],
            "final_verdict": pipeline_result["final_verdict"],
            "final_score": pipeline_result["final_score"],
            "passed": pipeline_result["passed"],
            "iterations": pipeline_result["iterations"],
            "final_evaluation": final_eval,
            "sdk_result": sdk_result,
        }

    def _print_gate_decision(self, result: dict[str, Any]) -> None:
        task = result["task"]
        threshold = result["gate_threshold"]
        verdict = result["final_verdict"]
        score = result["final_score"]
        total = result["total_iterations"]
        iterations = result["iterations"]

        print(f"\n{'=' * 60}")
        print("  CODE GATE DECISION")
        print(f"{'=' * 60}")
        print(f"  Task: {task[:70]}{'...' if len(task) > 70 else ''}")
        print(f"  Gate Threshold: {threshold:.1f} / 10.0")
        print(f"  Final Verdict: {verdict}")
        print(f"  Iterations: {total} / {self.args.max_iterations}")
        print(f"  Final Score: {score:.1f} / 10.0")
        print(f"{'-' * 60}")
        for it in iterations:
            print(f"  Iteration {it['iteration']}: {it['verdict']:<5} ({it['aggregate_score']:.1f})")
        print(f"{'=' * 60}")

        final_eval = result.get("final_evaluation", {})
        if final_eval:
            _print_scores(final_eval.get("scores", {}), final_eval.get("aggregate_score", 0.0), verdict=verdict)
            suggestions = final_eval.get("suggestions", [])
            if suggestions:
                print("  Improvement suggestions:")
                for s in suggestions:
                    print(f"    - {s}")
                print()


def main() -> None:
    """CLI entrypoint for the Code Gate demo."""
    CodeGateRunner().execute()


if __name__ == "__main__":
    main()
