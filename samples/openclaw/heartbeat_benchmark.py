"""
Heartbeat Benchmark -- OpenClaw Continuous Agent Quality Monitoring
=====================================================================

Has my OpenClaw agent's performance degraded after a model update?

Runs a versioned task battery against multiple OpenClaw agents backed
by different LLM models, scores each output against golden answers,
detects performance drift, and sends alerts when regressions are found.
Uses the LayerLens SDK for trace upload and real evaluation alongside
local judge scoring.

Usage::

    python -m samples.openclaw.heartbeat_benchmark \\
        --models claude-sonnet-4-20250514,gpt-4o,gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import uuid
from typing import Any

from ._runner import DemoRunner, _print_scores
from .judges.benchmark import BenchmarkJudge
from .lib.drift_detector import DriftDetector
from .lib.notifier import Notifier
from .lib.task_battery import BenchmarkTaskBattery

logger = logging.getLogger(__name__)

DEFAULT_MODELS = "claude-sonnet-4-20250514,gpt-4o,gemini-2.0-flash"
DEFAULT_ALERT_THRESHOLD = 7.0


def _simulate_model_output(model_id: str, prompt: str, golden_answer: str) -> str:
    """Generate a deterministic simulated model output for benchmark scoring."""
    seed_str = f"{model_id}:{prompt}:{golden_answer}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    model_quality = {"claude-sonnet-4-20250514": 0.85, "gpt-4o": 0.80, "gpt-4o-mini": 0.70,
                     "gemini-2.0-flash": 0.75, "gemini-3.1-pro": 0.82, "llama-3.3-70b": 0.68, "deepseek-v3": 0.72}
    quality = model_quality.get(model_id, 0.70)
    golden_words = golden_answer.split()
    if not golden_words:
        return f"The answer to '{prompt[:50]}' is not available."
    output_words = []
    for word in golden_words:
        if rng.random() < quality:
            output_words.append(word)
        elif rng.random() < 0.3:
            output_words.append(rng.choice(["approximately", "roughly", "about", "nearly", "around"]))
    prefix = rng.choice(["Based on my analysis, ", "The answer is: ", "", "To answer your question: "])
    result = prefix + " ".join(output_words)
    if rng.random() < 0.3:
        result += " This is a well-established result in the field."
    return result


class HeartbeatBenchmarkRunner(DemoRunner):
    """CLI-driven orchestrator for the OpenClaw Heartbeat Benchmark demo."""

    demo_id = "heartbeat-benchmark"
    demo_name = "Heartbeat Benchmark"
    description = (
        "OpenClaw continuous agent quality monitoring: run a versioned task "
        "battery against multiple OpenClaw agents, score against golden answers, "
        "detect drift, and alert on regressions."
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = super().build_parser()
        parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model IDs.")
        parser.add_argument("--task-battery", default="", help="Path to task battery JSON file.")
        parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD, help="Alert threshold (default: 7.0).")
        parser.add_argument("--drift-window", type=int, default=20, help="Rolling window size for drift detection.")
        parser.add_argument("--drift-sigma", type=float, default=2.0, help="Sigma threshold for drift alerts.")
        parser.add_argument("--notify", default="stdout://", help="Notification channel URI.")
        return parser

    async def run(self) -> dict[str, Any]:
        models = [m.strip() for m in self.args.models.split(",") if m.strip()]
        if not models:
            return {"error": "No models specified"}

        if self.args.task_battery:
            battery = BenchmarkTaskBattery.load_file(self.args.task_battery)
        else:
            battery = BenchmarkTaskBattery.load_default()

        judge = BenchmarkJudge()
        detector = DriftDetector(window_size=self.args.drift_window, sigma_threshold=self.args.drift_sigma)
        notifier = Notifier(channels=[self.args.notify])
        alert_threshold = self.args.alert_threshold

        if not self.args.json:
            print(f"\n{'=' * 60}")
            print("  HEARTBEAT BENCHMARK REPORT")
            print(f"{'=' * 60}")
            print(f"  Battery: {battery.battery_id} ({battery.task_count} tasks, {len(battery.categories)} categories)")
            print(f"  Models: {', '.join(models)}")
            print(f"  Alert Threshold: {alert_threshold:.1f} / 10.0")
            print(f"{'-' * 60}")

        model_scorecards: dict[str, dict[str, Any]] = {}
        all_drift_alerts: list[dict[str, Any]] = []

        for model_id in models:
            scorecard, drift_alerts = self._benchmark_model(model_id, battery, judge, detector, alert_threshold)
            model_scorecards[model_id] = scorecard
            all_drift_alerts.extend(drift_alerts)

        leaderboard = sorted(
            [{"model_id": mid, "aggregate_score": sc["weighted_aggregate"],
              "pass_rate": sc["pass_rate"],
              "verdict": "PASS" if sc["weighted_aggregate"] >= alert_threshold else "FAIL"}
             for mid, sc in model_scorecards.items()],
            key=lambda x: x["aggregate_score"], reverse=True)

        if not self.args.json:
            notifier.publish_leaderboard(title="Heartbeat Benchmark: Rankings", entries=leaderboard)
            if all_drift_alerts:
                print(f"\n  --- Drift Alerts ({len(all_drift_alerts)}) ---")
                for alert in all_drift_alerts:
                    sev_icon = {"critical": "[XX]", "warning": "[!!]", "info": "[--]"}.get(alert["severity"], "[??]")
                    print(f"    {sev_icon} {alert['model_id']}/{alert['task_id']}: {alert['drift_type']} ({alert['message'][:60]})")
            else:
                print("\n  No drift alerts detected.")
            print(f"{'=' * 60}\n")

        # SDK trace upload and real evaluation
        run_id = str(uuid.uuid4())
        sdk_judge_id = self.create_judge(
            name="Heartbeat Benchmark",
            evaluation_goal="Evaluate model output quality against golden answers for accuracy, completeness, and consistency.",
        )
        sdk_results: dict[str, dict[str, Any] | None] = {}
        for model_id in models:
            scorecard = model_scorecards[model_id]
            trace_id = self.upload_trace(
                input_text=f"Heartbeat benchmark: {battery.battery_id} -- {model_id}",
                output_text=f"Weighted aggregate: {scorecard['weighted_aggregate']}, pass rate: {scorecard['pass_rate']}%",
                metadata={"demo": self.demo_id, "battery_id": battery.battery_id,
                           "model_id": model_id, "source": "openclaw"})
            if trace_id:
                logger.info("Trace uploaded for %s: %s", model_id, trace_id)
                sdk_result = self.evaluate_trace(trace_id, sdk_judge_id)
                sdk_results[model_id] = sdk_result
                if sdk_result:
                    logger.info("SDK evaluation for %s: score=%.2f passed=%s",
                                model_id, sdk_result["score"], sdk_result["passed"])

        if sdk_results and not self.args.json:
            has_any = any(v is not None for v in sdk_results.values())
            if has_any:
                print(f"\n  --- SDK Evaluation Scores ---")
                for mid, sr in sdk_results.items():
                    if sr:
                        status = "PASS" if sr["passed"] else "FAIL"
                        print(f"    {mid:<30} score={sr['score']:>5.2f}  [{status}]")

        return {"run_id": run_id, "battery_id": battery.battery_id, "models": models,
                "alert_threshold": alert_threshold, "model_scorecards": model_scorecards,
                "leaderboard": leaderboard, "drift_alerts": all_drift_alerts,
                "battery_summary": battery.summary(), "sdk_results": sdk_results}

    def _benchmark_model(self, model_id: str, battery: BenchmarkTaskBattery,
                          judge: BenchmarkJudge, detector: DriftDetector,
                          alert_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not self.args.json:
            print(f"\n  --- {model_id} ---")

        task_results: list[dict[str, Any]] = []
        drift_alerts: list[dict[str, Any]] = []

        for task in battery.tasks:
            # Use OpenClaw execution or simulated fallback
            execution = self.execute_with_openclaw(
                task=task.prompt,
                model=model_id,
                agent_name=f"heartbeat-{model_id}",
            )
            output = execution["output"]
            latency_ms = execution["duration_ms"]

            # If output is simulated (starts with [Simulated), use the deterministic generator
            if output.startswith("[Simulated"):
                output = _simulate_model_output(model_id, task.prompt, task.golden_answer)

            result = judge.evaluate(
                trace_id=str(uuid.uuid4()), output=output,
                context={"golden_answer": task.golden_answer, "scoring_method": task.scoring_method,
                         "weight": task.weight, "task_id": task.task_id, "model_id": model_id})

            alerts = detector.record_and_check(model_id=model_id, task_id=task.task_id,
                                                score=result["aggregate_score"], latency_ms=latency_ms)
            for alert in alerts:
                drift_alerts.append(alert.model_dump())

            task_results.append({
                "task_id": task.task_id, "category": task.category, "difficulty": task.difficulty,
                "scoring_method": task.scoring_method, "weight": task.weight,
                "score": result["aggregate_score"], "verdict": result["verdict"],
                "rationale": result["rationale"], "latency_ms": latency_ms})

            if not self.args.json:
                icon = "[OK]" if result["verdict"] == "PASS" else "[XX]"
                print(f"    {icon} {task.task_id:<16} {task.scoring_method:<22} "
                      f"{result['aggregate_score']:>5.2f}  {result['verdict']:<5}  w={task.weight}")

        total_weight = sum(r["weight"] for r in task_results)
        weighted_aggregate = sum(r["score"] * r["weight"] for r in task_results) / total_weight if total_weight > 0 else 0.0
        pass_count = sum(1 for r in task_results if r["verdict"] == "PASS")
        pass_rate = (pass_count / len(task_results) * 100) if task_results else 0.0

        category_scores: dict[str, list[float]] = {}
        for r in task_results:
            category_scores.setdefault(r["category"], []).append(r["score"])
        category_averages = {cat: round(sum(s) / len(s), 2) for cat, s in category_scores.items()}

        if not self.args.json:
            print(f"    Weighted Aggregate: {weighted_aggregate:.2f} / 10.0")
            print(f"    Pass Rate: {pass_rate:.1f}%")
            _print_scores(category_averages, weighted_aggregate,
                         verdict="PASS" if weighted_aggregate >= alert_threshold else "FAIL")

        return {
            "model_id": model_id, "weighted_aggregate": round(weighted_aggregate, 2),
            "pass_count": pass_count, "fail_count": len(task_results) - pass_count,
            "pass_rate": round(pass_rate, 1), "task_results": task_results,
            "category_averages": category_averages, "method_stats": judge.get_method_stats(),
        }, drift_alerts


def main() -> None:
    """CLI entrypoint for the Heartbeat Benchmark demo."""
    HeartbeatBenchmarkRunner().execute()


if __name__ == "__main__":
    main()
