"""Run a full evaluation lifecycle on the LayerLens Stratix platform.

Demonstrates:
- Looking up models and benchmarks by name
- Creating an evaluation run (model vs. benchmark)
- Polling with exponential backoff until completion
- Displaying results in a formatted table

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def find_model(client, name: str):
    """Find a model by name substring. Returns the first match."""
    models = client.models.get(name=name)
    if not models:
        print(f"ERROR: No model matching '{name}' found.", file=sys.stderr)
        sys.exit(1)
    model = models[0]
    print(f"[model]     {model.name}  (id={model.id})")
    return model


def find_benchmark(client, name: str):
    """Find a benchmark by name substring. Returns the first match."""
    benchmarks = client.benchmarks.get(name=name)
    if not benchmarks:
        print(f"ERROR: No benchmark matching '{name}' found.", file=sys.stderr)
        sys.exit(1)
    bm = benchmarks[0]
    print(f"[benchmark] {bm.name}  (id={bm.id})")
    return bm


def poll_evaluation(client, evaluation, timeout_sec: int, initial_interval: int):
    """Poll an evaluation with exponential backoff."""
    start = time.time()
    interval = initial_interval

    while not evaluation.is_finished:
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            print(f"[timeout]  Evaluation did not finish within {timeout_sec}s.", file=sys.stderr)
            return evaluation

        print(f"[poll]     status={evaluation.status.value}  elapsed={elapsed:.0f}s  next_check={interval}s")
        time.sleep(interval)
        interval = min(interval * 2, 120)

        updated = client.evaluations.get(evaluation)
        if updated:
            evaluation = updated

    return evaluation


def display_results(evaluation):
    """Print evaluation results as a formatted table."""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  Evaluation Results: {evaluation.id}")
    print(sep)
    print(f"  Status      : {evaluation.status.value}")
    print(f"  Model       : {evaluation.model_name} ({evaluation.model_key})")
    print(f"  Benchmark   : {evaluation.benchmark_name}")
    print(f"  Accuracy    : {evaluation.accuracy:.2%}")
    if evaluation.readability_score:
        print(f"  Readability : {evaluation.readability_score:.2%}")
    if evaluation.toxicity_score:
        print(f"  Toxicity    : {evaluation.toxicity_score:.2%}")
    if evaluation.average_duration:
        print(f"  Avg Duration: {evaluation.average_duration}ms")
    if evaluation.failed_prompt_count:
        print(f"  Failed      : {evaluation.failed_prompt_count} prompts")
    print(sep)

    # Try to fetch per-prompt results
    try:
        results_resp = evaluation.get_results(page=1, page_size=10)
        if results_resp and results_resp.results:
            print(f"\n  Top {len(results_resp.results)} prompt results:")
            print(f"  {'Prompt':<30} {'Score':>8} {'Duration':>10}")
            print(f"  {'------':<30} {'-----':>8} {'--------':>10}")
            for r in results_resp.results:
                prompt_short = (r.prompt[:27] + "...") if len(r.prompt) > 30 else r.prompt
                print(f"  {prompt_short:<30} {r.score:>8.2%} {r.duration.total_seconds():>9.1f}s")
            print()
    except Exception as exc:
        print(f"  (Could not fetch per-prompt results: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an evaluation on the LayerLens Stratix platform."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or substring to evaluate (e.g. 'gpt-4o').",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name or substring to run against (e.g. 'MMLU').",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Max seconds to wait for completion (default: 1800).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Initial polling interval in seconds (default: 15).",
    )
    args = parser.parse_args()

    from layerlens import Stratix

    api_key = _require_env("LAYERLENS_STRATIX_API_KEY")
    client = Stratix(api_key=api_key)
    print(f"[init]     Connected to LayerLens (org={client.organization_id})")

    # Resolve model and benchmark
    model = find_model(client, args.model)
    benchmark = find_benchmark(client, args.benchmark)

    # Create evaluation
    print("[create]   Submitting evaluation...")
    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    if evaluation is None:
        print("ERROR: Failed to create evaluation.", file=sys.stderr)
        sys.exit(1)
    print(f"[create]   Evaluation created: id={evaluation.id}  status={evaluation.status.value}")

    # Poll until complete
    evaluation = poll_evaluation(client, evaluation, args.timeout, args.poll_interval)

    # Display results
    display_results(evaluation)


if __name__ == "__main__":
    main()
