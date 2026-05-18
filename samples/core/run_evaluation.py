#!/usr/bin/env python
"""
Evaluation Lifecycle -- LayerLens Python SDK Sample
===================================================

Demonstrates the full evaluation lifecycle using the SDK:

  1. Fetch available models and benchmarks.
  2. Create an evaluation run (model + benchmark).
  3. Poll for completion with configurable timeout.
  4. Fetch and display results with pagination.

This sample ports the ateam core/run_evaluation.py sample to use the
layerlens SDK client instead of raw httpx calls.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one model and benchmark configured in the project

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python run_evaluation.py
    python run_evaluation.py --model-key gpt-4o --benchmark-key mmlu
    python run_evaluation.py --timeout 600
"""

from __future__ import annotations

import sys
import logging
import argparse

from layerlens import Stratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.run_evaluation")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an evaluation with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--model-key",
        default="",
        help="Model key to evaluate (e.g., 'gpt-4o'). If omitted, uses the first available model.",
    )
    parser.add_argument(
        "--benchmark-key",
        default="",
        help="Benchmark key to use (e.g., 'mmlu'). If omitted, uses the first available benchmark.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for evaluation completion (default: 600).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between status polls (default: 15).",
    )
    parser.add_argument(
        "--results-page-size",
        type=int,
        default=20,
        help="Number of results per page to display (default: 20).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Initialize SDK client ---
    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Step 1: Fetch models and benchmarks ---
    logger.info("=" * 60)
    logger.info("Step 1: Fetch models and benchmarks")
    logger.info("=" * 60)

    models = client.models.get()
    if not models:
        logger.error("No models available in the project. Add models first.")
        sys.exit(1)
    logger.info("Found %d model(s)", len(models))
    for m in models[:5]:
        logger.info("  - %s (key=%s, id=%s)", m.name, m.key, m.id)

    benchmarks = client.benchmarks.get()
    if not benchmarks:
        logger.error("No benchmarks available in the project. Add benchmarks first.")
        sys.exit(1)
    logger.info("Found %d benchmark(s)", len(benchmarks))
    for b in benchmarks[:5]:
        logger.info("  - %s (key=%s, id=%s)", b.name, b.key, b.id)

    # Select model
    if args.model_key:
        model = client.models.get_by_key(args.model_key)
        if not model:
            logger.error("Model with key '%s' not found", args.model_key)
            sys.exit(1)
    else:
        model = models[0]
    logger.info("Using model: %s (%s)", model.name, model.key)

    # Select benchmark
    if args.benchmark_key:
        benchmark = client.benchmarks.get_by_key(args.benchmark_key)
        if not benchmark:
            logger.error("Benchmark with key '%s' not found", args.benchmark_key)
            sys.exit(1)
    else:
        benchmark = benchmarks[0]
    logger.info("Using benchmark: %s (%s)", benchmark.name, benchmark.key)

    # --- Step 2: Create evaluation ---
    logger.info("=" * 60)
    logger.info("Step 2: Create evaluation")
    logger.info("=" * 60)

    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    if not evaluation:
        logger.error("Failed to create evaluation")
        sys.exit(1)

    logger.info("Evaluation created: %s (status=%s)", evaluation.id, evaluation.status)

    # --- Step 3: Wait for completion ---
    logger.info("=" * 60)
    logger.info("Step 3: Waiting for completion (timeout=%ds)...", args.timeout)
    logger.info("=" * 60)

    try:
        evaluation = client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=args.poll_interval,
            timeout_seconds=args.timeout,
        )
    except TimeoutError as exc:
        logger.error("Evaluation timed out: %s", exc)
        logger.info("Check status manually using evaluation ID: %s", evaluation.id)
        sys.exit(2)

    if not evaluation:
        logger.error("Evaluation disappeared during polling")
        sys.exit(1)

    logger.info("Evaluation %s finished: status=%s", evaluation.id, evaluation.status)

    # --- Step 4: Fetch results ---
    logger.info("=" * 60)
    logger.info("Step 4: Fetch results")
    logger.info("=" * 60)

    if evaluation.is_success:
        results_resp = client.results.get(
            evaluation=evaluation,
            page=1,
            page_size=args.results_page_size,
        )

        if results_resp and results_resp.results:
            logger.info("Results: %d items (page 1)", len(results_resp.results))

            # Display summary
            print("\n" + "=" * 70)
            print(f"  Evaluation: {evaluation.id}")
            print(f"  Model:      {model.name} ({model.key})")
            print(f"  Benchmark:  {benchmark.name} ({benchmark.key})")
            print(f"  Status:     {evaluation.status}")
            if hasattr(evaluation, "accuracy") and evaluation.accuracy is not None:
                print(f"  Accuracy:   {evaluation.accuracy:.2%}")
            print("=" * 70)

            # Display individual results
            for i, result in enumerate(results_resp.results[:10], 1):
                score = getattr(result, "score", None)
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                prompt_preview = str(getattr(result, "prompt", ""))[:60]
                print(f"  [{i:2d}] Score: {score_str}  Prompt: {prompt_preview}...")
            print()
        else:
            logger.warning("No results available for this evaluation")
    else:
        logger.warning("Evaluation did not succeed (status=%s), no results to show.", evaluation.status)

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
