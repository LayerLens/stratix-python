#!/usr/bin/env python
"""
Benchmark Evaluation -- LayerLens Python SDK Sample
====================================================

Demonstrates the model+benchmark evaluation workflow:

  1. Fetch available models and benchmarks.
  2. Create an evaluation that scores a model against a benchmark.
  3. Poll for completion with configurable timeout.
  4. Retrieve and display paginated results.

This is the standard evaluation path for comparing model performance
on public or custom benchmarks. For trace-level evaluation (scoring
individual LLM interactions with judges), see ``trace_evaluation.py``.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one model and one benchmark must be available in your project
  or the public catalog.

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python benchmark_evaluation.py
    python benchmark_evaluation.py --model gpt-4o --benchmark simpleQA
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from layerlens import Stratix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.benchmark_evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a model+benchmark evaluation via the LayerLens SDK.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name to evaluate (searches public catalog). If omitted, uses the first available.",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="Benchmark name to evaluate against. If omitted, uses the first available.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for evaluation completion (default: 600).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10,
        help="Number of results per page (default: 10).",
    )
    return parser


def _find_model(client: Stratix, name: str) -> Any:
    """Find a model by name, checking project then public catalog."""
    if name:
        models = client.models.get(type="public", name=name)
        if models:
            return models[0]
        models = client.models.get(name=name)
        if models:
            return models[0]
        logger.error("Model '%s' not found.", name)
        sys.exit(1)

    # No name specified -- use first available
    models = client.models.get()
    if models:
        return models[0]
    # Fall back to public catalog
    pub = client.public.models.get()
    if pub and hasattr(pub, "models") and pub.models:
        return pub.models[0]
    logger.error("No models available. Add a model to your project first.")
    sys.exit(1)


def _find_benchmark(client: Stratix, name: str) -> Any:
    """Find a benchmark by name, checking project then public catalog."""
    if name:
        benchmarks = client.benchmarks.get(type="public", name=name)
        if benchmarks:
            return benchmarks[0]
        benchmarks = client.benchmarks.get(name=name)
        if benchmarks:
            return benchmarks[0]
        logger.error("Benchmark '%s' not found.", name)
        sys.exit(1)

    benchmarks = client.benchmarks.get()
    if benchmarks:
        return benchmarks[0]
    pub = client.public.benchmarks.get()
    if pub:
        items = getattr(pub, "datasets", None) or getattr(pub, "benchmarks", None)
        if items:
            return items[0]
    logger.error("No benchmarks available. Add a benchmark to your project first.")
    sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Initialize client ---
    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info("Connected (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Step 1: Find model and benchmark ---
    logger.info("=" * 60)
    logger.info("Step 1: Find model and benchmark")
    logger.info("=" * 60)

    model = _find_model(client, args.model)
    benchmark = _find_benchmark(client, args.benchmark)

    logger.info("  Model:     %s (id=%s)", model.name, model.id)
    logger.info("  Benchmark: %s (id=%s)", benchmark.name, benchmark.id)

    # --- Step 2: Create evaluation ---
    logger.info("=" * 60)
    logger.info("Step 2: Create evaluation")
    logger.info("=" * 60)

    evaluation = client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )
    logger.info("  Evaluation ID: %s", evaluation.id)
    logger.info("  Status:        %s", evaluation.status)

    # --- Step 3: Wait for completion ---
    logger.info("=" * 60)
    logger.info("Step 3: Wait for completion (timeout=%ds)", args.timeout)
    logger.info("=" * 60)

    evaluation = client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=10,
        timeout_seconds=args.timeout,
    )
    logger.info("  Final status: %s", evaluation.status)

    # --- Step 4: Retrieve results ---
    logger.info("=" * 60)
    logger.info("Step 4: Retrieve results")
    logger.info("=" * 60)

    if not evaluation.is_success:
        logger.warning("Evaluation did not succeed (status=%s). No results.", evaluation.status)
        return

    # Page 1
    results_page = client.results.get(
        evaluation=evaluation,
        page=1,
        page_size=args.page_size,
    )
    if results_page and results_page.results:
        total = results_page.metrics.total_count if hasattr(results_page, "metrics") and results_page.metrics else "?"
        logger.info("  Page 1 of results (%s total):", total)
        for r in results_page.results:
            score = getattr(r, "score", "N/A")
            prompt_preview = (r.prompt[:60] + "...") if hasattr(r, "prompt") and r.prompt and len(r.prompt) > 60 else getattr(r, "prompt", "")
            logger.info("    score=%.4f  prompt=%s", score if isinstance(score, (int, float)) else 0, prompt_preview)
    else:
        logger.info("  No results returned.")

    # All results
    all_results = client.results.get_all(evaluation=evaluation)
    logger.info("  Total results (all pages): %d", len(all_results))

    if all_results:
        scores = [r.score for r in all_results if hasattr(r, "score") and isinstance(r.score, (int, float))]
        if scores:
            avg = sum(scores) / len(scores)
            logger.info("  Average score: %.4f", avg)
            logger.info("  Min score:     %.4f", min(scores))
            logger.info("  Max score:     %.4f", max(scores))

    logger.info("Benchmark evaluation complete.")


if __name__ == "__main__":
    main()
