#!/usr/bin/env python
"""
Async Workflow -- LayerLens Python SDK Sample
=============================================

Demonstrates using the AsyncStratix client for concurrent operations:

  1. Initialize the async client.
  2. Concurrently fetch models and benchmarks.
  3. Create an evaluation asynchronously.
  4. Asynchronously wait for completion.
  5. Fetch results.

This sample demonstrates the async capabilities of the SDK, porting
concepts from the existing async_client.py and async_run_evaluations.py
examples.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python async_workflow.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from layerlens import AsyncStratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.async_workflow")


async def main() -> None:
    try:
        client = AsyncStratix()
    except Exception as exc:
        logger.error("Failed to initialize async client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Step 1: Concurrent fetch ---
    logger.info("=" * 60)
    logger.info("Step 1: Concurrently fetch models and benchmarks")
    logger.info("=" * 60)

    models_task = asyncio.create_task(client.models.get())
    benchmarks_task = asyncio.create_task(client.benchmarks.get())

    models, benchmarks = await asyncio.gather(models_task, benchmarks_task)

    if not models:
        logger.error("No models available")
        sys.exit(1)
    if not benchmarks:
        logger.error("No benchmarks available")
        sys.exit(1)

    logger.info("Models: %d  |  Benchmarks: %d", len(models), len(benchmarks))

    # --- Step 2: Create evaluation ---
    logger.info("=" * 60)
    logger.info("Step 2: Create evaluation")
    logger.info("=" * 60)

    model = models[0]
    benchmark = benchmarks[0]
    logger.info("Model: %s  |  Benchmark: %s", model.name, benchmark.name)

    evaluation = await client.evaluations.create(model=model, benchmark=benchmark)
    if not evaluation:
        logger.error("Failed to create evaluation")
        sys.exit(1)

    logger.info("Evaluation created: %s (status=%s)", evaluation.id, evaluation.status)

    # --- Step 3: Wait for completion ---
    logger.info("=" * 60)
    logger.info("Step 3: Async wait for completion")
    logger.info("=" * 60)

    try:
        evaluation = await client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=10,
            timeout_seconds=300,
        )
    except TimeoutError as exc:
        logger.error("Timed out: %s", exc)
        sys.exit(2)

    if not evaluation:
        logger.error("Evaluation disappeared during polling")
        sys.exit(1)

    logger.info("Evaluation completed: status=%s", evaluation.status)

    # --- Step 4: Fetch results ---
    logger.info("=" * 60)
    logger.info("Step 4: Fetch results")
    logger.info("=" * 60)

    if evaluation.is_success:
        results = await client.results.get(evaluation=evaluation, page_size=10)
        if results and results.results:
            logger.info("Got %d result(s)", len(results.results))
            for i, r in enumerate(results.results[:5], 1):
                score = getattr(r, "score", None)
                logger.info("  [%d] Score: %s", i, f"{score:.2f}" if score else "N/A")
        else:
            logger.info("No results yet")

        # Fetch all results (across all pages)
        all_results = await client.results.get_all(evaluation=evaluation)
        logger.info("Total results across all pages: %d", len(all_results))
    else:
        logger.warning("Evaluation did not succeed: %s", evaluation.status)

    # --- Additional: Instance-method alternatives ---
    # The evaluation object itself has async convenience methods that mirror
    # the client-level calls above.  These are an alternative approach.
    logger.info("=" * 60)
    logger.info("Step 5: Instance-method async alternatives")
    logger.info("=" * 60)

    try:
        # wait_for_completion_async() on the evaluation instance
        evaluation2 = await client.evaluations.create(model=model, benchmark=benchmark)
        if evaluation2:
            logger.info("Created second evaluation: %s", evaluation2.id)
            evaluation2 = await evaluation2.wait_for_completion_async()
            logger.info("Instance wait complete: status=%s", evaluation2.status)

            # get_results_async() on the evaluation instance
            if evaluation2.is_success:
                results = await evaluation2.get_results_async()
                if results and results.results:
                    logger.info("Instance get_results: %d result(s)", len(results.results))
                else:
                    logger.info("Instance get_results: no results")
    except AttributeError:
        logger.info("Instance-level async methods not available on this SDK version")
    except Exception as exc:
        logger.info("Instance async methods failed: %s", exc)

    # --- Cleanup ---
    await client.aclose()
    logger.info("Sample complete.")


if __name__ == "__main__":
    asyncio.run(main())
