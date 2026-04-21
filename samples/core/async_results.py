"""
Async Patterns -- LayerLens Python SDK Sample
=============================================

Demonstrates async SDK usage with ``AsyncStratix``:

  1. **Concurrent result fetching** -- fetch results for multiple
     evaluations in parallel using ``asyncio.gather``.
  2. **Concurrent evaluation creation** -- create and run multiple
     evaluations in parallel with progress tracking.
  3. **Judge + trace combined workflow** -- create a judge, upload
     traces, run trace evaluations concurrently, and collect results.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one model and benchmark configured in the project

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python async_results.py
"""

from __future__ import annotations

import os
import sys
import time
import asyncio

from layerlens import Stratix, AsyncStratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge

# ---------------------------------------------------------------------------
# 1. Concurrent result fetching
# ---------------------------------------------------------------------------


async def fetch_evaluation_results(client: AsyncStratix, evaluation_id: str) -> tuple[str, list | None]:
    """Fetch results for a single evaluation."""
    try:
        print(f"  Fetching evaluation {evaluation_id}...")
        evaluation = await client.evaluations.get_by_id(evaluation_id)
        if not evaluation:
            print(f"  Evaluation {evaluation_id} not found")
            return evaluation_id, None

        print(f"  Found evaluation {evaluation.id}, status={evaluation.status}")
        results = await client.results.get_all(evaluation=evaluation)
        print(f"  Loaded {len(results)} results for {evaluation_id}")
        return evaluation_id, results
    except Exception as e:
        print(f"  Error fetching evaluation {evaluation_id}: {e}")
        return evaluation_id, None


async def demo_concurrent_fetch(client: AsyncStratix) -> None:
    """Fetch results from multiple evaluations concurrently."""
    # Get some existing evaluations to work with
    response = await client.evaluations.get_many(page_size=3)
    if not response or not response.evaluations:
        print("No evaluations found, skipping concurrent fetch demo.")
        return

    evaluation_ids = [e.id for e in response.evaluations]
    print(f"Fetching results for {len(evaluation_ids)} evaluations concurrently...")

    tasks = [fetch_evaluation_results(client, eid) for eid in evaluation_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if not isinstance(r, Exception) and r[1] is not None)
    print(f"Successfully fetched results for {successful}/{len(evaluation_ids)} evaluations")


# ---------------------------------------------------------------------------
# 2. Concurrent evaluation creation
# ---------------------------------------------------------------------------


async def create_and_run_evaluation(
    client: AsyncStratix, model, benchmark, eval_number: int
) -> tuple[int, str | None, int, bool]:
    """Create, run, and collect results for a single evaluation."""
    try:
        print(f"  Starting evaluation #{eval_number}...")
        evaluation = await client.evaluations.create(model=model, benchmark=benchmark)
        print(f"  Created evaluation #{eval_number}: {evaluation.id}")

        evaluation = await client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=10,
            timeout_seconds=600,
        )
        print(f"  Evaluation #{eval_number} ({evaluation.id}) finished: status={evaluation.status}")

        if evaluation.is_success:
            results = await client.results.get_all(evaluation=evaluation)
            print(f"  Evaluation #{eval_number} completed with {len(results)} results")
            return eval_number, evaluation.id, len(results), True
        else:
            print(f"  Evaluation #{eval_number} did not succeed")
            return eval_number, evaluation.id, 0, False
    except Exception as e:
        print(f"  Error in evaluation #{eval_number}: {e}")
        return eval_number, None, 0, False


async def demo_concurrent_evaluations(client: AsyncStratix) -> None:
    """Create and run multiple evaluations in parallel."""
    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    if not models or not benchmarks:
        print("No models or benchmarks available, skipping concurrent evaluation demo.")
        return

    target_model = models[0]
    target_benchmark = benchmarks[0]
    num_evaluations = 3

    print(
        f"Running {num_evaluations} evaluations in parallel "
        f"(model={target_model.name}, benchmark={target_benchmark.name})..."
    )

    tasks = [create_and_run_evaluation(client, target_model, target_benchmark, i + 1) for i in range(num_evaluations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    successful = 0
    total_results = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"  Exception: {r}")
        else:
            eval_num, eval_id, result_count, success = r
            if success:
                successful += 1
                total_results += result_count
                print(f"  Evaluation #{eval_num} ({eval_id}): SUCCESS - {result_count} results")
            else:
                print(f"  Evaluation #{eval_num} ({eval_id}): FAILED")

    print(f"Overall: {successful}/{num_evaluations} evaluations succeeded, {total_results} total results")


# ---------------------------------------------------------------------------
# 3. Judge + trace combined async workflow
# ---------------------------------------------------------------------------


async def demo_judge_and_traces(client: AsyncStratix) -> None:
    """Create a judge, upload traces, evaluate concurrently, fetch results."""
    # Use sync client + create_judge helper (resolves model automatically)
    sync_client = Stratix()
    judge = await asyncio.to_thread(
        create_judge,
        sync_client,
        name=f"Async Demo Judge {int(time.time())}",
        evaluation_goal="Evaluate whether the response is accurate, helpful, and well-structured",
    )
    print(f"Created judge {judge.id}: {judge.name}")

    try:
        # Upload traces
        traces_file = os.path.join(os.path.dirname(__file__), "..", "data", "traces", "example_traces.jsonl")
        if not os.path.exists(traces_file):
            print(f"Trace file not found at {traces_file}, skipping trace upload.")
            return

        upload_result = await client.traces.upload(traces_file)
        print(f"Uploaded {len(upload_result.trace_ids)} traces")

        # List traces and pick a subset
        traces_response = await client.traces.get_many(page_size=10)
        trace_ids = [t.id for t in traces_response.traces[:5]]
        print(f"Using {len(trace_ids)} traces for evaluation")

        # Estimate cost
        estimate = await client.trace_evaluations.estimate_cost(
            trace_ids=trace_ids,
            judge_id=judge.id,
        )
        if estimate and estimate.estimated_cost is not None:
            print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
        else:
            print("Estimated cost: unavailable")

        # Run evaluations concurrently
        tasks = [client.trace_evaluations.create(trace_id=tid, judge_id=judge.id) for tid in trace_ids]
        evaluations = await asyncio.gather(*tasks)

        for evaluation in evaluations:
            if evaluation:
                print(f"  Trace evaluation {evaluation.id}: {evaluation.status}")

        # Poll for results with exponential backoff
        print("Polling for evaluation results...")
        sync_client_for_poll = Stratix()
        for evaluation in evaluations:
            if not evaluation:
                continue
            delay = 2.0
            found = False
            for _ in range(30):
                await asyncio.sleep(delay)
                try:
                    resp = await asyncio.to_thread(sync_client_for_poll.trace_evaluations.get_results, evaluation.id)
                    if resp and resp.score is not None:
                        print(f"  Score: {resp.score}, Passed: {resp.passed}")
                        found = True
                        break
                except Exception:
                    pass
                delay = min(delay * 1.3, 10.0)
            if not found:
                print(f"  Evaluation {evaluation.id}: no results after polling")
    finally:
        # Clean up
        await client.judges.delete(judge.id)
        print(f"Cleaned up judge {judge.id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    client = AsyncStratix()

    print("=" * 60)
    print("1. CONCURRENT RESULT FETCHING")
    print("=" * 60)
    await demo_concurrent_fetch(client)

    print("\n" + "=" * 60)
    print("2. CONCURRENT EVALUATION CREATION")
    print("=" * 60)
    await demo_concurrent_evaluations(client)

    print("\n" + "=" * 60)
    print("3. JUDGE + TRACE COMBINED WORKFLOW")
    print("=" * 60)
    await demo_judge_and_traces(client)

    print("\nAll async demos complete.")


if __name__ == "__main__":
    asyncio.run(main())
