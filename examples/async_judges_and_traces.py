#!/usr/bin/env python3

import os
import time
import asyncio

# Also import sync client just to fetch a model (models resource works the same)
from layerlens import Stratix, AsyncStratix


async def main():
    # Fetch a model to use for judge creation (using sync client for simplicity)
    sync_client = Stratix()
    models = sync_client.models.get(type="public", name="gpt-4o")
    if not models:
        print("No models found, exiting")
        return
    model = models[0]
    print(f"Using model: {model.name} ({model.id})")

    # Construct async client
    client = AsyncStratix()

    # --- Create a judge
    judge = await client.judges.create(
        name=f"Response Quality Judge {int(time.time())}",
        evaluation_goal="Evaluate whether the response is accurate, helpful, and well-structured",
        model_id=model.id,
    )
    print(f"Created judge {judge.id}: {judge.name}")

    # --- Upload traces
    traces_file = os.path.join(os.path.dirname(__file__), "traces.jsonl")
    result = await client.traces.upload(traces_file)
    print(f"Uploaded {len(result.trace_ids)} traces")

    # --- List traces
    traces_response = await client.traces.get_many(page_size=10)
    print(f"Found {traces_response.total_count} traces")

    # --- Run judge on multiple traces concurrently
    trace_ids = [t.id for t in traces_response.traces[:5]]

    # Estimate cost first
    estimate = await client.trace_evaluations.estimate_cost(
        trace_ids=trace_ids,
        judge_id=judge.id,
    )
    print(f"Estimated cost: ${estimate.estimated_cost:.4f}")

    # Run evaluations concurrently
    tasks = [client.trace_evaluations.create(trace_id=tid, judge_id=judge.id) for tid in trace_ids]
    evaluations = await asyncio.gather(*tasks)

    for evaluation in evaluations:
        if evaluation:
            print(f"  Evaluation {evaluation.id}: {evaluation.status}")

    # --- Wait for evaluations to finish, then fetch results
    print("Waiting for evaluations to complete...")
    await asyncio.sleep(10)

    for evaluation in evaluations:
        if not evaluation:
            continue
        try:
            result = await client.trace_evaluations.get_results(evaluation.id)
            if result:
                print(f"  Score: {result.score}, Passed: {result.passed}")
            else:
                print(f"  Evaluation {evaluation.id}: no results yet")
        except Exception:
            print(f"  Evaluation {evaluation.id}: results not available yet")

    # --- Clean up
    await client.judges.delete(judge.id)
    print(f"Cleaned up judge {judge.id}")


if __name__ == "__main__":
    asyncio.run(main())
