#!/usr/bin/env -S poetry run python

import asyncio

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

    # --- Create a judge
    judge = await client.judges.create(
        name="Response Quality Judge",
        evaluation_goal="Evaluate whether the response is accurate, helpful, and well-structured",
    )
    print(f"Created judge {judge.id}: {judge.name}")

    # --- Upload traces
    result = await client.traces.upload("./traces.jsonl")
    print(f"Uploaded {result.count} traces")

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

    # --- Fetch results for all evaluations concurrently
    result_tasks = [client.trace_evaluations.get_results(e.id) for e in evaluations if e]
    all_results = await asyncio.gather(*result_tasks)

    for results_response in all_results:
        if results_response:
            for result in results_response.results:
                print(f"  Score: {result.score}, Passed: {result.passed}")


if __name__ == "__main__":
    asyncio.run(main())
