#!/usr/bin/env -S poetry run python

import asyncio

from layerlens import AsyncAtlas


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get()
    print(f"Found {len(models)} models")

    # --- Benchmarks
    benchmarks = await client.benchmarks.get()
    print(f"Found {len(benchmarks)} benchmarks")

    # --- Create evaluation
    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0],
    )
    print(f"Created evaluation {evaluation.id}, status={evaluation.status}")

    # --- Wait for completion
    evaluation = await client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=10,
        # Keep in mind that the evaluation will take a while to complete, so you may want to increase the timeout
        # or grab the evaluation id and check the status later
        timeout_seconds=600,  # 10 minutes
    )
    print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

    # --- All results at once without pagination
    results = await client.results.get_all(evaluation=evaluation)
    print(f"Found {len(results)} results")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
