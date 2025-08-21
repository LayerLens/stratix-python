#!/usr/bin/env -S poetry run python

import asyncio

from atlas import AsyncAtlas


async def main():
    # Construct async client
    client = await AsyncAtlas.create()

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
        timeout_seconds=600,  # 10 minutes
    )
    print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

    # --- Results
    if evaluation.is_success:
        results = await client.results.get(evaluation=evaluation)
        print("Results:", results)
    else:
        print("Evaluation did not succeed, no results to show.")


if __name__ == "__main__":
    asyncio.run(main())
