#!/usr/bin/env -S poetry run python

import asyncio

from atlas import AsyncAtlas


async def create_and_run_evaluation(client, model, benchmark, eval_number):
    """Create and run a single evaluation, tracking progress."""
    try:
        print(f"Starting evaluation #{eval_number}...")

        # Create evaluation
        evaluation = await client.evaluations.create(model=model, benchmark=benchmark)
        print(f"✓ Created evaluation #{eval_number}: {evaluation.id}, status={evaluation.status}")

        # Wait for completion
        evaluation = await client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=10,
            timeout_seconds=600,  # 10 minutes
        )
        print(f"✓ Evaluation #{eval_number} ({evaluation.id}) finished with status={evaluation.status}")

        # Get results if successful
        if evaluation.is_success:
            results = await client.results.get_all(evaluation=evaluation)
            print(f"✓ Evaluation #{eval_number} completed with {len(results)} results")
            return eval_number, evaluation.id, len(results), True
        else:
            print(f"✗ Evaluation #{eval_number} did not succeed")
            return eval_number, evaluation.id, 0, False

    except Exception as e:
        print(f"✗ Error in evaluation #{eval_number}: {e}")
        return eval_number, None, 0, False


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get()
    print(f"Found {len(models)} models")

    # --- Benchmarks
    benchmarks = await client.benchmarks.get()
    print(f"Found {len(benchmarks)} benchmarks")

    # Use first model and benchmark for all evaluations
    target_model = models[0]
    target_benchmark = benchmarks[0]

    print(f"Using model: {target_model}")
    print(f"Using benchmark: {target_benchmark}")
    print("=" * 80)

    # Create 3 evaluation tasks
    num_evaluations = 3
    print(f"Starting {num_evaluations} evaluations in parallel...")

    tasks = [create_and_run_evaluation(client, target_model, target_benchmark, i + 1) for i in range(num_evaluations)]

    # Execute all evaluations concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    print("=" * 80)
    print("SUMMARY:")
    successful = 0
    total_results = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"Exception occurred: {result}")
        else:
            eval_num, eval_id, result_count, success = result
            if success:
                successful += 1
                total_results += result_count
                print(f"Evaluation #{eval_num} ({eval_id}): SUCCESS - {result_count} results")
            else:
                print(f"Evaluation #{eval_num} ({eval_id}): FAILED")

    print(f"\nOverall: {successful}/{num_evaluations} evaluations succeeded")
    print(f"Total results collected: {total_results}")


if __name__ == "__main__":
    asyncio.run(main())
