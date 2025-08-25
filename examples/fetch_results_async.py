#!/usr/bin/env -S poetry run python

import asyncio

from atlas import AsyncAtlas


async def fetch_evaluation_results(client, evaluation_id):
    """Fetch results for a single evaluation and print when loaded."""
    try:
        print(f"Fetching evaluation {evaluation_id}...")
        evaluation = await client.evaluations.get_by_id(evaluation_id)
        print(f"Found evaluation {evaluation.id}, status={evaluation.status}")
        
        # Get all results for this evaluation
        results = await client.results.get_all(evaluation=evaluation)
        print(f"Loaded {len(results)} results for evaluation {evaluation_id}")
        print(f"Results for {evaluation_id}: {results}")
        print("-" * 80)
        
        return evaluation_id, results
    except Exception as e:
        print(f"Error fetching evaluation {evaluation_id}: {e}")
        return evaluation_id, None


async def main():
    # Construct async client
    client = AsyncAtlas()

    # List of evaluation IDs to fetch exmple

    evaluation_ids = [
        "68a65a3de7ad047fb5d8e7d4",
        "688a254c673f6b2835cc7278"
    ]

    print(f"Starting async fetch for {len(evaluation_ids)} evaluations...")
    print("=" * 80)

    # Create tasks for concurrent execution
    tasks = [
        fetch_evaluation_results(client, eval_id) 
        for eval_id in evaluation_ids
    ]

    # Execute all tasks concurrently and print results as they complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("=" * 80)
    print("Summary:")
    successful = sum(1 for _, result in results if result is not None and not isinstance(result, Exception))
    print(f"Successfully fetched results for {successful}/{len(evaluation_ids)} evaluations")


if __name__ == "__main__":
    asyncio.run(main())
