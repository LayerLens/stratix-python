# Creating Evaluations

Examples for creating evaluations on the Atlas platform using the Layerlens python sdk. 

> Before running the below examples ensure the model and benchmark being run are present on your organiztion.

## Basic Evaluation

### Using Synchronous Client

Below is an example showing how to trigger an evaluation, waiting for it to complete and finally fetching the evaluations results.

```python
from atlas import Atlas

# Construct sync client (API key from env or inline)
client = Atlas()

# --- Models
models = client.models.get(type="public", name="gpt-4o")

if not models:
    print("gpt-4o not found")

model = models[0]

# --- Benchmarks
benchmarks = client.benchmarks.get(type="public", name="simpleQA")

if not benchmarks:
    print("SimpleQA benchmark not found, exiting")

benchmark = benchmarks[0]

# --- Create evaluation
evaluation = client.evaluations.create(
    model=model,
    benchmark=benchmark,
)

# --- Wait for completion
evaluation = client.evaluations.wait_for_completion(
    evaluation,
    interval_seconds=10,
    timeout_seconds=600,  # 10 minutes
)

# --- Results
if evaluation.is_success:
    # Loads the first page of results
    results = client.results.get(evaluation=evaluation)
    print("Results:", results)

```

### Using Async Client

```python
import asyncio

from atlas import AsyncAtlas


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get(type="public", name="gpt-4o")
    model = models[0]

    # --- Benchmarks
    benchmarks = await client.benchmarks.get(type="public", name="simpleQA")
    benchmark = benchmarks[0]


    # --- Create evaluation
    evaluation = await client.evaluations.create(model=model, benchmark=benchmark)


    await evaluation.wait_for_completion_async(interval_seconds=10)

    # --- Results
    if evaluation.is_success:
        results = await evaluation.get_results_async()


if __name__ == "__main__":
    asyncio.run(main())
```


## Error Handling

```python
from atlas import Atlas
import atlas

client = Atlas()

try:
    models = client.models.get()
    benchmarks = client.benchmarks.get()
    
    evaluation = client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0]
    )
    
except atlas.AuthenticationError:
    print("Check your API key")
except atlas.NotFoundError:
    print("Model or benchmark not found")
except atlas.APIError as e:
    print(f"API error: {e}")
```

## Triggering Multiple Evaluations

```python
import asyncio

from atlas import AsyncAtlas


async def create_and_run_evaluation(client, model, benchmark, eval_number):
    """Create and run a single evaluation, tracking progress."""
    try:
        print(f"Starting evaluation #{eval_number}...")

        # Create evaluation
        evaluation = await client.evaluations.create(model=model, benchmark=benchmark)

        # Wait for completion
        evaluation = await client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=10,
            timeout_seconds=600,  # 10 minutes
        )

        # Get results if successful
        if evaluation.is_success:
            results = await client.results.get_all(evaluation=evaluation)
            return results
        else:
            return None

    except Exception as e:
        print(f"✗ Error in evaluation #{eval_number}: {e}")
        return eval_number, None, 0, False


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get()

    # --- Benchmarks
    benchmarks = await client.benchmarks.get()

    # Use first model and benchmark for all evaluations
    target_model = models[0]
    target_benchmark = benchmarks[0]

    print(f"Using model: {target_model}")
    print(f"Using benchmark: {target_benchmark}")

    # Create 3 evaluation tasks
    num_evaluations = 3
    print(f"Starting {num_evaluations} evaluations in parallel...")

    tasks = [create_and_run_evaluation(client, target_model, target_benchmark, i + 1) for i in range(num_evaluations)]

    # Execute all evaluations concurrently
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Fetching Results of Multiple Evaluations Async

```python

import asyncio

from atlas import AsyncAtlas


async def fetch_evaluation_results(client, evaluation_id):
    """Fetch results for a single evaluation and print when loaded."""
    try:
        print(f"Fetching evaluation {evaluation_id}...")
        evaluation = await client.evaluations.get_by_id(evaluation_id)
        # Get all results for this evaluation
        results = await client.results.get_all(evaluation=evaluation)
        print(f"Loaded {len(results)} results for evaluation {evaluation_id}")

        return evaluation_id, results
    except Exception as e:
        print(f"Error fetching evaluation {evaluation_id}: {e}")
        return evaluation_id, None


async def main():
    # Construct async client
    client = AsyncAtlas()

    # List of example evaluation IDs to fetch

    evaluation_ids = ["68a65a3de7ad047fbd8e7d4", "688a54c673f6b2835cc7278"]

    print(f"Starting async fetch for {len(evaluation_ids)} evaluations...")

    # Create tasks for concurrent execution
    tasks = [fetch_evaluation_results(client, eval_id) for eval_id in evaluation_ids]

    # Execute all tasks concurrently and print results as they complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("=" * 80)
    print("Summary:")
    successful = sum(1 for _, result in results if result is not None and not isinstance(result, Exception))

if __name__ == "__main__":
    asyncio.run(main())

```