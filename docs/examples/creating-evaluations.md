# Creating Evaluations

Examples for creating evaluations on the Stratix platform using the LayerLens Python SDK.

> Before running the below examples ensure the model and benchmark being run are present on your organization.

## Basic Evaluation

### Using Synchronous Client

> Source: [`examples/client.py`](../../examples/client.py)

```python
from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Models
models = client.models.get()
print(f"Found {len(models)} models")

# --- Benchmarks
benchmarks = client.benchmarks.get()
print(f"Found {len(benchmarks)} benchmarks")

# --- Create evaluation
evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0],
)
print(f"Created evaluation {evaluation.id}, status={evaluation.status}")

# --- Wait for completion
evaluation = client.evaluations.wait_for_completion(
    evaluation,
    interval_seconds=10,
    timeout_seconds=600,  # 10 minutes
)
print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

# --- Results
if evaluation.is_success:
    results = client.results.get(evaluation=evaluation)
    print("Results:", results)
else:
    print("Evaluation did not succeed, no results to show.")
```

### Minimal Sync Example

> Source: [`examples/client_simple.py`](../../examples/client_simple.py)

```python
from layerlens import Stratix

client = Stratix()

models = client.models.get(type="public", name="gpt-4o")
model = models[0]

benchmarks = client.benchmarks.get(type="public", name="simpleQA")
benchmark = benchmarks[0]

evaluation = client.evaluations.create(
    model=model,
    benchmark=benchmark,
)
```

### Using Async Client

> Source: [`examples/async_client_simple.py`](../../examples/async_client_simple.py)

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    print(f"Found {len(models)} models")

    benchmarks = await client.benchmarks.get()
    print(f"Found {len(benchmarks)} benchmarks")

    evaluation = await client.evaluations.create(model=models[0], benchmark=benchmarks[0])
    print(f"Created evaluation {evaluation.id}, status={evaluation.status}")

    await evaluation.wait_for_completion_async(interval_seconds=10, timeout_seconds=600)
    print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

    if evaluation.is_success:
        results = await evaluation.get_results_async()
        print("Results:", results)
    else:
        print("Evaluation did not succeed, no results to show.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Sorting and Filtering Evaluations

> Source: [`examples/evaluation_sorting.py`](../../examples/evaluation_sorting.py)

```python
import asyncio

from layerlens import AsyncStratix
from layerlens.models import EvaluationStatus


async def main():
    client = AsyncStratix()

    # --- Sort by accuracy (highest first)
    response = await client.evaluations.get_many(
        sort_by="accuracy",
        order="desc",
        page_size=10,
    )
    if response:
        print(f"Top {len(response.evaluations)} evaluations by accuracy:")
        for evaluation in response.evaluations:
            print(f"  - {evaluation.id}: accuracy={evaluation.accuracy:.2f}%")

    # --- Filter by status (only successful)
    response = await client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
    )
    if response:
        print(f"Successful evaluations: {response.pagination.total_count}")

    # --- Filter by model or benchmark IDs
    response = await client.evaluations.get_many(
        model_ids=["your-model-id"],
        sort_by="accuracy",
        order="desc",
    )

    # --- Combine sorting, filtering, and pagination
    response = await client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
        page=1,
        page_size=20,
    )
    if response:
        print(f"Page 1: {response.pagination.total_count} total, {response.pagination.total_pages} pages")


if __name__ == "__main__":
    asyncio.run(main())
```

## Comparing Evaluations

> Source: [`examples/compare_evaluations.py`](../../examples/compare_evaluations.py)

```python
from layerlens import PublicClient

client = PublicClient()

# Compare two models on a benchmark
comparison = client.comparisons.compare_models(
    benchmark_id="682bddc1e014f9fa440f8a91",
    model_id_1="699f9761e014f9c3072b0513",
    model_id_2="699f9761e014f9c3072b0512",
    page=1,
    page_size=10,
)

if comparison:
    print(f"Model 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
    print(f"Model 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")

# Filter: where model 1 fails but model 2 succeeds
comparison = client.comparisons.compare_models(
    benchmark_id="682bddc1e014f9fa440f8a91",
    model_id_1="699f9761e014f9c3072b0513",
    model_id_2="699f9761e014f9c3072b0512",
    outcome_filter="reference_fails",
)

# Or compare using evaluation IDs directly
comparison = client.comparisons.compare(
    evaluation_id_1="699f9938a03d70bf6607081f",
    evaluation_id_2="699f991ca782d00ebd666ba1",
)
```

## Running Multiple Evaluations in Parallel

> Source: [`examples/async_run_evaluations.py`](../../examples/async_run_evaluations.py)

```python
import asyncio

from layerlens import AsyncStratix


async def create_and_run_evaluation(client, model, benchmark, eval_number):
    try:
        evaluation = await client.evaluations.create(model=model, benchmark=benchmark)

        evaluation = await client.evaluations.wait_for_completion(
            evaluation,
            interval_seconds=10,
            timeout_seconds=600,
        )

        if evaluation.is_success:
            results = await client.results.get_all(evaluation=evaluation)
            print(f"Evaluation #{eval_number} completed with {len(results)} results")
            return eval_number, evaluation.id, len(results), True
        else:
            return eval_number, evaluation.id, 0, False

    except Exception as e:
        print(f"Error in evaluation #{eval_number}: {e}")
        return eval_number, None, 0, False


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    num_evaluations = 3
    tasks = [
        create_and_run_evaluation(client, models[0], benchmarks[0], i + 1)
        for i in range(num_evaluations)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Fetching Results

### Paginated Results

> Source: [`examples/paginated_results.py`](../../examples/paginated_results.py)

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    evaluation = await client.evaluations.create(model=models[0], benchmark=benchmarks[0])
    evaluation = await client.evaluations.wait_for_completion(evaluation, interval_seconds=10, timeout_seconds=600)

    if evaluation.is_success:
        all_results = []
        page = 1
        page_size = 50

        while True:
            results_data = await client.results.get_by_id(
                evaluation_id=evaluation.id, page=page, page_size=page_size
            )

            if not results_data or not results_data.results:
                break

            all_results.extend(results_data.results)

            if page >= results_data.pagination.total_pages:
                break
            page += 1

        print(f"Total results collected: {len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### All Results Without Pagination

> Source: [`examples/all_results_no_pagination.py`](../../examples/all_results_no_pagination.py)

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    evaluation = await client.evaluations.create(model=models[0], benchmark=benchmarks[0])
    evaluation = await client.evaluations.wait_for_completion(evaluation, interval_seconds=10, timeout_seconds=600)

    # Fetch all results at once
    results = await client.results.get_all(evaluation=evaluation)
    print(f"Found {len(results)} results")


if __name__ == "__main__":
    asyncio.run(main())
```

### Fetch Results for Multiple Evaluations Concurrently

> Source: [`examples/fetch_results_async.py`](../../examples/fetch_results_async.py)

```python
import asyncio

from layerlens import AsyncStratix


async def fetch_evaluation_results(client, evaluation_id):
    try:
        evaluation = await client.evaluations.get_by_id(evaluation_id)
        results = await client.results.get_all(evaluation=evaluation)
        print(f"Loaded {len(results)} results for evaluation {evaluation_id}")
        return evaluation_id, results
    except Exception as e:
        print(f"Error fetching evaluation {evaluation_id}: {e}")
        return evaluation_id, None


async def main():
    client = AsyncStratix()

    evaluation_ids = ["68a65a3de7ad047fb5d8e7d4", "688a254c673f6b2835cc7278"]

    tasks = [fetch_evaluation_results(client, eval_id) for eval_id in evaluation_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for _, result in results if result is not None and not isinstance(result, Exception))
    print(f"Successfully fetched {successful}/{len(evaluation_ids)} evaluations")


if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

```python
from layerlens import Stratix
import layerlens

client = Stratix()

try:
    models = client.models.get()
    benchmarks = client.benchmarks.get()

    evaluation = client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0],
    )

except layerlens.AuthenticationError:
    print("Check your API key")
except layerlens.NotFoundError:
    print("Model or benchmark not found")
except layerlens.APIError as e:
    print(f"API error: {e}")
```
