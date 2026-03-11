# Retrieving Results

Examples for fetching evaluation results using the LayerLens Python SDK, including pagination, bulk fetching, and concurrent retrieval.

## Paginated Results

> Source: [`examples/paginated_results.py`](../../examples/paginated_results.py)

Walk through results page by page with full control over page size.

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    evaluation = await client.evaluations.create(model=models[0], benchmark=benchmarks[0])
    evaluation = await client.evaluations.wait_for_completion(
        evaluation, interval_seconds=10, timeout_seconds=600
    )

    if evaluation.is_success:
        print("Fetching all results with pagination...")

        all_results = []
        page = 1
        page_size = 50

        while True:
            print(f"Fetching page {page} (page size: {page_size})...")

            results_data = await client.results.get_by_id(
                evaluation_id=evaluation.id, page=page, page_size=page_size
            )

            if not results_data or not results_data.results:
                print("No more results to fetch")
                break

            all_results.extend(results_data.results)

            if page == 1:
                total_count = results_data.pagination.total_count
                total_pages = results_data.pagination.total_pages
                print(f"Total results: {total_count:,}")
                print(f"Total pages: {total_pages}")

            print(f"Page {page}: Retrieved {len(results_data.results)} results")
            print(f"Running total: {len(all_results):,} results")

            if page >= results_data.pagination.total_pages:
                print("Reached last page")
                break

            page += 1

        print(f"\nTotal results collected: {len(all_results):,}")

        if all_results:
            correct_answers = sum(1 for r in all_results if r.score > 0.5)
            accuracy = correct_answers / len(all_results)
            avg_score = sum(r.score for r in all_results) / len(all_results)

            print(f"Overall accuracy: {accuracy:.1%} ({correct_answers:,}/{len(all_results):,})")
            print(f"Average score: {avg_score:.3f}")

            print(f"\nFirst 3 results:")
            for i, result in enumerate(all_results[:3], 1):
                print(f"  {i}. Score: {result.score:.3f}, Subset: {result.subset}")
                print(f"     Prompt: {result.prompt[:100]}...")
                print(f"     Response: {result.result[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
```

## All Results Without Pagination

> Source: [`examples/all_results_no_pagination.py`](../../examples/all_results_no_pagination.py)

Use `get_all()` to fetch every result in a single call. Simpler but loads everything into memory.

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    models = await client.models.get()
    benchmarks = await client.benchmarks.get()

    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0],
    )

    evaluation = await client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=10,
        timeout_seconds=600,
    )

    # Fetch all results at once
    results = await client.results.get_all(evaluation=evaluation)
    print(f"Found {len(results)} results")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
```

## Fetch Results for Multiple Evaluations Concurrently

> Source: [`examples/fetch_results_async.py`](../../examples/fetch_results_async.py)

Use `asyncio.gather` to load results for several evaluations in parallel.

```python
import asyncio

from layerlens import AsyncStratix


async def fetch_evaluation_results(client, evaluation_id):
    """Fetch results for a single evaluation and print when loaded."""
    try:
        print(f"Fetching evaluation {evaluation_id}...")
        evaluation = await client.evaluations.get_by_id(evaluation_id)
        print(f"Found evaluation {evaluation.id}, status={evaluation.status}")

        results = await client.results.get_all(evaluation=evaluation)
        print(f"Loaded {len(results)} results for evaluation {evaluation_id}")
        print(f"Results for {evaluation_id}: {results}")
        print("-" * 80)

        return evaluation_id, results
    except Exception as e:
        print(f"Error fetching evaluation {evaluation_id}: {e}")
        return evaluation_id, None


async def main():
    client = AsyncStratix()

    # Replace with your own evaluation IDs
    evaluation_ids = ["68a65a3de7ad047fb5d8e7d4", "688a254c673f6b2835cc7278"]

    print(f"Starting async fetch for {len(evaluation_ids)} evaluations...")
    print("=" * 80)

    tasks = [fetch_evaluation_results(client, eval_id) for eval_id in evaluation_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("=" * 80)
    print("Summary:")
    successful = sum(1 for _, result in results if result is not None and not isinstance(result, Exception))
    print(f"Successfully fetched results for {successful}/{len(evaluation_ids)} evaluations")


if __name__ == "__main__":
    asyncio.run(main())
```

## Using the Evaluation Object Helpers

Results can also be fetched directly from an `Evaluation` object when a client is attached:

```python
from layerlens import Stratix

client = Stratix()

# Get results via the client
results_response = client.results.get(evaluation=evaluation, page=1, page_size=50)

# Or via the evaluation object (client must be attached)
results_response = evaluation.get_results(page=1, page_size=50)
all_results = evaluation.get_all_results()

# Async equivalents
results_response = await evaluation.get_results_async(page=1, page_size=50)
all_results = await evaluation.get_all_results_async()
```
