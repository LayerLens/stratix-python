# Retrieving Results

Simple examples for getting evaluation results with the Atlas Python SDK.

## Basic Result Retrieval

### Get Results (First Page)

```python
from atlas import Atlas

client = Atlas()

# Get results for a specific evaluation
evaluation_id = "your_evaluation_id"
results = client.results.get_by_id(evaluation_id=evaluation_id)

if results:
    print(f"Results for evaluation: {results.evaluation_id}")
    print(f"Total results: {results.pagination.total_count:,}")
    print(f"Showing page 1 of {results.pagination.total_pages}")
    print(f"Results on this page: {len(results.results)}")
    
    # Show first few results
    for i, result in enumerate(results.results[:3]):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result.score}")
        print(f"  Subset: {result.subset}")
        print(f"  Prompt: {result.prompt[:100]}...")
        print(f"  Response: {result.result[:50]}...")
else:
    print("No results found")
```

## Get All Results Without Manual Pagination

### Using get_all() Method

The easiest way to retrieve all results is using the `get_all()` method, which handles pagination automatically:

```python
import asyncio
from atlas import AsyncAtlas

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
```

This approach is much simpler than manual pagination as it:
- Automatically handles all pagination internally
- Returns all results in a single call
- No need to manage page numbers or loop through pages
- Works with both sync and async clients

## Pagination - Get All Results (Manual)

### Simple Approach

```python
from atlas import Atlas

client = Atlas()
evaluation_id = "your_evaluation_id"

# Get all results across all pages
        all_results = []
        page = 1
        
        while True:
    results_data = client.results.get_by_id(
                evaluation_id=evaluation_id,
                page=page,
        page_size=100  # 100 results per page
            )
            
            if not results_data or not results_data.results:
                break
                
            all_results.extend(results_data.results)
    print(f"Loaded page {page}: {len(results_data.results)} results")
    
    # Check if we've reached the last page
            if page >= results_data.pagination.total_pages:
                break
                
            page += 1
        
print(f"\nTotal results collected: {len(all_results):,}")

# Calculate accuracy
correct = sum(1 for r in all_results if r.score > 0.5)
accuracy = correct / len(all_results) if all_results else 0
print(f"Overall accuracy: {accuracy:.1%} ({correct:,}/{len(all_results):,})")
```


## Key Points

- Results are paginated with default page size of 100
- Always loop through all pages to get complete results
- Use `page_size` parameter to control how many results per page (1-500)
- Check `pagination.total_pages` to know when to stop
- Results contain `score`, `subset`, `prompt`, `result`, and `truth` fields
- Use async versions for better performance with large datasets
