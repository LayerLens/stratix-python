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

### With Progress Tracking

```python
from atlas import Atlas

def get_all_results_with_progress(evaluation_id):
    client = Atlas()
    all_results = []
    page = 1
    page_size = 50
    
    while True:
        print(f"Fetching page {page}...")
        
        results_data = client.results.get_by_id(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
        
        all_results.extend(results_data.results)
        
        # Show progress
        if page == 1:
            total_count = results_data.pagination.total_count
            total_pages = results_data.pagination.total_pages
            print(f"Total results: {total_count:,} across {total_pages} pages")
        
        progress = len(all_results) / results_data.pagination.total_count * 100
        print(f"Progress: {progress:.1f}% ({len(all_results):,} collected)")
        
        if page >= results_data.pagination.total_pages:
            break
            
        page += 1
    
    return all_results

# Usage
results = get_all_results_with_progress("your_evaluation_id")
print(f"Done! Collected {len(results):,} results")
```

## Async Version

```python
import asyncio
from atlas import AsyncAtlas

async def get_results_async(evaluation_id):
    client = AsyncAtlas()
    
    all_results = []
    page = 1
    
    while True:
        results_data = await client.results.get_by_id(
            evaluation_id=evaluation_id,
            page=page,
            page_size=100
        )
        
        if not results_data or not results_data.results:
            break
        
        all_results.extend(results_data.results)
        print(f"Page {page}: {len(results_data.results)} results")
        
        if page >= results_data.pagination.total_pages:
            break
        
        page += 1
    
    return all_results

# Run it
results = asyncio.run(get_results_async("your_evaluation_id"))
print(f"Total: {len(results)} results")
```

## Complete Workflow

```python
from atlas import Atlas

    client = Atlas()
    
# 1. Create evaluation
models = client.models.get()
benchmarks = client.benchmarks.get()

evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0]
)
print(f"Created evaluation: {evaluation.id}")

# 2. Wait for completion
print("Waiting for evaluation to complete...")
completed_evaluation = client.evaluations.wait_for_completion(
    evaluation,
    interval_seconds=30,
    timeout_seconds=1800  # 30 minutes
)

# 3. Get all results
if completed_evaluation.is_success:
    print("Getting results...")
    
    all_results = []
    page = 1
    
    while True:
        results_data = client.results.get_by_id(
            evaluation_id=completed_evaluation.id,
            page=page,
            page_size=100
        )
        
        if not results_data or not results_data.results:
            break
        
        all_results.extend(results_data.results)
        if page >= results_data.pagination.total_pages:
            break
        page += 1
    
    # 4. Analyze results
    correct = sum(1 for r in all_results if r.score > 0.5)
    accuracy = correct / len(all_results)
    avg_score = sum(r.score for r in all_results) / len(all_results)
    
    print(f"Results: {len(all_results):,} total")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average score: {avg_score:.3f}")
else:
    print(f"Evaluation failed: {completed_evaluation.status}")
```

## Analyze Results by Subset

```python
from atlas import Atlas
from collections import defaultdict

    client = Atlas()
evaluation_id = "your_evaluation_id"

# Get all results
all_results = []
page = 1

while True:
    results_data = client.results.get_by_id(evaluation_id=evaluation_id, page=page)
    if not results_data or not results_data.results:
        break
    all_results.extend(results_data.results)
    if page >= results_data.pagination.total_pages:
        break
    page += 1

# Group by subset
subset_results = defaultdict(list)
for result in all_results:
    subset_results[result.subset].append(result)

# Analyze each subset
print(f"Analysis by subset:")
for subset, results in subset_results.items():
    correct = sum(1 for r in results if r.score > 0.5)
    accuracy = correct / len(results)
    avg_score = sum(r.score for r in results) / len(results)
    
    print(f"  {subset}:")
    print(f"    Cases: {len(results)}")
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Avg Score: {avg_score:.3f}")
```

## Key Points

- Results are paginated with default page size of 100
- Always loop through all pages to get complete results
- Use `page_size` parameter to control how many results per page (1-500)
- Check `pagination.total_pages` to know when to stop
- Results contain `score`, `subset`, `prompt`, `result`, and `truth` fields
- Use async versions for better performance with large datasets
