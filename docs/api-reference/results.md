# Results

The `results` resource allows you to retrieve detailed results from completed evaluations. This provides granular insight into how your model performed on individual test cases.

## Overview

Results contain detailed information about each test case in an evaluation, including the prompt, model response, expected answer, scoring metrics, and performance data.

## Methods

### `get(evaluation_id, page=None, page_size=None, timeout=None)`

Retrieves detailed results for a specific evaluation with optional pagination support.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evaluation_id` | `str` | Yes | The evaluation identifier to get results for |
| `page` | `int \| None` | No | Page number for pagination (1-based). If not provided, returns first page or all results based on API default |
| `page_size` | `int \| None` | No | Number of results per page (default: 100). Maximum allowed may be limited by API |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a `ResultsData` object containing results, evaluation metadata, and pagination information if successful, `None` if no results are found or the evaluation doesn't exist.

The `ResultsData` object includes:
- `results`: List of `Result` objects for the current page
- `evaluation_id`: The evaluation ID
- `metrics`: Performance metrics including score ranges
- `pagination`: Pagination metadata (total_count, page_size, total_pages)

#### Examples

##### Basic Usage (All Results)
```python
from atlas import Atlas

client = Atlas()

# Get all results for a specific evaluation
results_data = client.results.get(evaluation_id="eval_12345")

if results_data:
    print(f"Evaluation ID: {results_data.evaluation_id}")
    print(f"Retrieved {len(results_data.results)} results")
    print(f"Total available: {results_data.pagination.total_count}")
    print(f"Page size: {results_data.pagination.page_size}")
    print(f"Total pages: {results_data.pagination.total_pages}")
    
    # Access individual results
    for i, result in enumerate(results_data.results[:3]):  # Show first 3
        print(f"\nResult {i+1}:")
        print(f"  Subset: {result.subset}")
        print(f"  Score: {result.score}")
        print(f"  Duration: {result.duration}")
else:
    print("No results found or evaluation doesn't exist")
```

##### Paginated Access
```python
# Get specific page with custom page size
results_data = client.results.get(
    evaluation_id="eval_12345",
    page=2,
    page_size=50
)

if results_data:
    print(f"Page 2 of {results_data.pagination.total_pages}")
    print(f"Showing {len(results_data.results)} of {results_data.pagination.total_count} total results")
    
    # Process current page
    for result in results_data.results:
        # Process each result
        pass
```

##### Iterating Through All Pages
```python
# Process all results by iterating through pages
evaluation_id = "eval_12345"
page = 1
page_size = 100

while True:
    results_data = client.results.get(
        evaluation_id=evaluation_id,
        page=page,
        page_size=page_size
    )
    
    if not results_data or not results_data.results:
        break
    
    print(f"Processing page {page}/{results_data.pagination.total_pages}")
    
    # Process current page results
    for result in results_data.results:
        # Your processing logic here
        pass
    
    # Move to next page
    if page >= results_data.pagination.total_pages:
        break
    page += 1

print("Finished processing all results")
```

#### With Custom Timeout

```python
# Get results with custom timeout (2 minutes) and pagination
results_data = client.results.get(
    evaluation_id="eval_12345",
    page=1,
    page_size=50,
    timeout=120.0
)
```

## Pagination Information

The `pagination` object in the response provides detailed pagination metadata:

```python
results_data = client.results.get(evaluation_id="eval_12345", page=1, page_size=50)

if results_data:
    pagination = results_data.pagination
    
    print(f"Current page info:")
    print(f"  Total results available: {pagination.total_count}")
    print(f"  Results per page: {pagination.page_size}")
    print(f"  Total pages: {pagination.total_pages}")
    print(f"  Results on current page: {len(results_data.results)}")
    
    # Calculate current page number (if needed)
    # Page number isn't stored in pagination object, so track it yourself
    current_page = 1  # You would track this in your code
    print(f"  Current page: {current_page}")
    
    # Check if there are more pages
    has_more_pages = current_page < pagination.total_pages
    print(f"  Has more pages: {has_more_pages}")
```

### Pagination Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_count` | `int` | Total number of results available across all pages |
| `page_size` | `int` | Number of results per page (as requested or default) |
| `total_pages` | `int` | Total number of pages available |

## Result Object

Each `Result` object contains the following properties:

### Core Properties

| Property | Type | Description |
|----------|------|-------------|
| `subset` | `str` | The benchmark subset or category this test case belongs to |
| `prompt` | `str` | The input prompt given to the model |
| `result` | `str` | The model's response/output |
| `truth` | `str` | The expected or correct answer |
| `score` | `float` | Individual score for this test case (typically 0.0 to 1.0) |
| `duration` | `timedelta` | Time taken for the model to respond |
| `metrics` | `Dict[str, float]` | Additional metrics specific to this test case |

### Understanding Properties

- **`subset`**: Groups related test cases (e.g., "elementary_mathematics", "world_history")
- **`prompt`**: The exact input sent to the model
- **`result`**: The model's actual response 
- **`truth`**: The ground truth or expected answer for comparison
- **`score`**: Individual test case score, usually binary (0.0 or 1.0) for correctness
- **`duration`**: Response latency as a Python `timedelta` object
- **`metrics`**: Additional scoring metrics that may be benchmark-specific

## Complete Example

```python
import atlas
from atlas import Atlas
from datetime import timedelta

def analyze_evaluation_results(evaluation_id: str):
    client = Atlas()
    
    try:
        # Get results
        results_data = client.results.get(evaluation_id=evaluation_id)
        
        if not results_data:
            print(f"No results found for evaluation {evaluation_id}")
            return
            
        results = results_data.results
        print(f"Analysis for evaluation {evaluation_id}")
        print(f"Total test cases: {results_data.pagination.total_count}")
        print(f"Results on current page: {len(results)}")
        
        # Calculate overall statistics for current page
        total_score = sum(result.score for result in results)
        avg_score = total_score / len(results)
        correct_answers = sum(1 for result in results if result.score > 0.5)
        accuracy = correct_answers / len(results)
        
        # Calculate timing statistics  
        durations = [result.duration for result in results]
        avg_duration = sum(durations, timedelta()) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Accuracy: {accuracy:.1%} ({correct_answers}/{len(results)})")
        print(f"   Average Duration: {avg_duration}")
        print(f"   Min Duration: {min_duration}")
        print(f"   Max Duration: {max_duration}")
        
        # Group by subset
        subset_stats = {}
        for result in results:
            if result.subset not in subset_stats:
                subset_stats[result.subset] = {"scores": [], "count": 0}
            subset_stats[result.subset]["scores"].append(result.score)
            subset_stats[result.subset]["count"] += 1
        
        print(f"\nPerformance by Subset:")
        for subset, stats in subset_stats.items():
            subset_avg = sum(stats["scores"]) / len(stats["scores"])
            subset_acc = sum(1 for s in stats["scores"] if s > 0.5) / len(stats["scores"])
            print(f"   {subset}: {subset_acc:.1%} accuracy ({subset_avg:.3f} avg score, {stats['count']} cases)")
        
        # Show some example results
        print(f"\nSample Results:")
        for i, result in enumerate(results[:3]):
            status = "Correct" if result.score > 0.5 else "Incorrect"
            print(f"\n   Example {i+1} [{result.subset}] - {status}")
            print(f"   Prompt: {result.prompt[:100]}...")
            print(f"   Model Answer: {result.result[:100]}...")
            print(f"   Expected: {result.truth[:100]}...")
            print(f"   Score: {result.score}, Duration: {result.duration}")
            
            if result.metrics:
                print(f"   Additional Metrics: {result.metrics}")
        
        return results
        
    except atlas.NotFoundError:
        print(f"Evaluation {evaluation_id} not found")
    except atlas.AuthenticationError:
        print("Authentication failed - check your API key")
    except atlas.APIConnectionError as e:
        print(f"Connection error: {e}")
    except atlas.APIError as e:
        print(f"API error: {e}")
    
    return None

if __name__ == "__main__":
    # Example usage
    evaluation_id = "eval_12345"  # Replace with actual evaluation ID
    results = analyze_evaluation_results(evaluation_id)
```

## Working with Large Result Sets

For evaluations with many test cases, use pagination to efficiently process results:

```python
from atlas import Atlas

def process_results_efficiently(evaluation_id: str, page_size: int = 100):
    """Process large result sets using pagination"""
    client = Atlas()
    
    # Get first page to understand total scope
    first_page = client.results.get(evaluation_id=evaluation_id, page=1, page_size=page_size)
    if not first_page:
        print("No results found")
        return
    
    total_count = first_page.pagination.total_count
    total_pages = first_page.pagination.total_pages
    
    print(f"Processing {total_count} results across {total_pages} pages...")
    
    # Process each page
    for page_num in range(1, total_pages + 1):
        print(f"Processing page {page_num}/{total_pages}...")
        
        # Get current page (reuse first_page for page 1)
        if page_num == 1:
            results_data = first_page
        else:
            results_data = client.results.get(
                evaluation_id=evaluation_id,
                page=page_num,
                page_size=page_size
            )
        
        if not results_data:
            print(f"Failed to get page {page_num}")
            continue
            
        # Process current page
        for result in results_data.results:
            # Your processing logic here
            pass
        
        print(f"Completed page {page_num} ({len(results_data.results)} results)")
    
    print(f"Finished processing all {total_count} results")

# Usage
process_results_efficiently("eval_12345", page_size=50)
```

### Memory-Efficient Processing

The pagination approach is more memory-efficient than loading all results at once:

```python
# Good - Memory efficient with pagination
def analyze_large_evaluation(evaluation_id: str):
    client = Atlas()
    
    # Aggregate statistics across pages
    total_processed = 0
    total_score = 0
    total_correct = 0
    
    page = 1
    page_size = 100
    
    while True:
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
            
        # Process current page
        page_score = sum(r.score for r in results_data.results)
        page_correct = sum(1 for r in results_data.results if r.score > 0.5)
        
        total_score += page_score
        total_correct += page_correct
        total_processed += len(results_data.results)
        
        print(f"Page {page}: {len(results_data.results)} results, {page_correct} correct")
        
        # Check if we're done
        if page >= results_data.pagination.total_pages:
            break
            
        page += 1
    
    # Final statistics
    overall_accuracy = total_correct / total_processed if total_processed > 0 else 0
    overall_avg_score = total_score / total_processed if total_processed > 0 else 0
    
    print(f"\nFinal Results:")
    print(f"   Total processed: {total_processed}")
    print(f"   Overall accuracy: {overall_accuracy:.1%}")
    print(f"   Overall average score: {overall_avg_score:.3f}")

# Bad - Loads everything into memory at once (may cause issues with large datasets)
def analyze_evaluation_inefficient(evaluation_id: str):
    results_data = client.results.get(evaluation_id=evaluation_id)  # No pagination
    # This could load thousands of results into memory
    for result in results_data.results:
        # Process all results at once
        pass
```

## Filtering and Analysis

### Filter by Subset

```python
def analyze_subset_performance(results, target_subset):
    subset_results = [r for r in results if r.subset == target_subset]
    
    if not subset_results:
        print(f"No results found for subset '{target_subset}'")
        return
        
    accuracy = sum(1 for r in subset_results if r.score > 0.5) / len(subset_results)
    avg_duration = sum(r.duration for r in subset_results) / len(subset_results)
    
    print(f"Subset '{target_subset}' Performance:")
    print(f"  Test cases: {len(subset_results)}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Average duration: {avg_duration}")

# Usage
results = client.results.get(evaluation_id="eval_12345")
if results:
    analyze_subset_performance(results, "elementary_mathematics")
```

### Find Difficult Cases

```python
def find_difficult_cases(results, score_threshold=0.3):
    """Find test cases where the model struggled"""
    difficult_cases = [r for r in results if r.score < score_threshold]
    
    print(f"Found {len(difficult_cases)} difficult cases (score < {score_threshold})")
    
    for case in difficult_cases[:5]:  # Show first 5
        print(f"\nDifficult Case [{case.subset}]:")
        print(f"  Prompt: {case.prompt[:100]}...")
        print(f"  Model: {case.result[:50]}...")
        print(f"  Expected: {case.truth[:50]}...")
        print(f"  Score: {case.score}")

# Usage
results = client.results.get(evaluation_id="eval_12345")
if results:
    find_difficult_cases(results)
```

## Error Handling

### Common Errors

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    results = client.results.get(evaluation_id="nonexistent_eval")
except atlas.NotFoundError:
    print("Evaluation not found or no results available")
except atlas.AuthenticationError:
    print("Authentication failed")
except atlas.PermissionDeniedError:
    print("No permission to access this evaluation")
```

### Handling Empty Results

```python
def safe_get_results(client, evaluation_id):
    """Safely get results with proper error handling"""
    try:
        results = client.results.get(evaluation_id=evaluation_id)
        
        if results is None:
            print(f"No results found for evaluation {evaluation_id}")
            print("This could mean:")
            print("- Evaluation doesn't exist")
            print("- Evaluation not yet completed")
            print("- No permission to access results")
            return []
            
        if len(results) == 0:
            print(f"Evaluation {evaluation_id} has no test cases")
            return []
            
        return results
        
    except atlas.APIError as e:
        print(f"Error retrieving results: {e}")
        return []
```

## Performance Considerations

### Large Result Sets
Results can contain thousands of individual test cases. Consider:

```python
# Good - check result size first
results = client.results.get(evaluation_id="eval_12345")
if results:
    print(f"Retrieved {len(results)} results")
    if len(results) > 1000:
        print("Large result set - consider processing in chunks")

# Bad - not considering memory usage
results = client.results.get(evaluation_id="eval_12345")
# Process all results in memory without considering size
```

### Caching Results
For repeated analysis, consider caching results:

```python
import pickle
from pathlib import Path

def get_cached_results(client, evaluation_id, cache_dir="cache"):
    cache_path = Path(cache_dir) / f"{evaluation_id}_results.pkl"
    
    if cache_path.exists():
        print("Loading cached results...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Fetching fresh results...")
    results = client.results.get(evaluation_id=evaluation_id)
    
    if results:
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    
    return results
```

## Best Practices

### 1. Always Check for Results
```python
# Good - check if results exist
results = client.results.get(evaluation_id="eval_12345")
if results:
    print(f"Found {len(results)} results")
else:
    print("No results available")

# Bad - assume results exist
results = client.results.get(evaluation_id="eval_12345") 
print(f"Found {len(results)} results")  # Could raise AttributeError
```

### 2. Handle Large Result Sets Appropriately  
```python
# Good - process in chunks for large sets
if len(results) > 1000:
    for i in range(0, len(results), 100):
        chunk = results[i:i+100]
        process_chunk(chunk)

# Bad - process everything in memory
for result in results:  # Could be thousands of results
    expensive_processing(result)
```

### 3. Use Meaningful Analysis
```python
# Good - extract meaningful insights
subset_performance = {}
for result in results:
    if result.subset not in subset_performance:
        subset_performance[result.subset] = []
    subset_performance[result.subset].append(result.score)

# Bad - just print raw data
for result in results:
    print(result.score)  # Not very useful
```

## Next Steps

- Learn about [error handling](errors.md) for robust applications
- Explore [code examples](../examples/retrieving-results.md) for common analysis patterns
- Check out [troubleshooting](../troubleshooting/) for common issues