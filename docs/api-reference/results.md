# Results

The `results` resource allows you to retrieve detailed results from completed evaluations. This provides granular insight into how your model performed on individual test cases.

## Overview

Results contain detailed information about each test case in an evaluation, including the prompt, model response, expected answer, scoring metrics, and performance data.

## Methods

### `get(evaluation_id, timeout=None)`

Retrieves detailed results for a specific evaluation.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evaluation_id` | `str` | Yes | The evaluation identifier to get results for |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a list of `Result` objects if successful, `None` if no results are found or the evaluation doesn't exist.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Get results for a specific evaluation
results = client.results.get(evaluation_id="eval_12345")

if results:
    print(f"Retrieved {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"\nResult {i+1}:")
        print(f"  Subset: {result.subset}")
        print(f"  Score: {result.score}")
        print(f"  Duration: {result.duration}")
else:
    print("No results found or evaluation doesn't exist")
```

#### With Custom Timeout

```python
# Get results with custom timeout (2 minutes)
results = client.results.get(
    evaluation_id="eval_12345",
    timeout=120.0
)
```

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
        results = client.results.get(evaluation_id=evaluation_id)
        
        if not results:
            print(f"❌ No results found for evaluation {evaluation_id}")
            return
            
        print(f"📊 Analysis for evaluation {evaluation_id}")
        print(f"📈 Total test cases: {len(results)}")
        
        # Calculate overall statistics
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
        
        print(f"\n📋 Performance by Subset:")
        for subset, stats in subset_stats.items():
            subset_avg = sum(stats["scores"]) / len(stats["scores"])
            subset_acc = sum(1 for s in stats["scores"] if s > 0.5) / len(stats["scores"])
            print(f"   {subset}: {subset_acc:.1%} accuracy ({subset_avg:.3f} avg score, {stats['count']} cases)")
        
        # Show some example results
        print(f"\n🔍 Sample Results:")
        for i, result in enumerate(results[:3]):
            status = "✅ Correct" if result.score > 0.5 else "❌ Incorrect"
            print(f"\n   Example {i+1} [{result.subset}] - {status}")
            print(f"   Prompt: {result.prompt[:100]}...")
            print(f"   Model Answer: {result.result[:100]}...")
            print(f"   Expected: {result.truth[:100]}...")
            print(f"   Score: {result.score}, Duration: {result.duration}")
            
            if result.metrics:
                print(f"   Additional Metrics: {result.metrics}")
        
        return results
        
    except atlas.NotFoundError:
        print(f"❌ Evaluation {evaluation_id} not found")
    except atlas.AuthenticationError:
        print("❌ Authentication failed - check your API key")
    except atlas.APIConnectionError as e:
        print(f"❌ Connection error: {e}")
    except atlas.APIError as e:
        print(f"❌ API error: {e}")
    
    return None

if __name__ == "__main__":
    # Example usage
    evaluation_id = "eval_12345"  # Replace with actual evaluation ID
    results = analyze_evaluation_results(evaluation_id)
```

## Working with Large Result Sets

For evaluations with many test cases, consider processing results in batches:

```python
from atlas import Atlas

def process_results_efficiently(evaluation_id: str):
    client = Atlas()
    
    results = client.results.get(evaluation_id=evaluation_id)
    if not results:
        return
    
    print(f"Processing {len(results)} results...")
    
    # Process in chunks to avoid memory issues with very large result sets
    chunk_size = 100
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        
        print(f"Processing results {i+1}-{min(i+chunk_size, len(results))}...")
        
        # Process this chunk
        for result in chunk:
            # Your processing logic here
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
# ✅ Good - check result size first
results = client.results.get(evaluation_id="eval_12345")
if results:
    print(f"Retrieved {len(results)} results")
    if len(results) > 1000:
        print("Large result set - consider processing in chunks")

# ❌ Bad - not considering memory usage
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
# ✅ Good - check if results exist
results = client.results.get(evaluation_id="eval_12345")
if results:
    print(f"Found {len(results)} results")
else:
    print("No results available")

# ❌ Bad - assume results exist
results = client.results.get(evaluation_id="eval_12345") 
print(f"Found {len(results)} results")  # Could raise AttributeError
```

### 2. Handle Large Result Sets Appropriately  
```python
# ✅ Good - process in chunks for large sets
if len(results) > 1000:
    for i in range(0, len(results), 100):
        chunk = results[i:i+100]
        process_chunk(chunk)

# ❌ Bad - process everything in memory
for result in results:  # Could be thousands of results
    expensive_processing(result)
```

### 3. Use Meaningful Analysis
```python
# ✅ Good - extract meaningful insights
subset_performance = {}
for result in results:
    if result.subset not in subset_performance:
        subset_performance[result.subset] = []
    subset_performance[result.subset].append(result.score)

# ❌ Bad - just print raw data
for result in results:
    print(result.score)  # Not very useful
```

## Next Steps

- Learn about [error handling](errors.md) for robust applications
- Explore [code examples](../examples/retrieving-results.md) for common analysis patterns
- Check out [troubleshooting](../troubleshooting/) for common issues