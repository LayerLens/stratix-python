# Advanced Usage

Common patterns and best practices for using the Atlas Python SDK in production.

## Environment Setup

```python
# Set your API key as an environment variable
export LAYERLENS_ATLAS_API_KEY="your_api_key_here"

# Then in your code:
from atlas import Atlas
client = Atlas()  # Automatically reads from environment
```

## Async Operations

### Basic Async Usage

```python
import asyncio
from atlas import AsyncAtlas

async def run_evaluation():
    # Create async client
    client = AsyncAtlas()
    
    # Get models and benchmarks concurrently
    models_task = client.models.get()
    benchmarks_task = client.benchmarks.get()
    models, benchmarks = await asyncio.gather(models_task, benchmarks_task)
    
    # Create evaluation
    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0]
    )
    
    # Wait for completion
    completed = await client.evaluations.wait_for_completion(evaluation)
    
    if completed.is_success:
        # Get results
        results = await client.results.get_by_id(evaluation_id=completed.id)
        return results
    
    return None

# Run it
results = asyncio.run(run_evaluation())
```

### Multiple Concurrent Evaluations

```python
import asyncio
from atlas import AsyncAtlas

async def create_multiple_evaluations():
    client = AsyncAtlas()
    
    models = await client.models.get()
    benchmarks = await client.benchmarks.get()
    
    # Create multiple evaluations concurrently
    tasks = []
    for model in models[:3]:  # First 3 models
        for benchmark in benchmarks[:2]:  # First 2 benchmarks
            task = client.evaluations.create(model=model, benchmark=benchmark)
            tasks.append(task)
    
    # Wait for all to complete
    evaluations = await asyncio.gather(*tasks)
    
    # Filter successful ones
    successful = [e for e in evaluations if e is not None]
    print(f"Created {len(successful)} evaluations")
    
    return successful

# Run it
evaluations = asyncio.run(create_multiple_evaluations())
```

## Error Handling

### Basic Error Handling

```python
from atlas import Atlas
import atlas

def safe_create_evaluation():
    client = Atlas()
    
    try:
        models = client.models.get()
        benchmarks = client.benchmarks.get()
        
        evaluation = client.evaluations.create(
            model=models[0],
            benchmark=benchmarks[0]
        )
        
        return evaluation
        
    except atlas.AuthenticationError:
        print("Invalid API key")
        return None
    except atlas.NotFoundError:
        print("Model or benchmark not found")
        return None
    except atlas.RateLimitError as e:
        print(f"Rate limited. Try again in {e.retry_after} seconds")
        return None
    except atlas.APIError as e:
        print(f"API error: {e}")
        return None

evaluation = safe_create_evaluation()
```

### Retry Logic

```python
import time
from atlas import Atlas
import atlas

def create_evaluation_with_retry(max_retries=3):
    client = Atlas()
    
    for attempt in range(max_retries):
        try:
            models = client.models.get()
            benchmarks = client.benchmarks.get()
            
            evaluation = client.evaluations.create(
                model=models[0],
                benchmark=benchmarks[0]
            )
            
            print(f"Success on attempt {attempt + 1}")
            return evaluation
            
        except atlas.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = e.retry_after or (2 ** attempt)
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded")
                return None
                
        except atlas.APIError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    
    return None

evaluation = create_evaluation_with_retry()
```

## Timeouts

### Custom Timeouts

```python
from atlas import Atlas

# Different timeout strategies
quick_client = Atlas(timeout=30.0)      # 30 seconds for testing
normal_client = Atlas(timeout=300.0)    # 5 minutes for normal use
patient_client = Atlas(timeout=1800.0)  # 30 minutes for long evaluations

# Use appropriate client for your use case
evaluation = patient_client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0]
)
```

### Per-Request Timeouts

```python
from atlas import Atlas

client = Atlas()

# Override timeout for specific operations
models = client.models.get()
benchmarks = client.benchmarks.get()

# Long evaluation with extended timeout
evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0],
    timeout=3600.0  # 1 hour timeout for this specific request
)
```

## Batch Processing

### Process Multiple Evaluations

```python
from atlas import Atlas

def run_evaluation_batch():
    client = Atlas()
    
    models = client.models.get()
    benchmarks = client.benchmarks.get()
    
    results = []
    
    # Create evaluations
    for model in models[:2]:
        for benchmark in benchmarks[:2]:
            try:
                evaluation = client.evaluations.create(
                    model=model,
                    benchmark=benchmark
                )
                
                print(f"Created: {model.id} + {benchmark.id} = {evaluation.id}")
                results.append({
                    'model': model.id,
                    'benchmark': benchmark.id,
                    'evaluation_id': evaluation.id,
                    'status': 'created'
                })
                
                # Small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed: {model.id} + {benchmark.id} - {e}")
                results.append({
                    'model': model.id,
                    'benchmark': benchmark.id,
                    'error': str(e),
                    'status': 'failed'
                })
    
    return results

batch_results = run_evaluation_batch()
```

## Pagination Best Practices

### Memory-Efficient Pagination

```python
from atlas import Atlas

def process_large_results(evaluation_id):
    """Process large result sets without loading everything into memory"""
    client = Atlas()
    
    page = 1
    total_processed = 0
    total_correct = 0
    
    while True:
        # Get one page at a time
        results_data = client.results.get_by_id(
            evaluation_id=evaluation_id,
            page=page,
            page_size=100
        )
        
        if not results_data or not results_data.results:
            break
        
        # Process this page
        page_correct = sum(1 for r in results_data.results if r.score > 0.5)
        total_correct += page_correct
        total_processed += len(results_data.results)
        
        # Show progress
        accuracy = total_correct / total_processed
        print(f"Page {page}: {accuracy:.1%} accuracy ({total_processed:,} processed)")
        
        if page >= results_data.pagination.total_pages:
            break
        
        page += 1
    
    final_accuracy = total_correct / total_processed if total_processed > 0 else 0
    print(f"Final: {final_accuracy:.1%} accuracy ({total_processed:,} total)")
    
    return final_accuracy

accuracy = process_large_results("your_evaluation_id")
```

## Health Checks

### Simple Health Check

```python
from atlas import Atlas
import atlas

def check_atlas_health():
    """Check if Atlas API is reachable"""
    try:
        client = Atlas(timeout=10.0)
        
        # Try to get models (quick operation)
        models = client.models.get()
        
        return {
            'status': 'healthy',
            'models_count': len(models)
        }
        
    except atlas.AuthenticationError:
        return {'status': 'unhealthy', 'error': 'authentication_failed'}
    except atlas.APIConnectionError:
        return {'status': 'unhealthy', 'error': 'connection_failed'}
    except atlas.APITimeoutError:
        return {'status': 'unhealthy', 'error': 'timeout'}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

health = check_atlas_health()
if health['status'] == 'healthy':
    print(f"Atlas is healthy ({health['models_count']} models available)")
else:
    print(f"Atlas is unhealthy: {health['error']}")
```

## Monitoring and Logging

### Basic Logging

```python
import logging
from atlas import Atlas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_evaluation_with_logging():
    client = Atlas()
    
    logger.info("Starting evaluation creation")
    
    try:
        models = client.models.get()
        benchmarks = client.benchmarks.get()
        
        logger.info(f"Found {len(models)} models, {len(benchmarks)} benchmarks")
        
        evaluation = client.evaluations.create(
            model=models[0],
            benchmark=benchmarks[0]
        )
        
        logger.info(f"Created evaluation: {evaluation.id}")
        return evaluation
        
    except Exception as e:
        logger.error(f"Failed to create evaluation: {e}")
        return None

evaluation = create_evaluation_with_logging()
```
