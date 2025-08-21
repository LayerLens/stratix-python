# Working with Timeouts

Simple examples for configuring timeouts with the Atlas Python SDK.

## Basic Timeouts

### Set Client Timeout

```python
from atlas import Atlas

# Set timeout for all requests (in seconds)
client = Atlas(timeout=300.0)  # 5 minutes

# Use the client normally
models = client.models.get()
evaluation = client.evaluations.create(model=models[0], benchmark=benchmarks[0])
```

### Different Timeouts for Different Use Cases

```python
from atlas import Atlas

# Quick operations (testing, health checks)
quick_client = Atlas(timeout=30.0)  # 30 seconds

# Normal operations
normal_client = Atlas(timeout=300.0)  # 5 minutes

# Long operations (large evaluations)
patient_client = Atlas(timeout=1800.0)  # 30 minutes

# Use appropriate client
models = quick_client.models.get()  # Fast operation
evaluation = patient_client.evaluations.create(...)  # Slow operation
```

## Per-Request Timeouts

### Override Timeout for Specific Requests

```python
from atlas import Atlas

client = Atlas(timeout=60.0)  # Default 1 minute

# Override timeout for specific operations
models = client.models.get()
benchmarks = client.benchmarks.get()

# This evaluation might take longer, so use extended timeout
evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0],
    timeout=1800.0  # 30 minutes for this specific request
)
```

## Async Timeouts

```python
import asyncio
from atlas import AsyncAtlas

async def main():
    # Set timeout for async client
    client = await AsyncAtlas.create(timeout=300.0)
    
    models = await client.models.get()
    benchmarks = await client.benchmarks.get()
    
    # Create evaluation with custom timeout
    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0],
        timeout=1200.0  # 20 minutes
    )
    
    return evaluation

asyncio.run(main())
```

## Handling Timeout Errors

```python
from atlas import Atlas
import atlas

def safe_operation_with_timeout():
    client = Atlas(timeout=60.0)  # 1 minute timeout
    
    try:
        models = client.models.get()
        benchmarks = client.benchmarks.get()
        
        evaluation = client.evaluations.create(
            model=models[0],
            benchmark=benchmarks[0]
        )
        
        return evaluation
            
    except atlas.APITimeoutError:
        print("Operation timed out")
        print("Try increasing the timeout or check your connection")
        return None
        
    except atlas.APIError as e:
        print(f"Other API error: {e}")
    return None

result = safe_operation_with_timeout()
```

## Wait for Completion with Timeout

```python
from atlas import Atlas

client = Atlas()
models = client.models.get()
benchmarks = client.benchmarks.get()

# Create evaluation
evaluation = client.evaluations.create(model=models[0], benchmark=benchmarks[0])

# Wait for completion with timeout
try:
    completed = client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=30,  # Check every 30 seconds
        timeout_seconds=3600         # Give up after 1 hour
    )
    
    if completed and completed.is_success:
        print("Evaluation completed successfully")
    else:
        print("Evaluation failed")
        
except atlas.APITimeoutError:
    print("Evaluation did not complete within 1 hour")
    print(f"Current status: {evaluation.status}")
```

## Advanced Timeout Configuration

### Granular Control

```python
import httpx
from atlas import Atlas

# Configure different timeouts for different operations
client = Atlas(
    timeout=httpx.Timeout(
        connect=10.0,   # 10 seconds to connect
        read=600.0,     # 10 minutes to read response
        write=30.0,     # 30 seconds to send request
        pool=30.0       # 30 seconds for connection pool
    )
)

# Use normally
evaluation = client.evaluations.create(model=models[0], benchmark=benchmarks[0])
```

## Recommended Timeouts

### By Operation Type

```python
from atlas import Atlas

# Quick operations (< 30 seconds)
quick_client = Atlas(timeout=30.0)
models = quick_client.models.get()
benchmarks = quick_client.benchmarks.get()

# Normal operations (5 minutes)
normal_client = Atlas(timeout=300.0)
evaluation = normal_client.evaluations.create(...)

# Long operations (30+ minutes)
long_client = Atlas(timeout=1800.0)
completed = long_client.evaluations.wait_for_completion(...)

# Getting results (depends on size)
results_client = Atlas(timeout=600.0)  # 10 minutes
results = results_client.results.get_by_id(...)
```

## Production Best Practices

### Adaptive Timeouts

```python
from atlas import Atlas
import atlas

def create_evaluation_with_adaptive_timeout(model, benchmark):
    """Try with increasing timeouts"""
    
    timeouts = [60.0, 300.0, 900.0]  # 1min, 5min, 15min
    
    for timeout in timeouts:
        try:
            client = Atlas(timeout=timeout)
            
            print(f"Trying with {timeout}s timeout...")
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            
            print(f"Success with {timeout}s timeout")
            return evaluation
            
        except atlas.APITimeoutError:
            print(f"Timed out with {timeout}s timeout")
            continue
        except atlas.APIError as e:
            print(f"API error (not timeout): {e}")
            break
    
    print("Failed with all timeout attempts")
    return None

# Usage
models = Atlas().models.get()
benchmarks = Atlas().benchmarks.get()
evaluation = create_evaluation_with_adaptive_timeout(models[0], benchmarks[0])
```
