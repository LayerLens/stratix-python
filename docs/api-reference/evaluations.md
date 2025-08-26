# Evaluations

The `evaluations` resource on the atlas client allows you to create and manage evaluations against various benchmarks for your organization on the atlas platform. This is one of the core functionalities of the Atlas platform.

## Overview

An evaluation runs a specified model against a benchmark dataset and returns comprehensive metrics.


The below example trigger evaluations using `gpt-4o` against `simpleQA`.

> Before running the below examples ensure the model and benchmark being run are present on your organiztion.

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

## Methods

Both the `Atlas` (synchronous) and `AsyncAtlas` (asynchronous) clients support the following methods.

### `create(model, benchmark, timeout=None)`

Creates a new evaluation for the specified model and benchmark.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `Model` | Yes | The model to evaluate |
| `benchmark` | `Benchmark` | Yes | The benchmark to evaluate |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns an `Evaluation` object if successful, `None` if the evaluation could not be created.

### `wait_for_completion(evaluation, interval_seconds=30, timeout_seconds=None)`

Polls an evaluation until it completes (success, failure, or timeout) or the specified timeout is reached.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evaluation` | `Evaluation` | Yes | The evaluation object to monitor |
| `interval_seconds` | `int` | No | Polling interval in seconds (default: 30) |
| `timeout_seconds` | `int \| None` | No | Maximum time to wait in seconds (no limit if None) |

#### Returns

Returns the updated `Evaluation` object when completed, or `None` if polling fails.


### `get_by_id(evaluation_id, timeout=None)`

Retrieves an existing evaluation by its unique identifier.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evaluation_id` | `str` | Yes | The unique evaluation identifier |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns an `Evaluation` object if found, `None` if the evaluation does not exist or cannot be accessed.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Retrieve an evaluation by ID
evaluation_id = "eval_abc123xyz"
evaluation = client.evaluations.get_by_id(evaluation_id)
```

#### Async Usage

```python
from atlas import AsyncAtlas
import asyncio

async def get_evaluation():
    client = AsyncAtlas()
    
    evaluation = await client.evaluations.get_by_id("eval_abc123xyz")
    if evaluation:
        print(f"Found evaluation: {evaluation.id}")
        return evaluation
    else:
        print("Evaluation not found")
        return None

# Run the async function
asyncio.run(get_evaluation())
```

### `get_many(page=None, page_size=None, timeout=None)`

Retrieves multiple evaluations with optional pagination support.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | `int \| None` | No | Page number for pagination (1-based, defaults to 1) |
| `page_size` | `int \| None` | No | Number of evaluations per page (default: 100, max: 500) |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns an `EvaluationsResponse` object containing:
- `evaluations`: List of `Evaluation` objects
- `pagination`: Pagination metadata with `page`, `page_size`, `total_pages`, and `total_count`

Returns `None` if the request fails.


### `get_results(page=None, page_size=None, timeout=None)`

Fetches results for this evaluation with pagination support. This is a synchronous method.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | `int \| None` | No | Page number for pagination (1-based, defaults to 1) |
| `page_size` | `int \| None` | No | Number of results per page (default: 100, max: 500) |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a `ResultsResponse` object containing results and pagination metadata, or `None` if the request fails.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Get evaluation first
evaluation = client.evaluations.get_by_id("eval_12345")
if not evaluation:
    print("Evaluation not found")
else:
    #
    results_first_page = evaluation.get_results()
```

Fetches all results for this evaluation by automatically handling pagination. This is a synchronous method

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Get an evaluation
evaluation = client.evaluations.get_by_id("eval_12345")

if evaluation:
    # Fetch all results (handles pagination automatically)
    all_results = evaluation.get_all_results()
```

{{ ... }}

| `"success"` | Evaluation finished successfully |
| `"failure"` | Evaluation failed due to an error |



### `get_results_async(page=None, page_size=None, timeout=None)`

Asynchronously fetches results for this evaluation with pagination support. This requires an async client to be attached.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | `int \| None` | No | Page number for pagination (1-based, defaults to 1) |
| `page_size` | `int \| None` | No | Number of results per page (default: 100, max: 500) |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a `ResultsResponse` object containing results and pagination metadata, or `None` if the request fails.

#### Example

```python
from atlas import AsyncAtlas
import asyncio

async def fetch_evaluation_results():
    client = AsyncAtlas()
    
    # Get an evaluation
    evaluation = await client.evaluations.get_by_id("eval_12345")
    
    if evaluation:
        # Fetch first page of results asynchronously
        first_page_results = await evaluation.get_results_async(page=1, page_size=50)
        
        if first_page_results:
            return first_page_results
    
    return []

results = asyncio.run(fetch_evaluation_results())
```

### `get_all_results(timeout=None)`
Fetches all results for this evaluation by automatically handling pagination. This is a synchronous method.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Get an evaluation
evaluation = client.evaluations.get_by_id("eval_12345")

if evaluation:
    # Fetch all results (handles pagination automatically)
    all_results = evaluation.get_all_results()
    
    print(f"Retrieved {len(all_results)} total results")
```


### `get_all_results_async(timeout=None)`

Asynchronously fetches all results for this evaluation by automatically handling pagination.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from atlas import AsyncAtlas
import asyncio

async def fetch_all_evaluation_results():
    client = AsyncAtlas()
    
    # Get an evaluation
    evaluation = await client.evaluations.get_by_id("eval_12345")
    
    if evaluation:
        # Fetch all results asynchronously (handles pagination automatically)
        all_results = await evaluation.get_all_results_async()
        
        print(f"Retrieved {len(all_results)} total results")
        
        return all_results
    
    return None

results = asyncio.run(fetch_all_evaluation_results())
```


## Response Objects

The `create`, `get_by_id` and `get_many` method returns an `Evaluation` objects with the following properties:

### Evaluation Object Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique evaluation identifier |
| `status` | `EvaluationStatus` | Current evaluation status (enum) |
| `submitted_at` | `int` | Unix timestamp when evaluation was submitted |
| `finished_at` | `int` | Unix timestamp when evaluation finished |
| `model_id` | `str` | ID of the model used in the evaluation |
| `benchmark_id` | `str` | ID of the benchmark used (aliased as "dataset_id" in API) |
| `average_duration` | `int` | Average response time in milliseconds |
| `accuracy` | `float` | Overall accuracy score (0.0 to 1.0) |


## Evaluation Status

The `status` field is an `EvaluationStatus` enum with the following values:

| Status | Description |
|--------|-------------|
| `"pending"` | Evaluation queued but not yet started |
| `"in-progress"` | Evaluation currently in progress |
| `"paused"` | Evaluation has been paused |
| `"success"` | Evaluation finished successfully |
| `"failure"` | Evaluation failed due to an error |


## Next Steps
- Explore [code examples](../examples/retrieving-results.md) for common analysis patterns
