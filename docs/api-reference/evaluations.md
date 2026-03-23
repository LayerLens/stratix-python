# Evaluations

The `evaluations` resource on the Stratix client allows you to create and manage evaluations against various benchmarks for your organization on the Stratix platform. This is one of the core functionalities of the Stratix platform.

## Overview

An evaluation runs a specified model against a benchmark dataset and returns comprehensive metrics.

The below example trigger evaluations using `gpt-4o` against `simpleQA`.

> Before running the below examples ensure the model and benchmark being run are present on your organiztion.

### Using Synchronous Client

Below is an example showing how to trigger an evaluation, waiting for it to complete and finally fetching the evaluations results.

```python
from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

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

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

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

Both the `Stratix` (synchronous) and `AsyncStratix` (asynchronous) clients support the following methods.

### `create(model, benchmark, timeout=None)`

Creates a new evaluation for the specified model and benchmark.

#### Parameters

| Parameter   | Type                             | Required | Description               |
| ----------- | -------------------------------- | -------- | ------------------------- |
| `model`     | `Model`                          | Yes      | The model to evaluate     |
| `benchmark` | `Benchmark`                      | Yes      | The benchmark to evaluate |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout  |

#### Returns

Returns an `Evaluation` object if successful, `None` if the evaluation could not be created.

### `wait_for_completion(evaluation, interval_seconds=30, timeout_seconds=None)`

Polls an evaluation until it completes (success, failure, or timeout) or the specified timeout is reached.

#### Parameters

| Parameter          | Type          | Required | Description                                        |
| ------------------ | ------------- | -------- | -------------------------------------------------- |
| `evaluation`       | `Evaluation`  | Yes      | The evaluation object to monitor                   |
| `interval_seconds` | `int`         | No       | Polling interval in seconds (default: 30)          |
| `timeout_seconds`  | `int \| None` | No       | Maximum time to wait in seconds (no limit if None) |

#### Returns

Returns the updated `Evaluation` object when completed, or `None` if polling fails.

### `get_by_id(evaluation_id, timeout=None)`

Retrieves an existing evaluation by its unique identifier.

#### Parameters

| Parameter       | Type                             | Required | Description                      |
| --------------- | -------------------------------- | -------- | -------------------------------- |
| `evaluation_id` | `str`                            | Yes      | The unique evaluation identifier |
| `timeout`       | `float \| httpx.Timeout \| None` | No       | Override request timeout         |

#### Returns

Returns an `Evaluation` object if found, `None` if the evaluation does not exist or cannot be accessed.

#### Example

```python
from layerlens import Stratix

client = Stratix()

# Retrieve an evaluation by ID
evaluation_id = "eval_abc123xyz"
evaluation = client.evaluations.get_by_id(evaluation_id)
```

#### Async Usage

```python
from layerlens import AsyncStratix
import asyncio

async def get_evaluation():
    client = AsyncStratix()

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

### `get_many(page=None, page_size=None, sort_by=None, order=None, model_ids=None, benchmark_ids=None, status=None, unique=False, timeout=None)`

Retrieves multiple evaluations with optional pagination, sorting, and filtering.

#### Parameters

| Parameter       | Type                             | Required | Description                                                                         |
| --------------- | -------------------------------- | -------- | ----------------------------------------------------------------------------------- |
| `page`          | `int \| None`                    | No       | Page number for pagination (1-based, defaults to 1)                                 |
| `page_size`     | `int \| None`                    | No       | Number of evaluations per page (default: 100, max: 500)                             |
| `sort_by`       | `str \| None`                    | No       | Sort by field: `submitted_at`, `accuracy`, or `average_duration`                    |
| `order`         | `str \| None`                    | No       | Sort order: `asc` or `desc`                                                         |
| `model_ids`     | `List[str] \| None`              | No       | Filter by model IDs                                                                 |
| `benchmark_ids` | `List[str] \| None`              | No       | Filter by benchmark/dataset IDs                                                     |
| `status`        | `EvaluationStatus \| None`       | No       | Filter by evaluation status                                                         |
| `unique`        | `bool`                           | No       | If `True`, deduplicate by model+benchmark pair, keeping only the latest evaluation  |
| `timeout`       | `float \| httpx.Timeout \| None` | No       | Override request timeout                                                            |

#### Returns

Returns an `EvaluationsResponse` object containing:

- `evaluations`: List of `Evaluation` objects
- `pagination`: Pagination metadata with `page`, `page_size`, `total_pages`, and `total_count`

Returns `None` if the request fails.

#### Example

```python
from layerlens import Stratix
from layerlens.models import EvaluationStatus

client = Stratix()

# Get top evaluations by accuracy
response = client.evaluations.get_many(
    sort_by="accuracy",
    order="desc",
    status=EvaluationStatus.SUCCESS,
    page_size=10,
)

if response:
    for evaluation in response.evaluations:
        print(f"{evaluation.id}: accuracy={evaluation.accuracy:.2f}%")

# Get only the latest evaluation per model+benchmark pair
response = client.evaluations.get_many(
    unique=True,
    sort_by="accuracy",
    order="desc",
)
```

### `get_results(page=None, page_size=None, timeout=None)`

Fetches results for this evaluation with pagination support. This is a synchronous method.

#### Parameters

| Parameter   | Type                             | Required | Description                                         |
| ----------- | -------------------------------- | -------- | --------------------------------------------------- |
| `page`      | `int \| None`                    | No       | Page number for pagination (1-based, defaults to 1) |
| `page_size` | `int \| None`                    | No       | Number of results per page (default: 100, max: 500) |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout                            |

#### Returns

Returns a `ResultsResponse` object containing results and pagination metadata, or `None` if the request fails.

#### Example

```python
from layerlens import Stratix

client = Stratix()

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

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from layerlens import Stratix

client = Stratix()

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

| Parameter   | Type                             | Required | Description                                         |
| ----------- | -------------------------------- | -------- | --------------------------------------------------- |
| `page`      | `int \| None`                    | No       | Page number for pagination (1-based, defaults to 1) |
| `page_size` | `int \| None`                    | No       | Number of results per page (default: 100, max: 500) |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout                            |

#### Returns

Returns a `ResultsResponse` object containing results and pagination metadata, or `None` if the request fails.

#### Example

```python
from layerlens import AsyncStratix
import asyncio

async def fetch_evaluation_results():
    client = AsyncStratix()

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

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from layerlens import Stratix

client = Stratix()

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

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a list of `Result` objects containing all results for the evaluation.

#### Example

```python
from layerlens import AsyncStratix
import asyncio

async def fetch_all_evaluation_results():
    client = AsyncStratix()

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

| Property             | Type                          | Description                                               |
| -------------------- | ----------------------------- | --------------------------------------------------------- |
| `id`                 | `str`                         | Unique evaluation identifier                              |
| `status`             | `EvaluationStatus`            | Current evaluation status (enum)                          |
| `status_description` | `str`                         | Human-readable status description (default: `""`)         |
| `submitted_at`       | `int`                         | Unix timestamp when evaluation was submitted              |
| `finished_at`        | `int`                         | Unix timestamp when evaluation finished                   |
| `model_id`           | `str`                         | ID of the model used in the evaluation                    |
| `model_name`         | `str`                         | Name of the model (default: `""`)                         |
| `model_key`          | `str`                         | Key identifier of the model (default: `""`)               |
| `model_company`      | `str`                         | Company/provider of the model (default: `""`)             |
| `benchmark_id`       | `str`                         | ID of the benchmark used (aliased as "dataset_id" in API) |
| `benchmark_name`     | `str`                         | Name of the benchmark (aliased as "dataset_name" in API, default: `""`) |
| `average_duration`   | `int`                         | Average response time in milliseconds                     |
| `accuracy`           | `float`                       | Overall accuracy score (0.0 to 1.0)                       |
| `readability_score`  | `float`                       | Readability score (default: `0.0`)                        |
| `toxicity_score`     | `float`                       | Toxicity score (default: `0.0`)                           |
| `ethics_score`       | `float`                       | Ethics score (default: `0.0`)                             |
| `failed_prompt_count`| `int`                         | Number of failed prompts (default: `0`)                   |
| `queue_id`           | `int`                         | Queue identifier (default: `0`)                           |
| `summary`            | `EvaluationSummary \| None`   | Rich evaluation summary (see below, default: `None`)      |

### EvaluationSummary Object

The `summary` field contains a rich analysis of the evaluation when available.

| Property              | Type                            | Description                              |
| --------------------- | ------------------------------- | ---------------------------------------- |
| `name`                | `str`                           | Summary title                            |
| `goal`                | `str`                           | Goal of the evaluation                   |
| `metrics`             | `List[EvaluationMetric]`        | Metrics used (each has `name`, `description`) |
| `task_types`          | `List[EvaluationTaskType]`      | Task types (each has `name`, `description`)   |
| `dataset`             | `EvaluationDataset \| None`     | Dataset info (`total_size`, `training_size`, `test_size`, `characteristics`) |
| `model`               | `EvaluationModelInfo \| None`   | Model info (`model_name`, `performance`)  |
| `performance_details` | `PerformanceDetails \| None`    | Strengths and challenges lists            |
| `error_analysis`      | `ErrorAnalysis \| None`         | Common failure modes and example          |
| `analysis_summary`    | `AnalysisSummary \| None`       | Key takeaways list                        |

#### Evaluation Status

The `status` field is an `EvaluationStatus` enum with the following values:

| Status          | Description                           |
| --------------- | ------------------------------------- |
| `"pending"`     | Evaluation queued but not yet started |
| `"in-progress"` | Evaluation currently in progress      |
| `"paused"`      | Evaluation has been paused            |
| `"success"`     | Evaluation finished successfully      |
| `"failure"`     | Evaluation failed due to an error     |
| `"timeout"`     | Evaluation timed out                  |
| `"cancelled"`   | Evaluation was cancelled              |

## Next Steps

- Explore [code examples](../examples/retrieving-results.md) for common analysis patterns
