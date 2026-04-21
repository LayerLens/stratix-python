# Trace Evaluations

The `trace_evaluations` resource on the Stratix client allows you to run judges against traces and retrieve scored results. This is how you assess the quality of your trace data using the evaluation criteria defined in your judges.

## Overview

A trace evaluation runs a specific judge against a specific trace, producing a scored result with reasoning and step-by-step analysis. You can estimate costs before running evaluations and retrieve detailed results afterwards.

### Using Synchronous Client

```python
from layerlens import Stratix

client = Stratix()

# Estimate cost before running
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=["trace-1", "trace-2"],
    judge_id="judge-123",
)
print(f"Estimated cost: ${estimate.estimated_cost:.4f}")

# Run a judge on a trace
evaluation = client.trace_evaluations.create(
    trace_id="trace-1",
    judge_id="judge-123",
)

# Wait for completion and get results
result = client.trace_evaluations.wait_for_completion(evaluation.id)
if result:
    print(f"Score: {result.score}, Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
```

### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix

async def main():
    client = AsyncStratix()

    evaluation = await client.trace_evaluations.create(
        trace_id="trace-1",
        judge_id="judge-123",
    )

    result = await client.trace_evaluations.wait_for_completion(evaluation.id)
    if result:
        print(f"Score: {result.score}, Passed: {result.passed}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Methods

Both the `Stratix` (synchronous) and `AsyncStratix` (asynchronous) clients support the following methods.

### `create(trace_id, judge_id, timeout=None)`

Runs a judge against a trace, creating a new trace evaluation.

#### Parameters

| Parameter  | Type                             | Required | Description                    |
| ---------- | -------------------------------- | -------- | ------------------------------ |
| `trace_id` | `str`                            | Yes      | ID of the trace to evaluate    |
| `judge_id` | `str`                            | Yes      | ID of the judge to run         |
| `timeout`  | `float \| httpx.Timeout \| None` | No       | Override request timeout       |

#### Returns

Returns a `TraceEvaluation` object if successful, `None` otherwise.

#### Example

```python
evaluation = client.trace_evaluations.create(
    trace_id="trace-abc",
    judge_id="judge-xyz",
)
print(f"Evaluation {evaluation.id}: {evaluation.status}")
```

### `get(id, timeout=None)`

Retrieves a trace evaluation by its unique identifier.

#### Parameters

| Parameter | Type                             | Required | Description                        |
| --------- | -------------------------------- | -------- | ---------------------------------- |
| `id`      | `str`                            | Yes      | The unique trace evaluation ID     |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout           |

#### Returns

Returns a `TraceEvaluation` object if found, `None` otherwise.

### `get_many(page=None, page_size=None, judge_id=None, trace_id=None, outcome=None, time_range=None, search=None, sort_by=None, sort_order=None, timeout=None)`

Retrieves multiple trace evaluations with filtering and pagination.

#### Parameters

| Parameter    | Type                             | Required | Description                                                |
| ------------ | -------------------------------- | -------- | ---------------------------------------------------------- |
| `page`       | `int \| None`                    | No       | Page number (1-based, defaults to 1)                       |
| `page_size`  | `int \| None`                    | No       | Number of evaluations per page (default: 20, max: 100)     |
| `judge_id`   | `str \| None`                    | No       | Filter by judge                                            |
| `trace_id`   | `str \| None`                    | No       | Filter by trace                                            |
| `outcome`    | `str \| None`                    | No       | Filter by outcome (e.g., "pass", "fail")                   |
| `time_range` | `str \| None`                    | No       | Filter by time range (e.g., "7d", "30d")                   |
| `search`     | `str \| None`                    | No       | Search term to filter evaluations                          |
| `sort_by`    | `str \| None`                    | No       | Field to sort by (e.g., "created_at")                      |
| `sort_order` | `str \| None`                    | No       | Sort direction: "asc" or "desc"                            |
| `timeout`    | `float \| httpx.Timeout \| None` | No       | Override request timeout                                   |

#### Returns

Returns a `TraceEvaluationsResponse` object containing:

- `trace_evaluations`: List of `TraceEvaluation` objects
- `count`: Number of evaluations in this page
- `total`: Total number of matching evaluations

Returns `None` if the request fails.

#### Example

```python
# Get all evaluations
response = client.trace_evaluations.get_many()
print(f"Total: {response.total}")

# Filtered by judge and outcome
response = client.trace_evaluations.get_many(
    judge_id="judge-123",
    outcome="pass",
    sort_by="created_at",
    sort_order="desc",
)
```

### `get_results(id, timeout=None)`

Retrieves the detailed results of a completed trace evaluation, including scores, reasoning, and step-by-step analysis.

Returns `None` if results are not yet available (evaluation still pending or in progress).

#### Parameters

| Parameter | Type                             | Required | Description                        |
| --------- | -------------------------------- | -------- | ---------------------------------- |
| `id`      | `str`                            | Yes      | The unique trace evaluation ID     |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout           |

#### Returns

Returns a `TraceEvaluationResultsResponse` object with the evaluation result fields (score, passed, reasoning, etc.).

Returns `None` if the evaluation has not completed yet or if the request fails.

#### Example

```python
result = client.trace_evaluations.get_results("eval-123")
if result:
    print(f"Score: {result.score}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
    for step in result.steps:
        print(f"  Tool: {step.tool}, Result: {step.result}")
```

### `wait_for_completion(id, interval_seconds=3, timeout_seconds=300)`

Polls the evaluation status until it reaches a terminal state (success or failure), then returns the results. This is the recommended way to wait for trace evaluation results.

#### Parameters

| Parameter          | Type           | Required | Default | Description                                      |
| ------------------ | -------------- | -------- | ------- | ------------------------------------------------ |
| `id`               | `str`          | Yes      |         | The unique trace evaluation ID                   |
| `interval_seconds` | `int`          | No       | `3`     | Seconds between status polls                     |
| `timeout_seconds`  | `int \| None`  | No       | `300`   | Maximum wait time. `None` waits indefinitely     |

#### Returns

Returns a `TraceEvaluationResultsResponse` object if the evaluation completes successfully.

Returns `None` if the evaluation failed or no results are available.

Raises `TimeoutError` if `timeout_seconds` is exceeded.

#### Example

```python
evaluation = client.trace_evaluations.create(
    trace_id="trace-abc",
    judge_id="judge-xyz",
)

# Wait up to 5 minutes for results
result = client.trace_evaluations.wait_for_completion(evaluation.id)
if result:
    print(f"Score: {result.score}, Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")

# Custom timeout and polling interval
result = client.trace_evaluations.wait_for_completion(
    evaluation.id,
    interval_seconds=5,
    timeout_seconds=600,
)
```

### `estimate_cost(trace_ids, judge_id, timeout=None)`

Estimates the cost of running a judge against a set of traces before actually executing the evaluations.

#### Parameters

| Parameter   | Type                             | Required | Description                         |
| ----------- | -------------------------------- | -------- | ----------------------------------- |
| `trace_ids` | `List[str]`                      | Yes      | List of trace IDs to evaluate       |
| `judge_id`  | `str`                            | Yes      | ID of the judge to run              |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout            |

#### Returns

Returns a `CostEstimateResponse` object if successful, `None` otherwise.

#### Example

```python
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=["trace-1", "trace-2", "trace-3"],
    judge_id="judge-123",
)
if estimate:
    print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
    print(f"Traces: {estimate.trace_count}")
    print(f"Model: {estimate.model}")
```

## Response Objects

### TraceEvaluation Object Properties

| Property         | Type                         | Description                              |
| ---------------- | ---------------------------- | ---------------------------------------- |
| `id`             | `str`                        | Unique evaluation identifier             |
| `trace_id`       | `str`                        | ID of the evaluated trace                |
| `judge_id`       | `str`                        | ID of the judge used                     |
| `status`         | `TraceEvaluationStatus`      | Current status of the evaluation         |
| `judge_snapshot` | `JudgeSnapshot \| None`      | Snapshot of judge config at run time     |
| `created_at`     | `str \| None`                | ISO 8601 creation timestamp              |
| `started_at`     | `str \| None`                | When evaluation started                  |
| `finished_at`    | `str \| None`                | When evaluation finished                 |

#### TraceEvaluationStatus

| Status          | Description                          |
| --------------- | ------------------------------------ |
| `"pending"`     | Evaluation queued but not started    |
| `"in_progress"` | Evaluation currently running         |
| `"success"`     | Evaluation completed successfully    |
| `"failure"`     | Evaluation failed                    |

### TraceEvaluationResult Object Properties

| Property              | Type                          | Description                               |
| --------------------- | ----------------------------- | ----------------------------------------- |
| `id`                  | `str`                         | Unique result identifier                  |
| `trace_evaluation_id` | `str`                         | Parent evaluation ID                      |
| `trace_id`            | `str`                         | ID of the evaluated trace                 |
| `judge_id`            | `str`                         | ID of the judge used                      |
| `score`               | `float \| None`               | Numerical score                           |
| `passed`              | `bool \| None`                | Whether the trace passed the evaluation   |
| `reasoning`           | `str \| None`                 | Overall reasoning for the score           |
| `steps`               | `List[TraceEvaluationStep]`   | Step-by-step reasoning                    |
| `model`               | `str \| None`                 | Model used for evaluation                 |
| `turns`               | `int \| None`                 | Number of turns in evaluation             |
| `latency_ms`          | `int \| None`                 | Evaluation latency in milliseconds        |
| `prompt_tokens`       | `int \| None`                 | Number of prompt tokens used              |
| `completion_tokens`   | `int \| None`                 | Number of completion tokens used          |
| `total_cost`          | `float \| None`               | Total cost of the evaluation              |
| `created_at`          | `str \| None`                 | ISO 8601 creation timestamp               |

### CostEstimateResponse Properties

| Property         | Type    | Description                        |
| ---------------- | ------- | ---------------------------------- |
| `estimated_cost` | `float` | Estimated cost in dollars          |
| `input_tokens`   | `int`   | Estimated input tokens             |
| `output_tokens`  | `int`   | Estimated output tokens            |
| `model`          | `str`   | Model that would be used           |
| `trace_count`    | `int`   | Number of traces to evaluate       |

## Next Steps

- Learn about [Judges](judges.md) to create evaluation criteria
- Learn about [Traces](traces.md) to upload data for evaluation
