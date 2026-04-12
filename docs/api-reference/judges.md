# Judges

The `judges` resource on the Stratix client allows you to create and manage LLM-based judges for evaluating traces. Judges define evaluation criteria that can be run against traces to assess quality, correctness, and other properties.

## Overview

A judge encapsulates an evaluation goal and is backed by a specific LLM model. Once created, judges can be run against traces via the trace evaluations resource to produce scored results.

### Using Synchronous Client

```python
from layerlens import Stratix

client = Stratix()

# Fetch a model to use for the judge
models = client.models.get(type="public", name="gpt-4o")
model = models[0]

# Create a judge
judge = client.judges.create(
    name="Code Quality Judge",
    evaluation_goal="Evaluate the quality of code output including correctness and style",
    model_id=model.id,
)

print(f"Created judge: {judge.name} (v{judge.version})")

# List all judges
response = client.judges.get_many()
for j in response.judges:
    print(f"  {j.name}: {j.run_count} runs")
```

### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix

async def main():
    client = AsyncStratix()

    judge = await client.judges.create(
        name="Code Quality Judge",
        evaluation_goal="Evaluate the quality of code output including correctness and style",
        model_id="model-abc123",
    )

    print(f"Created judge: {judge.name} (v{judge.version})")

if __name__ == "__main__":
    asyncio.run(main())
```

## Methods

Both the `Stratix` (synchronous) and `AsyncStratix` (asynchronous) clients support the following methods.

### `create(name, evaluation_goal, model_id=None, timeout=None)`

Creates a new judge with the specified evaluation criteria.

#### Parameters

| Parameter         | Type                             | Required | Description                                  |
| ----------------- | -------------------------------- | -------- | -------------------------------------------- |
| `name`            | `str`                            | Yes      | Display name for the judge                   |
| `evaluation_goal` | `str`                            | Yes      | Description of what the judge should evaluate |
| `model_id`        | `str \| None`                    | No       | ID of the LLM model to use. If omitted, the server uses a default model |
| `timeout`         | `float \| httpx.Timeout \| None` | No       | Override request timeout                     |

#### Returns

Returns a `Judge` object if successful, `None` if the judge could not be created.

#### Example

```python
judge = client.judges.create(
    name="Accuracy Judge",
    evaluation_goal="Evaluate whether the response is factually accurate",
    model_id="model-abc123",
)
```

### `get(id, timeout=None)`

Retrieves a judge by its unique identifier.

#### Parameters

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `id`      | `str`                            | Yes      | The unique judge ID      |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a `Judge` object if found, `None` otherwise.

### `get_many(page=None, page_size=None, timeout=None)`

Retrieves multiple judges with pagination.

#### Parameters

| Parameter   | Type                             | Required | Description                                        |
| ----------- | -------------------------------- | -------- | -------------------------------------------------- |
| `page`      | `int \| None`                    | No       | Page number (1-based, defaults to 1)               |
| `page_size` | `int \| None`                    | No       | Number of judges per page (default: 100, max: 500) |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout                           |

#### Returns

Returns a `JudgesResponse` object containing:

- `judges`: List of `Judge` objects
- `count`: Number of judges in this page
- `total_count`: Total number of judges

Returns `None` if the request fails.

#### Example

```python
# Get first page
response = client.judges.get_many()
print(f"Total judges: {response.total_count}")

# Get specific page
response = client.judges.get_many(page=2, page_size=50)
```

### `update(id, name=None, evaluation_goal=None, model_id=None, timeout=None)`

Updates an existing judge. Only provided fields are modified; omitted fields remain unchanged. Updating a judge creates a new version.

#### Parameters

| Parameter         | Type                             | Required | Description                                  |
| ----------------- | -------------------------------- | -------- | -------------------------------------------- |
| `id`              | `str`                            | Yes      | The unique judge ID                          |
| `name`            | `str \| None`                    | No       | Updated display name                         |
| `evaluation_goal` | `str \| None`                    | No       | Updated evaluation criteria                  |
| `model_id`        | `str \| None`                    | No       | Updated model ID                             |
| `timeout`         | `float \| httpx.Timeout \| None` | No       | Override request timeout                     |

#### Returns

Returns an `UpdateJudgeResponse` if successful, `None` otherwise.

#### Example

```python
updated = client.judges.update(
    "judge-123",
    evaluation_goal="Evaluate code for correctness, readability, and security",
)
```

### `delete(id, timeout=None)`

Deletes a judge by its unique identifier.

#### Parameters

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `id`      | `str`                            | Yes      | The unique judge ID      |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a `DeleteJudgeResponse` if successful, `None` otherwise.

## Response Objects

### Judge Object Properties

| Property          | Type                  | Description                              |
| ----------------- | --------------------- | ---------------------------------------- |
| `id`              | `str`                 | Unique judge identifier                  |
| `organization_id` | `str`                 | Organization the judge belongs to        |
| `project_id`      | `str`                 | Project the judge belongs to             |
| `name`            | `str`                 | Display name                             |
| `evaluation_goal` | `str`                 | Description of evaluation criteria       |
| `model_id`        | `str \| None`         | ID of the backing LLM model             |
| `model_name`      | `str \| None`         | Name of the backing LLM model           |
| `model_company`   | `str \| None`         | Company that provides the model         |
| `version`         | `int`                 | Current version number                   |
| `run_count`       | `int`                 | Number of times this judge has been run  |
| `created_at`      | `str`                 | ISO 8601 creation timestamp              |
| `updated_at`      | `str \| None`         | ISO 8601 last update timestamp           |
| `versions`        | `List[JudgeVersion]`  | Version history                          |

### JudgeVersion Object Properties

| Property          | Type          | Description                        |
| ----------------- | ------------- | ---------------------------------- |
| `version`         | `int`         | Version number                     |
| `name`            | `str`         | Name at this version               |
| `evaluation_goal` | `str`         | Evaluation goal at this version    |
| `model_id`        | `str \| None` | Model ID at this version           |
| `model_name`      | `str \| None` | Model name at this version         |
| `model_company`   | `str \| None` | Model company at this version      |
| `updated_at`      | `str \| None` | When this version was created      |
| `updated_by`      | `str \| None` | Who created this version           |

## Next Steps

- Learn about [Traces](traces.md) to upload data for evaluation
- Learn about [Trace Evaluations](trace-evaluations.md) to run judges against traces
