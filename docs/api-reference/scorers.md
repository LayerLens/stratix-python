# Scorers

The `scorers` resource on the Stratix client allows you to create and manage custom scorers for evaluating benchmark results. Scorers use an LLM model to evaluate model outputs using a custom prompt.

## Overview

A scorer defines a custom evaluation criterion backed by a specific LLM model and a prompt template. Custom scorers can be attached to custom benchmarks to provide additional scoring beyond the built-in metrics.

### Using Synchronous Client

```python
from layerlens import Stratix

client = Stratix()

# Fetch a model to use for the scorer
models = client.models.get(type="public", name="gpt-4o")
model = models[0]

# Create a scorer
scorer = client.scorers.create(
    name="Helpfulness Scorer",
    description="Evaluates how helpful the response is",
    model_id=model.id,
    prompt="Rate the helpfulness of the following response on a scale of 0 to 1.",
)

if scorer:
    print(f"Created scorer: {scorer.name} (id={scorer.id})")

# List all scorers
response = client.scorers.get_many()
if response:
    for s in response.scorers:
        print(f"  {s.name}: {s.description}")
```

### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix

async def main():
    client = AsyncStratix()

    scorer = await client.scorers.create(
        name="Helpfulness Scorer",
        description="Evaluates how helpful the response is",
        model_id="model-abc123",
        prompt="Rate the helpfulness of the following response on a scale of 0 to 1.",
    )

    if scorer:
        print(f"Created scorer: {scorer.name}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Methods

Both the `Stratix` (synchronous) and `AsyncStratix` (asynchronous) clients support the following methods.

### `create(name, description, model_id, prompt, timeout=None)`

Creates a new custom scorer.

#### Parameters

| Parameter     | Type                             | Required | Description                                        |
| ------------- | -------------------------------- | -------- | -------------------------------------------------- |
| `name`        | `str`                            | Yes      | Display name for the scorer                        |
| `description` | `str`                            | Yes      | Description of what the scorer evaluates           |
| `model_id`    | `str`                            | Yes      | ID of the LLM model to use for scoring             |
| `prompt`      | `str`                            | Yes      | Prompt template used to evaluate model outputs     |
| `timeout`     | `float \| httpx.Timeout \| None` | No       | Override request timeout                           |

#### Returns

Returns a `Scorer` object if successful, `None` if the scorer could not be created.

### `get(id, timeout=None)`

Retrieves a scorer by its unique identifier.

#### Parameters

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `id`      | `str`                            | Yes      | The unique scorer ID     |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns a `Scorer` object if found, `None` otherwise.

### `get_many(page=None, page_size=None, timeout=None)`

Retrieves multiple scorers with pagination.

#### Parameters

| Parameter   | Type                             | Required | Description                                          |
| ----------- | -------------------------------- | -------- | ---------------------------------------------------- |
| `page`      | `int \| None`                    | No       | Page number (1-based, defaults to 1)                 |
| `page_size` | `int \| None`                    | No       | Number of scorers per page (default: 100, max: 500)  |
| `timeout`   | `float \| httpx.Timeout \| None` | No       | Override request timeout                             |

#### Returns

Returns a `ScorersResponse` object containing:

- `scorers`: List of `Scorer` objects
- `count`: Number of scorers in this page
- `total_count`: Total number of scorers

Returns `None` if the request fails.

### `update(id, name=None, description=None, model_id=None, prompt=None, timeout=None)`

Updates an existing scorer. Only provided fields are modified; omitted fields remain unchanged.

#### Parameters

| Parameter     | Type                             | Required | Description                            |
| ------------- | -------------------------------- | -------- | -------------------------------------- |
| `id`          | `str`                            | Yes      | The unique scorer ID                   |
| `name`        | `str \| None`                    | No       | Updated display name                   |
| `description` | `str \| None`                    | No       | Updated description                    |
| `model_id`    | `str \| None`                    | No       | Updated model ID                       |
| `prompt`      | `str \| None`                    | No       | Updated prompt template                |
| `timeout`     | `float \| httpx.Timeout \| None` | No       | Override request timeout               |

#### Returns

Returns `True` if the update succeeded, `False` otherwise.

### `delete(id, timeout=None)`

Deletes a scorer by its unique identifier.

#### Parameters

| Parameter | Type                             | Required | Description              |
| --------- | -------------------------------- | -------- | ------------------------ |
| `id`      | `str`                            | Yes      | The unique scorer ID     |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout |

#### Returns

Returns `True` if the scorer was deleted, `False` otherwise.

## Response Objects

### Scorer Object Properties

| Property          | Type          | Description                          |
| ----------------- | ------------- | ------------------------------------ |
| `id`              | `str`         | Unique scorer identifier             |
| `organization_id` | `str`         | Organization the scorer belongs to   |
| `project_id`      | `str`         | Project the scorer belongs to        |
| `name`            | `str`         | Display name                         |
| `description`     | `str \| None` | Description of what it evaluates     |
| `model_id`        | `str \| None` | ID of the backing LLM model         |
| `model_name`      | `str \| None` | Name of the backing LLM model       |
| `model_key`       | `str \| None` | Key of the backing LLM model        |
| `model_company`   | `str \| None` | Company that provides the model      |
| `prompt`          | `str \| None` | Prompt template for scoring          |
| `created_at`      | `str \| None` | ISO 8601 creation timestamp          |
| `updated_at`      | `str \| None` | ISO 8601 last update timestamp       |

## Next Steps

- Learn about [Benchmarks](models-benchmarks.md) to attach custom scorers to custom benchmarks
- Learn about [Judges](judges.md) for evaluating traces
