# Models & Benchmarks

This page provides reference information about accessing the available models and benchmarks on your organization using the layerlens python sdk.

## Overview

Running evaluations on the Stratix platform require a model and benchmark to be selected. The availability of models and benchmarks depends on your organizations configuration.

> Before running the below examples ensure the model and benchmark being run are present on your organiztion.

### Finding Available Models and Benchmarks

#### 1. Using the Stratix Dashboard

The most reliable way to find available models and benchmarks:

1. Log into your Stratix Dashboard.
2. Navigate to the evaluation creation page.
3. View dropdown lists of available models and benchmarks.

#### 2. Using the python sdk

```python
from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Models
models = client.models.get()

# --- Benchmarks
benchmarks = client.benchmarks.get()
```

## Models

### `get(type=None, name=None, key=None, categories=None, companies=None, regions=None, licenses=None, timeout=None)`

Retrieves a list of available models with optional filtering parameters. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter    | Type                                  | Required | Description                                                                                    |
| ------------ | ------------------------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `type`       | `Literal["custom", "public"] \| None` | No       | Filter by model type. If `None`, returns both custom and public models                         |
| `name`       | `str \| None`                         | No       | Filter models by name (partial match search)                                                   |
| `key`        | `str \| None`                         | No       | Filter models by key (partial match search)                                                    |
| `categories` | `List[str] \| None`                   | No       | Filter by categories: `Transformer`, `MoE`, `Open-Source`, `Closed-Source`                     |
| `companies`  | `List[str] \| None`                   | No       | Filter by model companies/providers                                                            |
| `regions`    | `List[str] \| None`                   | No       | Filter by supported regions                                                                    |
| `licenses`   | `List[str] \| None`                   | No       | Filter by license types                                                                        |
| `timeout`    | `float \| httpx.Timeout \| None`      | No       | Override request timeout                                                                       |

> **Note:** When filtering by `categories`, `companies`, `regions`, or `licenses`, only public models are returned since custom models do not have these fields.

#### Returns

Returns an `Optional[List[Model]]` - a list of `Model` objects that match the filter criteria. Returns an empty list `[]` if no models match the criteria, or `None` if there's an error.

#### Model Object Properties

Each `Model` object in the returned list contains:

| Property      | Type  | Description                                             |
| ------------- | ----- | ------------------------------------------------------- |
| `id`          | `str` | Unique model identifier for use in evaluations          |
| `name`        | `str` | Human-readable model name                               |
| `key`         | `str` | Unique model key/identifier that is similar to the name |
| `description` | `str` | Text description of the model                           |

### `get_by_id(id, timeout=None)`

Retrieves a specific model by its unique identifier. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter | Type                             | Required | Description                  |
| --------- | -------------------------------- | -------- | ---------------------------- |
| `id`      | `str`                            | Yes      | Unique model identifier      |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout     |

#### Returns

Returns an `Optional[Model]` - a single `Model` object if found, or `None` if the model doesn't exist or there's an error.

### `get_by_key(key, timeout=None)`

Retrieves a specific model by its unique key. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter | Type                             | Required | Description                  |
| --------- | -------------------------------- | -------- | ---------------------------- |
| `key`     | `str`                            | Yes      | Unique model key identifier  |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout     |

#### Returns

Returns an `Optional[Model]` - a single `Model` object if found, or `None` if the model doesn't exist or there's an error.

### `add(*model_ids, timeout=None)`

Adds models (public or custom) to the project by their IDs.

#### Parameters

| Parameter    | Type                             | Required | Description                    |
| ------------ | -------------------------------- | -------- | ------------------------------ |
| `*model_ids` | `str`                            | Yes      | One or more model IDs to add   |
| `timeout`    | `float \| httpx.Timeout \| None` | No       | Override request timeout       |

#### Returns

Returns `bool` - `True` if the operation succeeded, `False` otherwise.

#### Example

```python
client = Stratix()
success = client.models.add("model-id-1", "model-id-2")
```

### `remove(*model_ids, timeout=None)`

Removes models (public or custom) from the project's model list. The underlying records are not deleted — use `delete_custom` to fully tear down a custom model.

#### Parameters

| Parameter    | Type                             | Required | Description                       |
| ------------ | -------------------------------- | -------- | --------------------------------- |
| `*model_ids` | `str`                            | Yes      | One or more model IDs to remove   |
| `timeout`    | `float \| httpx.Timeout \| None` | No       | Override request timeout          |

#### Returns

Returns `bool` - `True` if the operation succeeded, `False` otherwise.

#### Example

```python
client = Stratix()
success = client.models.remove("model-id-1", "model-id-2")
```

### `create_custom(name, key, description, api_url, max_tokens, api_key=None, extra_payload=None, timeout=None)`

Creates a custom model backed by an OpenAI-compatible API endpoint. This allows you to evaluate any model accessible via a chat completions endpoint.

#### Parameters

| Parameter       | Type                              | Required | Description                                                                       |
| --------------- | --------------------------------- | -------- | --------------------------------------------------------------------------------- |
| `name`          | `str`                             | Yes      | Model name (max 256 characters)                                                   |
| `key`           | `str`                             | Yes      | Unique model key, lowercase alphanumeric with dots/hyphens/slashes (max 256 chars)|
| `description`   | `str`                             | Yes      | Model description (max 500 characters)                                            |
| `api_url`       | `str`                             | Yes      | Full URL of the OpenAI-compatible chat completions endpoint                       |
| `max_tokens`    | `int`                             | Yes      | Maximum number of tokens the model supports                                       |
| `api_key`       | `str \| None`                     | No       | API key for the model provider                                                    |
| `extra_payload` | `Dict[str, Any] \| None`          | No       | JSON object merged into every outgoing chat-completions request body (see below)  |
| `timeout`       | `float \| httpx.Timeout \| None`  | No       | Override request timeout                                                          |

#### `extra_payload` semantics

When set, the keys/values in `extra_payload` are deep-merged into every outgoing request body. Customer values **win on conflict** with our hardcoded defaults — use it to override `temperature` (we send `0` for reproducible evaluations) or to add provider-specific fields like `top_p`, `presence_penalty`, or `max_completion_tokens` (required by some OpenAI reasoning models that reject `max_tokens`).

The keys `messages`, `model`, and `stream` are reserved and will be rejected.

#### Returns

Returns an `Optional[CreateModelResponse]` containing:

- `organization_id`: Organization identifier
- `project_id`: Project identifier
- `model_id`: The newly created model's identifier

Returns `None` if the request fails.

#### Example

```python
client = Stratix()

result = client.models.create_custom(
    name="My Custom Model",
    key="my-org/custom-model-v1",
    description="Custom fine-tuned model served via vLLM",
    api_url="https://my-model-endpoint.example.com/v1/chat/completions",
    api_key="my-provider-api-key",
    max_tokens=4096,
    # Optional — provider-specific overrides merged into every request body.
    extra_payload={"top_p": 0.9},
)

if result:
    print(f"Created model: {result.model_id}")
```

### `update_custom(model_id, *, api_url=None, api_key=None, max_tokens=None, extra_payload=None, timeout=None)`

Updates a custom model's mutable fields. At least one of `api_url`, `api_key`, `max_tokens`, or `extra_payload` must be provided. Primary use case: repointing `api_url` for ephemeral vLLM endpoints behind cloudflared tunnels whose URL changes between sessions.

#### Parameters

| Parameter       | Type                              | Required | Description                                                                      |
| --------------- | --------------------------------- | -------- | -------------------------------------------------------------------------------- |
| `model_id`      | `str`                             | Yes      | ID of the custom model to update                                                 |
| `api_url`       | `str \| None`                     | No       | New full URL of the OpenAI-compatible chat completions endpoint                  |
| `api_key`       | `str \| None`                     | No       | New API key for the model provider                                               |
| `max_tokens`    | `int \| None`                     | No       | New maximum tokens value                                                         |
| `extra_payload` | `Dict[str, Any] \| None`          | No       | New JSON object merged into every outgoing request. Pass `{}` to clear it.       |
| `timeout`       | `float \| httpx.Timeout \| None`  | No       | Override request timeout                                                         |

#### Returns

Returns `bool` — `True` on success, `False` otherwise.

#### Example

```python
client = Stratix()

# Repoint the api_url without re-creating the model
client.models.update_custom(
    "model-id-from-create-custom",
    api_url="https://my-new-endpoint.example.com/v1/chat/completions",
)

# Override request parameters for a model that doesn't accept temperature=0
client.models.update_custom(
    "model-id-from-create-custom",
    extra_payload={"temperature": 1},
)
```

### `delete_custom(model_id, *, timeout=None)`

Disables a custom model and removes it from `Project.Models`. The backend tears down the model's S3 yaml artifacts and AWS secret, and marks the record as disabled (preserving any evaluation references). Public models cannot be deleted via the SDK.

#### Parameters

| Parameter  | Type                             | Required | Description                       |
| ---------- | -------------------------------- | -------- | --------------------------------- |
| `model_id` | `str`                            | Yes      | ID of the custom model to delete  |
| `timeout`  | `float \| httpx.Timeout \| None` | No       | Override request timeout          |

#### Returns

Returns `bool` — `True` on success, `False` otherwise.

#### Example

```python
client = Stratix()
client.models.delete_custom("model-id-from-create-custom")
```

## Benchmarks

### `get(type=None, name=None, key=None, categories=None, languages=None, timeout=None)`

Retrieves a list of available benchmarks with optional filtering parameters. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter    | Type                                  | Required | Description                                                                    |
| ------------ | ------------------------------------- | -------- | ------------------------------------------------------------------------------ |
| `type`       | `Literal["custom", "public"] \| None` | No       | Filter by benchmark type. If `None`, returns both custom and public benchmarks |
| `name`       | `str \| None`                         | No       | Filter benchmarks by name (partial match search)                               |
| `key`        | `str \| None`                         | No       | Filter benchmarks by key (partial match search)                                |
| `categories` | `List[str] \| None`                   | No       | Filter by categories (e.g., `reasoning`, `knowledge`, `coding`)                |
| `languages`  | `List[str] \| None`                   | No       | Filter by language (e.g., `english`, `french`)                                 |
| `timeout`    | `float \| httpx.Timeout \| None`      | No       | Override request timeout                                                       |

> **Note:** When filtering by `categories` or `languages`, only public benchmarks are returned since custom benchmarks do not have these fields.

#### Returns

Returns `Optional[List[Benchmark]]` - a list of `Benchmark` objects that match the filter criteria. Returns an empty list `[]` if no benchmarks match the criteria, or `None` if there's an error.

#### Benchmark Object Properties

Each `Benchmark` object in the returned list contains:

| Property | Type  | Description                                                 |
| -------- | ----- | ----------------------------------------------------------- |
| `id`     | `str` | Unique benchmark identifier for use in evaluations          |
| `key`    | `str` | Unique benchmark key/identifier that is similar to the name |
| `name`   | `str` | Human-readable benchmark name                               |

### `get_by_id(id, timeout=None)`

Retrieves a specific benchmark by its unique identifier. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter | Type                             | Required | Description                     |
| --------- | -------------------------------- | -------- | ------------------------------- |
| `id`      | `str`                            | Yes      | Unique benchmark identifier     |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout        |

#### Returns

Returns an `Optional[Benchmark]` - a single `Benchmark` object if found, or `None` if the benchmark doesn't exist or there's an error.

### `get_by_key(key, timeout=None)`

Retrieves a specific benchmark by its unique key. Both the `Stratix` and `AsyncStratix` clients have this method.

#### Parameters

| Parameter | Type                             | Required | Description                        |
| --------- | -------------------------------- | -------- | ---------------------------------- |
| `key`     | `str`                            | Yes      | Unique benchmark key identifier    |
| `timeout` | `float \| httpx.Timeout \| None` | No       | Override request timeout           |

#### Returns

Returns an `Optional[Benchmark]` - a single `Benchmark` object if found, or `None` if the benchmark doesn't exist or there's an error.

### `add(*benchmark_ids, timeout=None)`

Adds benchmarks to the project by their IDs.

#### Parameters

| Parameter        | Type                             | Required | Description                        |
| ---------------- | -------------------------------- | -------- | ---------------------------------- |
| `*benchmark_ids` | `str`                            | Yes      | One or more benchmark IDs to add   |
| `timeout`        | `float \| httpx.Timeout \| None` | No       | Override request timeout           |

#### Returns

Returns `bool` - `True` if the operation succeeded, `False` otherwise.

#### Example

```python
client = Stratix()
success = client.benchmarks.add("benchmark-id-1", "benchmark-id-2")
```

### `remove(*benchmark_ids, timeout=None)`

Removes benchmarks from the project by their IDs.

#### Parameters

| Parameter        | Type                             | Required | Description                           |
| ---------------- | -------------------------------- | -------- | ------------------------------------- |
| `*benchmark_ids` | `str`                            | Yes      | One or more benchmark IDs to remove   |
| `timeout`        | `float \| httpx.Timeout \| None` | No       | Override request timeout              |

#### Returns

Returns `bool` - `True` if the operation succeeded, `False` otherwise.

#### Example

```python
client = Stratix()
success = client.benchmarks.remove("benchmark-id-1", "benchmark-id-2")
```

### `create_custom(name, description, file_path, additional_metrics=None, custom_scorer_ids=None, input_type=None, timeout=None)`

Creates a custom benchmark by uploading a JSONL file. The file should contain one JSON object per line with `input` and `truth` fields.

#### Parameters

| Parameter            | Type                             | Required | Description                                                          |
| -------------------- | -------------------------------- | -------- | -------------------------------------------------------------------- |
| `name`               | `str`                            | Yes      | Benchmark name (max 64 characters)                                   |
| `description`        | `str`                            | Yes      | Benchmark description (max 280 characters)                           |
| `file_path`          | `str`                            | Yes      | Path to a JSONL file with benchmark prompts                          |
| `additional_metrics` | `List[str] \| None`              | No       | Additional metrics: `readability`, `toxicity`, `hallucination`       |
| `custom_scorer_ids`  | `List[str] \| None`              | No       | List of custom scorer IDs to use                                     |
| `input_type`         | `str \| None`                    | No       | Input type: `messages` or `json_payload`                             |
| `timeout`            | `float \| httpx.Timeout \| None` | No       | Override request timeout                                             |

#### JSONL File Format

Each line should be a JSON object:

```json
{"input": "What is 2+2?", "truth": "4"}
{"input": "Capital of France?", "truth": "Paris"}
```

Optional fields: `subset` (for grouping prompts into categories).

#### Returns

Returns an `Optional[CreateBenchmarkResponse]` containing:

- `organization_id`: Organization identifier
- `project_id`: Project identifier
- `benchmark_id`: The newly created benchmark's identifier

Returns `None` if the request fails.

#### Example

```python
client = Stratix()

result = client.benchmarks.create_custom(
    name="QA Benchmark",
    description="Tests model factual accuracy",
    file_path="benchmark_data.jsonl",
    additional_metrics=["hallucination"],
)

if result:
    print(f"Created benchmark: {result.benchmark_id}")
```

### `create_smart(name, description, system_prompt, file_paths, metrics=None, timeout=None)`

Creates a smart benchmark from uploaded files. The platform uses AI to automatically generate benchmark prompts from the provided documents. The benchmark is generated asynchronously.

#### Parameters

| Parameter       | Type                             | Required | Description                                                         |
| --------------- | -------------------------------- | -------- | ------------------------------------------------------------------- |
| `name`          | `str`                            | Yes      | Benchmark name (max 256 characters)                                 |
| `description`   | `str`                            | Yes      | Benchmark description (max 500 characters)                          |
| `system_prompt` | `str`                            | Yes      | System prompt guiding benchmark generation (max 4000 characters)    |
| `file_paths`    | `List[str]`                      | Yes      | List of file paths to upload (1-20 files, max 50 MB each)           |
| `metrics`       | `List[str] \| None`              | No       | Additional metrics: `readability`, `toxicity`, `hallucination`      |
| `timeout`       | `float \| httpx.Timeout \| None` | No       | Override request timeout                                            |

#### Supported File Types

`.txt`, `.pdf`, `.html`, `.docx`, `.csv`, `.json`, `.jsonl`, `.parquet`

#### Returns

Returns an `Optional[CreateBenchmarkResponse]` containing:

- `organization_id`: Organization identifier
- `project_id`: Project identifier
- `benchmark_id`: The newly created benchmark's identifier

Returns `None` if the request fails.

#### Example

```python
client = Stratix()

result = client.benchmarks.create_smart(
    name="Product Knowledge Benchmark",
    description="Evaluates model knowledge of product docs",
    system_prompt="Generate QA pairs testing understanding of product features.",
    file_paths=["product_docs.pdf", "faq.txt"],
    metrics=["hallucination"],
)

if result:
    print(f"Smart benchmark created: {result.benchmark_id}")
    print("Check the dashboard for generation progress.")
```
