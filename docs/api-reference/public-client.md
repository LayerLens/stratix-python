# Public Client

The `PublicClient` (synchronous) and `AsyncPublicClient` (asynchronous) classes provide access to public LayerLens API endpoints for browsing public models, benchmarks, benchmark content, and comparing evaluations.

## Basic Usage

### Synchronous Client

```python
from layerlens import PublicClient

# Loads API key from the "LAYERLENS_STRATIX_API_KEY" environment variable
client = PublicClient()

# Browse public models
models = client.models.get(companies=["OpenAI"])

# Browse public benchmarks
benchmarks = client.benchmarks.get(languages=["English"])
```

### Asynchronous Client

```python
import asyncio
from layerlens import AsyncPublicClient

async def main():
    client = AsyncPublicClient()

    models = await client.models.get(companies=["OpenAI"])
    benchmarks = await client.benchmarks.get(languages=["English"])

asyncio.run(main())
```

### Accessing from an Authenticated Client

If you already have an authenticated `Stratix` or `AsyncStratix` client, you can access public endpoints through the `.public` property:

```python
from layerlens import Stratix

client = Stratix()  # requires API key

# Access public endpoints through the authenticated client
public_models = client.public.models.get(query="claude")
```

## Constructor Parameters

### `PublicClient(api_key, base_url, timeout)` and `AsyncPublicClient(api_key, base_url, timeout)`

| Parameter  | Type                             | Required | Default         | Description                   |
| ---------- | -------------------------------- | -------- | --------------- | ----------------------------- |
| `api_key`  | `str \| None`                    | Yes\*    | `None`          | Your LayerLens Stratix API key  |
| `base_url` | `str \| httpx.URL \| None`       | No       | Stratix API URL | Custom API base URL           |
| `timeout`  | `float \| httpx.Timeout \| None` | No       | 10 minutes      | Request timeout configuration |

\*Required unless set via the `LAYERLENS_STRATIX_API_KEY` environment variable

## Public Models

### `models.get(...)`

Retrieves a list of public models with optional filtering, sorting, and pagination.

#### Parameters

| Parameter            | Type                   | Required | Description                                                                                  |
| -------------------- | ---------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `query`              | `str \| None`          | No       | Full-text search on model name                                                               |
| `name`               | `str \| None`          | No       | Filter by model name                                                                         |
| `key`                | `str \| None`          | No       | Filter by model key                                                                          |
| `ids`                | `List[str] \| None`    | No       | Filter by specific model IDs                                                                 |
| `categories`         | `List[str] \| None`    | No       | Filter by categories (e.g. `transformer`, `moe`, `open-source`, `closed-source`, `usa`, `china`, `size-sm`, `size-md`, `size-lg`, `size-xl`) |
| `companies`          | `List[str] \| None`    | No       | Filter by company names                                                                      |
| `regions`            | `List[str] \| None`    | No       | Filter by regions                                                                            |
| `licenses`           | `List[str] \| None`    | No       | Filter by license types                                                                      |
| `sizes`              | `List[str] \| None`    | No       | Filter by size (Small, Medium, Large, Extra Large)                                           |
| `sort_by`            | `str \| None`          | No       | Sort column: `name`, `createdAt`, `releasedAt`, `architectureType`, `contextLength`, `license`, `region` |
| `order`              | `str \| None`          | No       | Sort order: `asc` or `desc`                                                                  |
| `page`               | `int \| None`          | No       | Page number (1-based)                                                                        |
| `page_size`          | `int \| None`          | No       | Results per page                                                                             |
| `include_deprecated` | `bool \| None`         | No       | Include deprecated models (default: false)                                                   |
| `timeout`            | `float \| httpx.Timeout \| None` | No | Override request timeout                                                                     |

#### Returns

Returns a `PublicModelsListResponse` containing:

- `models`: List of `PublicModelDetail` objects
- `categories`: List of available category strings
- `count`: Number of results in current page
- `total_count`: Total number of matching results

Returns `None` if the request fails.

#### PublicModelDetail Properties

| Property               | Type             | Description                        |
| ---------------------- | ---------------- | ---------------------------------- |
| `id`                   | `str`            | Unique model identifier            |
| `key`                  | `str`            | Unique model key                   |
| `name`                 | `str`            | Human-readable model name          |
| `description`          | `str \| None`    | Text description                   |
| `company`              | `str \| None`    | Model provider company             |
| `released_at`          | `int \| None`    | Release timestamp                  |
| `parameters`           | `float \| None`  | Number of parameters               |
| `modality`             | `str \| None`    | Model modality                     |
| `context_length`       | `int \| None`    | Maximum context length             |
| `architecture_type`    | `str \| None`    | Architecture type                  |
| `license`              | `str \| None`    | License type                       |
| `open_weights`         | `bool \| None`   | Whether weights are open           |
| `region`               | `str \| None`    | Region                             |
| `key_takeaways`        | `List[str] \| None` | Key takeaways                   |
| `deprecated`           | `bool \| None`   | Whether the model is deprecated    |
| `cost_per_input_token` | `str \| None`    | Cost per input token               |
| `cost_per_output_token`| `str \| None`    | Cost per output token              |

#### Example

```python
from layerlens import PublicClient

client = PublicClient()

# Get newest OpenAI models
response = client.models.get(
    companies=["OpenAI"],
    sort_by="releasedAt",
    order="desc",
    page_size=5,
)

for model in response.models:
    print(f"{model.name} - {model.context_length} context length")
```

## Public Benchmarks

### `benchmarks.get(...)`

Retrieves a list of public benchmarks with optional filtering, sorting, and pagination.

#### Parameters

| Parameter            | Type                   | Required | Description                                |
| -------------------- | ---------------------- | -------- | ------------------------------------------ |
| `query`              | `str \| None`          | No       | Full-text search                           |
| `name`               | `str \| None`          | No       | Filter by name                             |
| `key`                | `str \| None`          | No       | Filter by key                              |
| `ids`                | `List[str] \| None`    | No       | Filter by specific IDs                     |
| `categories`         | `List[str] \| None`    | No       | Filter by categories                       |
| `languages`          | `List[str] \| None`    | No       | Filter by languages                        |
| `sort_by`            | `str \| None`          | No       | Sort column (currently: `name`)            |
| `order`              | `str \| None`          | No       | Sort order: `asc` or `desc`                |
| `page`               | `int \| None`          | No       | Page number (1-based)                      |
| `page_size`          | `int \| None`          | No       | Results per page                           |
| `include_deprecated` | `bool \| None`         | No       | Include deprecated benchmarks              |
| `timeout`            | `float \| httpx.Timeout \| None` | No | Override request timeout               |

#### Returns

Returns a `PublicBenchmarksListResponse` containing:

- `datasets`: List of `PublicBenchmarkDetail` objects
- `categories`: List of available category strings
- `count`: Number of results in current page
- `total_count`: Total number of matching results

Returns `None` if the request fails.

#### PublicBenchmarkDetail Properties

| Property          | Type               | Description                           |
| ----------------- | ------------------ | ------------------------------------- |
| `id`              | `str`              | Unique benchmark identifier           |
| `key`             | `str`              | Unique benchmark key                  |
| `name`            | `str`              | Human-readable name                   |
| `description`     | `str \| None`      | Text description                      |
| `prompt_count`    | `int \| None`      | Number of prompts in the benchmark    |
| `language`        | `str \| None`      | Language of the benchmark             |
| `categories`      | `List[str] \| None`| Categories                            |
| `characteristics` | `List[str] \| None`| Characteristics                       |
| `deprecated`      | `bool \| None`     | Whether the benchmark is deprecated   |
| `is_public`       | `bool \| None`     | Whether the benchmark is public       |

### `benchmarks.get_prompts(benchmark_id, ...)`

Fetches prompts/content from a public benchmark with optional search and pagination.

#### Parameters

| Parameter      | Type                   | Required | Description                                    |
| -------------- | ---------------------- | -------- | ---------------------------------------------- |
| `benchmark_id` | `str`                  | Yes      | The benchmark ID to fetch prompts from         |
| `page`         | `int \| None`          | No       | Page number (1-based)                          |
| `page_size`    | `int \| None`          | No       | Results per page                               |
| `search_field` | `str \| None`          | No       | Search field: `id`, `input`, or `truth`        |
| `search_value` | `str \| None`          | No       | Search value                                   |
| `sort_by`      | `str \| None`          | No       | Sort field: `id`, `input`, or `truth`          |
| `sort_order`   | `str \| None`          | No       | Sort order: `asc` or `desc`                    |
| `timeout`      | `float \| httpx.Timeout \| None` | No | Override request timeout                   |

#### Returns

Returns a `BenchmarkPromptsResponse` containing:

- `status`: Response status string
- `data.prompts`: List of `BenchmarkPrompt` objects
- `data.count`: Total number of prompts

Returns `None` if the request fails.

#### BenchmarkPrompt Properties

| Property | Type  | Description                            |
| -------- | ----- | -------------------------------------- |
| `id`     | `str` | Unique prompt identifier               |
| `input`  | `str \| List \| Dict` | The prompt input          |
| `truth`  | `str` | The expected/ground truth answer       |

### `benchmarks.get_all_prompts(benchmark_id, timeout=None)`

Fetches all prompts from a benchmark by automatically handling pagination.

#### Parameters

| Parameter      | Type                   | Required | Description                             |
| -------------- | ---------------------- | -------- | --------------------------------------- |
| `benchmark_id` | `str`                  | Yes      | The benchmark ID to fetch prompts from  |
| `timeout`      | `float \| httpx.Timeout \| None` | No | Override request timeout            |

#### Returns

Returns a `List[BenchmarkPrompt]` containing all prompts in the benchmark.

#### Example

```python
from layerlens import PublicClient

client = PublicClient()

# List benchmarks
benchmarks = client.benchmarks.get(query="mmlu")

if benchmarks and benchmarks.datasets:
    benchmark = benchmarks.datasets[0]

    # Get first page of prompts
    prompts = client.benchmarks.get_prompts(benchmark.id, page=1, page_size=10)

    if prompts:
        print(f"Total prompts: {prompts.data.count}")
        for prompt in prompts.data.prompts:
            print(f"  Input: {str(prompt.input)[:80]}...")
            print(f"  Truth: {prompt.truth[:50]}")

    # Or fetch all prompts at once
    all_prompts = client.benchmarks.get_all_prompts(benchmark.id)
    print(f"All prompts: {len(all_prompts)}")
```

## Comparisons

### `comparisons.compare(...)`

Compares results between two evaluations side-by-side.

#### Parameters

| Parameter          | Type                   | Required | Description                                                                |
| ------------------ | ---------------------- | -------- | -------------------------------------------------------------------------- |
| `evaluation_id_1`  | `str`                  | Yes      | First evaluation ID                                                        |
| `evaluation_id_2`  | `str`                  | Yes      | Second evaluation ID                                                       |
| `page`             | `int \| None`          | No       | Page number (1-based)                                                      |
| `page_size`        | `int \| None`          | No       | Results per page                                                           |
| `outcome_filter`   | `str \| None`          | No       | Filter by outcome (see below)                                              |
| `search`           | `str \| None`          | No       | Search within results                                                      |
| `timeout`          | `float \| httpx.Timeout \| None` | No | Override request timeout                                               |

#### Outcome Filter Options

| Value                | Description                                    |
| -------------------- | ---------------------------------------------- |
| `"all"`              | All results (default)                          |
| `"both_succeed"`     | Both models answered correctly                 |
| `"both_fail"`        | Both models answered incorrectly               |
| `"reference_fails"`  | First model fails, second succeeds             |
| `"comparison_fails"` | Second model fails, first succeeds             |

#### Returns

Returns a `ComparisonResponse` containing:

- `results`: List of `ComparisonResult` objects
- `total_count`: Total number of comparable results
- `correct_count_1`: Number of correct answers for evaluation 1
- `total_results_1`: Total results for evaluation 1
- `correct_count_2`: Number of correct answers for evaluation 2
- `total_results_2`: Total results for evaluation 2

Returns `None` if the request fails.

#### ComparisonResult Properties

| Property      | Type            | Description                           |
| ------------- | --------------- | ------------------------------------- |
| `result_id_1` | `int \| None`   | Result ID from evaluation 1           |
| `result_id_2` | `int \| None`   | Result ID from evaluation 2           |
| `prompt`      | `str`           | The prompt text                       |
| `truth`       | `str`           | The ground truth answer               |
| `result1`     | `str \| None`   | Model 1's response                    |
| `score1`      | `float \| None` | Model 1's score                       |
| `result2`     | `str \| None`   | Model 2's response                    |
| `score2`      | `float \| None` | Model 2's score                       |

#### Example

```python
from layerlens import PublicClient

client = PublicClient()

comparison = client.comparisons.compare(
    evaluation_id_1="eval-abc",
    evaluation_id_2="eval-def",
    outcome_filter="reference_fails",
    page=1,
    page_size=20,
)

if comparison:
    print(f"Eval 1: {comparison.correct_count_1}/{comparison.total_results_1}")
    print(f"Eval 2: {comparison.correct_count_2}/{comparison.total_results_2}")

    for result in comparison.results:
        print(f"  Prompt: {result.prompt[:80]}...")
        print(f"  Model 1 score: {result.score1}, Model 2 score: {result.score2}")
```
