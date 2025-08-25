# Models & Benchmarks

This page provides reference information about available models and benchmarks in the Atlas platform, along with guidance on selecting appropriate combinations for your evaluations.

## Overview

Atlas evaluations require two key components:
- **Model**: The AI model you want to evaluate
- **Benchmark**: The dataset/test suite to evaluate the model against

The availability of models and benchmarks depends on your organizations configuration.

### Finding Available Models and Benchmarks

#### Check the Atlas Dashboard
The most reliable way to find available models and benchmarks:

1. Log into your Atlas dashboard
2. Navigate to the evaluation creation page
3. View dropdown lists of available models and benchmarks


## Models

### `get(type=None, name=None, companies=None, regions=None, licenses=None, timeout=None)`

Retrieves a list of available models with optional filtering parameters.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | `Literal["custom", "public"] \| None` | No | Filter by model type. If `None`, returns both custom and public models |
| `name` | `str \| None` | No | Filter models by name (partial match search) |
| `companies` | `List[str] \| None` | No | Filter by model companies/providers |
| `regions` | `List[str] \| None` | No | Filter by supported regions |
| `licenses` | `List[str] \| None` | No | Filter by license types |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a `List[Model]` containing available models that match the filter criteria. Returns `None` if no models are found or if there's an error.

#### Examples

##### Get All Models

```python
from atlas import Atlas

client = Atlas()

# Get all available models (both custom and public)
models = client.models.get()

if models:
    print(f"Found {len(models)} models:")
    for model in models:
        print(f"  {model.id} - {model.name} ({model.company})")
else:
    print("No models available")
```

##### Filter by Company

```python
# Get models from specific companies
openai_models = client.models.get(companies=["OpenAI"])
anthropic_models = client.models.get(companies=["Anthropic"])

# Get models from multiple companies
major_models = client.models.get(companies=["OpenAI", "Anthropic", "Google"])

if major_models:
    for model in major_models:
        print(f"{model.name} by {model.company}")
```

##### Search by Name

```python
# Search for models with "gpt" in the name
gpt_models = client.models.get(name=["gpt"])


if gpt_models:
    print("GPT models found:")
    for model in gpt_models:
        print(f"  {model.id} - {model.name}")
```

#### Model Object Properties

Each `Model` object in the returned list contains:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique model identifier for use in evaluations |
| `name` | `str` | Human-readable model name |
| `company` | `str` | Company/provider that created the model |
| `type` | `str` | Model type ("custom" or "public") |
| `region` | `str` | Supported region |
| `license` | `str` | License type |

#### Error Handling

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    models = client.models.get()
    if models:
        print(f"Retrieved {len(models)} models")
    else:
        print("No models available or error occurred")
except atlas.AuthenticationError:
    print("Authentication failed - check your API key")
except atlas.PermissionDeniedError:
    print("No permission to access models")
except atlas.APIConnectionError as e:
    print(f"Connection error: {e}")
except atlas.APIError as e:
    print(f"API error: {e}")
```

### Model Identification

Models are identified by string IDs that you pass to the `evaluations.create()` method:

```python
from atlas import Atlas

client = Atlas()

# Using model ID
evaluation = client.evaluations.create(
    model="gpt-4",  # Model ID
    benchmark="mmlu"
)
```

### Model Information

When you create an evaluation, the response includes detailed model information:

```python
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

if evaluation:
    print(f"Model ID: {evaluation.model_id}")           # "gpt-4"
    print(f"Model Name: {evaluation.model_name}")       # "GPT-4"
    print(f"Model Key: {evaluation.model_key}")         # Internal key
    print(f"Model Company: {evaluation.model_company}") # "OpenAI"
```

## Benchmarks

### `get(type=None, name=None, timeout=None)`

Retrieves a list of available benchmarks with optional filtering parameters.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | `Literal["custom", "public"] \| None` | No | Filter by benchmark type. If `None`, returns both custom and public benchmarks |
| `name` | `str \| None` | No | Filter benchmarks by name (partial match search) |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns a `List[Benchmark]` containing available benchmarks that match the filter criteria. Returns `None` if no benchmarks are found or if there's an error.

#### Examples

##### Get All Benchmarks

```python
from atlas import Atlas

client = Atlas()

# Get all available benchmarks (both custom and public)
benchmarks = client.benchmarks.get()

if benchmarks:
    print(f"Found {len(benchmarks)} benchmarks:")
    for benchmark in benchmarks:
        print(f"  {benchmark.id} - {benchmark.name}")
else:
    print("No benchmarks available")
```

##### Search by Name

```python
# Search for benchmarks with "mmlu" in the name
mmlu_benchmark = client.benchmarks.get(name=["mmlu"])[0]

if mmlu_benchmark:
    print("MMLU benchmark found:")
    print(f"  {mmlu_benchmark.id} - {mmlu_benchmark.name}")
```


#### Benchmark Object Properties

Each `Benchmark` object in the returned list contains:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique benchmark identifier for use in evaluations |
| `name` | `str` | Human-readable benchmark name |
| `type` | `str` | Benchmark type ("custom" or "public") |
| `description` | `str` | Description of what the benchmark tests |

#### Error Handling

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    benchmarks = client.benchmarks.get()
    if benchmarks:
        print(f"Retrieved {len(benchmarks)} benchmarks")
    else:
        print("No benchmarks available or error occurred")
except atlas.AuthenticationError:
    print("Authentication failed - check your API key")
except atlas.PermissionDeniedError:
    print("No permission to access benchmarks")
except atlas.APIConnectionError as e:
    print(f"Connection error: {e}")
except atlas.APIError as e:
    print(f"API error: {e}")
```

## Discovery and Validation

### Finding Available Models and Benchmarks

#### Check the Atlas Dashboard
The most reliable way to find available models and benchmarks:

1. Log into your Atlas dashboard
2. Navigate to the evaluation creation page
3. View dropdown lists of available models and benchmarks


### Benchmark Identification

Benchmarks are identified by string IDs representing different evaluation datasets:

```python
from atlas import Atlas

client = Atlas()

evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"  # Benchmark ID
)
```

### Benchmark Information

Evaluation responses include benchmark details:

```python
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

if evaluation:
    print(f"Dataset ID: {evaluation.dataset_id}")       # "mmlu"
    print(f"Dataset Name: {evaluation.dataset_name}")   # "MMLU"
```


## Troubleshooting

### Model or Benchmark Not Found

```python
try:
    evaluation = client.evaluations.create(
        model="nonexistent-model",
        benchmark="mmlu"
    )
except atlas.NotFoundError:
    print("Model or benchmark not found. Check:")
    print("1. Spelling of model/benchmark ID")
    print("2. Available options in Atlas dashboard")
    print("3. Your organization's access permissions")
```

### Permission Issues

```python
try:
    evaluation = client.evaluations.create(
        model="restricted-model",
        benchmark="private-benchmark"
    )
except atlas.PermissionDeniedError:
    print("Access denied. Possible causes:")
    print("1. Model requires higher permission level")
    print("2. Benchmark is not available to your organization")
    print("3. Project doesn't have access to these resources")
```

### Validation Errors

```python
try:
    evaluation = client.evaluations.create(
        model="",  # Empty string
        benchmark="mmlu"
    )
except atlas.BadRequestError:
    print("Invalid request parameters:")
    print("- Model and benchmark IDs cannot be empty")
    print("- IDs must be valid strings")
```

For more information about available models and benchmarks, consult your Atlas dashboard or contact your LayerLens administrator.