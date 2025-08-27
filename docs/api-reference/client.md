# Client Configuration

The `Atlas` (syncronous) and `AsyncAtlas` (asyncronous) classes are the main entry points for interacting with the LayerLens Atlas sdk. This page covers client initialization, configuration options, and usage patterns.

## Basic Usage

### Syncronous Client

```python
from layerlens import Atlas

# Construct syncronous client
# Loads for api key from the "LAYERLENS_ATLAS_API_KEY" enviornment variable
client = Atlas()

# Explicit configuration
client = Atlas(api_key="your_api_key")
```

### Asyncronous Client

```python
import asyncio
from layerlens import AsyncAtlas

# Construct async client
# Loads for api key from the "LAYERLENS_ATLAS_API_KEY" enviornment variable
client = AsyncAtlas()

# Explicit configuration
client = AsyncAtlas(api_key="your_api_key")
```

## Constructor Parameters

### `Atlas(api_key, base_url, timeout)` and `AsyncAtlas(api_key, base_url, timeout)`

| Parameter  | Type                             | Required | Default       | Description                   |
| ---------- | -------------------------------- | -------- | ------------- | ----------------------------- |
| `api_key`  | `str \| None`                    | Yes\*    | `None`        | Your LayerLens Atlas API key  |
| `base_url` | `str \| httpx.URL \| None`       | No       | Atlas API URL | Custom API base URL           |
| `timeout`  | `float \| httpx.Timeout \| None` | No       | 10 minutes    | Request timeout configuration |

\*Required unless set via environment variables

## Environment Variable Configuration

The client automatically loads configuration from these environment variables:

```bash
LAYERLENS_ATLAS_API_KEY="your_api_key_here"
```

## Timeout Configuration

### Simple Timeout

```python
from layerlens import Atlas

# 30-second timeout for all requests
client = Atlas(timeout=30.0)
```

### Per-Request Timeout Override

```python
client = Atlas()

# --- Models replace with the model name you want to run
models = client.models.get(type="public", name="gpt-4o")

if not models:
    print("gpt-4o not found on organization, exiting")

model = models[0]

# --- Benchmarks replace with the benchmark name you want to run
benchmarks = client.benchmarks.get(type="public", name="simpleQA")

if not benchmarks:
    print("SimpleQA benchmark not found on organization, exiting")

benchmark = benchmarks[0]

# Override timeout for a specific request
evaluation = client.with_options(timeout=120.0).evaluations.create(
    model=model,
    benchmark=benchmark
)
```
