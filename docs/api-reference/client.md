# Client Configuration

The `Stratix` (syncronous) and `AsyncStratix` (asyncronous) classes are the main entry points for interacting with the LayerLens Stratix sdk. This page covers client initialization, configuration options, and usage patterns.

## Basic Usage

### Syncronous Client

```python
from layerlens import Stratix

# Construct syncronous client
# Loads for api key from the "LAYERLENS_STRATIX_API_KEY" enviornment variable
client = Stratix()

# Explicit configuration
client = Stratix(api_key="your_api_key")
```

### Asyncronous Client

```python
import asyncio
from layerlens import AsyncStratix

# Construct async client
# Loads for api key from the "LAYERLENS_STRATIX_API_KEY" enviornment variable
client = AsyncStratix()

# Explicit configuration
client = AsyncStratix(api_key="your_api_key")
```

## Constructor Parameters

### `Stratix(api_key, base_url, timeout, max_retries)` and `AsyncStratix(api_key, base_url, timeout, max_retries)`

| Parameter     | Type                             | Required | Default       | Description                   |
| ------------- | -------------------------------- | -------- | ------------- | ----------------------------- |
| `api_key`     | `str \| None`                    | Yes\*    | `None`        | Your LayerLens Stratix API key  |
| `base_url`    | `str \| httpx.URL \| None`       | No       | Stratix API URL | Custom API base URL           |
| `timeout`     | `float \| httpx.Timeout \| None` | No       | 10 minutes    | Request timeout configuration |
| `max_retries` | `int`                            | No       | `2`           | Maximum number of retries on retryable errors (429, 500, 502, 503, 504) |

\*Required unless set via environment variables

## Environment Variable Configuration

The client automatically loads configuration from these environment variables:

```bash
LAYERLENS_STRATIX_API_KEY="your_api_key_here"
```

## Public Client

For accessing public endpoints (models, benchmarks, comparisons), use `PublicClient` or `AsyncPublicClient`. See the [Public Client](public-client.md) reference for full details.

```python
from layerlens import PublicClient

# Loads API key from the "LAYERLENS_STRATIX_API_KEY" environment variable
public = PublicClient()
models = public.models.get(companies=["OpenAI"])
```

You can also access public endpoints from an authenticated client via the `.public` property:

```python
client = Stratix()
public_models = client.public.models.get(query="claude")
```

## Timeout Configuration

### Simple Timeout

```python
from layerlens import Stratix

# 30-second timeout for all requests
client = Stratix(timeout=30.0)
```

### Retry Configuration

The client automatically retries requests that fail with retryable status codes (429 Too Many Requests, 500, 502, 503, 504) using exponential backoff. If the server sends a `Retry-After` header, the client respects it.

```python
from layerlens import Stratix

# Default: 2 retries
client = Stratix()

# More retries for batch-heavy workloads
client = Stratix(max_retries=5)

# Disable retries entirely
client = Stratix(max_retries=0)
```

### Per-Request Timeout Override

```python
client = Stratix()

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
