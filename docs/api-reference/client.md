# Client Configuration

The `Atlas` class is the main entry point for interacting with the LayerLens Atlas API. This page covers client initialization, configuration options, and advanced usage patterns.

## Basic Usage

```python
from atlas import Atlas

# Using environment variables (recommended)
client = Atlas()

# Explicit configuration
client = Atlas(api_key="your_api_key")
```

## Constructor Parameters

### `Atlas(api_key, base_url, timeout)`

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
LAYERLENS_ATLAS_BASE_URL="https://custom-endpoint.com/api/v1"  # Optional
```

## Timeout Configuration

### Simple Timeout

```python
from atlas import Atlas

# 30-second timeout for all requests
client = Atlas(timeout=30.0)
```

### Advanced Timeout Configuration

```python
import httpx
from atlas import Atlas

client = Atlas(
    timeout=httpx.Timeout(
        connect=5.0,    # Connection timeout: 5 seconds
        read=60.0,      # Read timeout: 60 seconds
        write=30.0,     # Write timeout: 30 seconds
        pool=10.0       # Connection pool timeout: 10 seconds
    )
)
```

### Per-Request Timeout Override

```python
client = Atlas()

# Override timeout for a specific request
evaluation = client.with_options(timeout=120.0).evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)
```

## Client Methods

### `copy(**kwargs)`

Create a new client instance with modified configuration:

```python
# Base client
client = Atlas(api_key="key1")

# Create a copy with different timeout
slow_client = client.copy(timeout=300.0)  # 5 minutes
```

### `with_options(**kwargs)`

Temporarily override client options for a single request chain:

```python
client = Atlas()

# Use different timeout for this request only
evaluation = client.with_options(timeout=60.0).evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)

# Back to original timeout for subsequent requests
results = client.results.get(evaluation_id=evaluation.id)
```

## Resource Access

The client provides access to different API resources through properties:

```python
client = Atlas()

# Access evaluations resource
client.evaluations.create(model="gpt-4", benchmark="mmlu")

# Access results resource
client.results.get(evaluation_id="eval_123")
```

Available resources:

- `client.evaluations` - Create and manage evaluations
- `client.results` - Retrieve evaluation results
- More resources coming soon...

## Error Handling

The client raises specific exceptions for different error conditions:

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    evaluation = client.evaluations.create(model="invalid", benchmark="invalid")
except atlas.AuthenticationError:
    # 401 - Invalid API key
    print("Authentication failed")
except atlas.PermissionDeniedError:
    # 403 - Valid API key, insufficient permissions
    print("Permission denied")
except atlas.NotFoundError:
    # 404 - Resource not found
    print("Model or benchmark not found")
except atlas.RateLimitError:
    # 429 - Too many requests
    print("Rate limit exceeded")
except atlas.InternalServerError:
    # 500+ - Server error
    print("Server error occurred")
except atlas.APIConnectionError:
    # Network/connection issues
    print("Connection failed")
except atlas.APITimeoutError:
    # Request timeout
    print("Request timed out")
```

## Authentication Headers

The client automatically handles authentication by adding the required headers:

```python
# The client adds this header to all requests:
# x-api-key: your_api_key_value
```

You don't need to manually handle authentication headers.

## Base URL Configuration

### Default Base URL

The client uses the default LayerLens Atlas API endpoint unless overridden.

### Custom Base URL

For enterprise or self-hosted deployments:

```python
from atlas import Atlas

client = Atlas(
    base_url="https://your-atlas-instance.com/api/v1"
)

# Or via environment variable
# LAYERLENS_ATLAS_BASE_URL="https://your-atlas-instance.com/api/v1"
client = Atlas()  # Will use custom base URL from environment
```

## Best Practices

### 1. Use Environment Variables

```python
# ✅ Good - secure and flexible
client = Atlas()

# ❌ Bad - hardcoded credentials
client = Atlas(api_key="hardcoded_key")
```

### 2. Configure Appropriate Timeouts

```python
# ✅ Good - reasonable timeout for evaluation creation
client = Atlas(timeout=120.0)  # 2 minutes

# ❌ Bad - too short for long-running operations
client = Atlas(timeout=5.0)  # 5 seconds might be too short
```

### 3. Handle Errors Gracefully

```python
# ✅ Good - specific error handling
try:
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.RateLimitError:
    time.sleep(60)  # Wait before retrying
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIError as e:
    logger.error(f"API error: {e}")
    raise
```

### 4. Reuse Client Instances

```python
# ✅ Good - reuse the same client
client = Atlas()
eval1 = client.evaluations.create(model="gpt-4", benchmark="mmlu")
eval2 = client.evaluations.create(model="claude-3", benchmark="hellaswag")

# ❌ Bad - creating new clients unnecessarily
client1 = Atlas()
eval1 = client1.evaluations.create(model="gpt-4", benchmark="mmlu")
client2 = Atlas()  # Unnecessary
eval2 = client2.evaluations.create(model="claude-3", benchmark="hellaswag")
```

## Thread Safety

The Atlas client is thread-safe and can be shared across multiple threads:

```python
import threading
from atlas import Atlas

client = Atlas()

def create_evaluation(model_name):
    evaluation = client.evaluations.create(
        model=model_name,
        benchmark="mmlu"
    )
    print(f"Created evaluation for {model_name}: {evaluation.id}")

# Safe to use the same client across threads
threads = []
for model in ["gpt-4", "claude-3", "llama-2"]:
    thread = threading.Thread(target=create_evaluation, args=(model,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```
