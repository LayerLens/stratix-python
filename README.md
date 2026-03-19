# LayerLens Stratix Python SDK

The official Python library for the [LayerLens Stratix](https://app.layerlens.ai) evaluation API.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```

## Authentication

Set your API key as an environment variable:

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

Or pass it directly when creating a client:

```python
from layerlens import Stratix

client = Stratix(api_key="your-api-key")
```

## Quick Start

### Run an evaluation

```python
import os
from layerlens import Stratix

client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"))

# Get a model and benchmark by key
model = client.models.get_by_key("openai/gpt-4o")
benchmark = client.benchmarks.get_by_key("arc-agi-2")

# Create an evaluation (pass the full model and benchmark objects)
evaluation = client.evaluations.create(
    model=model,
    benchmark=benchmark,
)

# Wait for results (pass the evaluation object, not just the ID)
result = client.evaluations.wait_for_completion(evaluation)
print(f"Accuracy: {result.accuracy}")
```

### Async usage

```python
import os
import asyncio
from layerlens import AsyncStratix

async def main():
    client = AsyncStratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"))

    model = await client.models.get_by_key("openai/gpt-4o")
    benchmark = await client.benchmarks.get_by_key("arc-agi-2")

    evaluation = await client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )

    result = await client.evaluations.wait_for_completion(evaluation)
    print(f"Accuracy: {result.accuracy}")

asyncio.run(main())
```

### Public endpoints

Public models, benchmarks, and evaluations are accessible through `client.public`. Note: the public client still requires an API key.

```python
import os
from layerlens import Stratix

client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"))

# Browse public models
models = client.public.models.get()
for model in models.models:
    print(f"{model.key}: {model.name}")
```

Or instantiate the public client directly:

```python
import os
from layerlens import PublicClient

public = PublicClient(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"))
models = public.models.get()
```

## Resources

The SDK provides access to these resource types:

| Resource                     | Description                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `client.models`              | Manage models (get, get_by_key, add, remove, create_custom)                   |
| `client.benchmarks`          | Manage benchmarks (get, get_by_key, add, remove, create_custom, create_smart) |
| `client.evaluations`         | Create evaluations and wait for results                                       |
| `client.judges`              | CRUD operations for evaluation judges                                         |
| `client.traces`              | Upload trace files and manage traces                                          |
| `client.trace_evaluations`   | Run trace-level evaluations with judges                                       |
| `client.judge_optimizations` | Optimize judge configurations                                                 |
| `client.results`             | Retrieve evaluation results                                                   |
| `client.public`              | Public models, benchmarks, evaluations, and comparisons                       |

Every resource is available in both sync (`Stratix`) and async (`AsyncStratix`) clients.

## Examples

### Working with judges

```python
# Create a judge (name and evaluation_goal are required)
judge = client.judges.create(
    name="Response Quality Judge",
    evaluation_goal="Rate whether the response is accurate, complete, and well-structured",
)

# List judges (returns a JudgesResponse with .judges list)
response = client.judges.get_many()
for j in response.judges:
    print(f"{j.name} (id: {j.id})")

# Update a judge
client.judges.update(judge.id, name="Updated Judge Name")

# Delete a judge
client.judges.delete(judge.id)
```

### Uploading and evaluating traces

Trace upload works with JSON or JSONL files (up to 50 MB). The SDK handles presigned S3 uploads automatically.

```python
# Upload a trace file (pass a file path, not raw data)
result = client.traces.upload("./my_traces.json")
print(f"Uploaded trace IDs: {result.trace_ids}")

# List traces
traces = client.traces.get_many()
for t in traces.traces:
    print(f"Trace {t.id}")

# Create a trace evaluation
trace_eval = client.trace_evaluations.create(
    trace_id=t.id,
    judge_id=judge.id,
)

# Get results
results = client.trace_evaluations.get_results(trace_eval.id)
```

### Custom models

Custom models require an OpenAI-compatible API endpoint.

```python
response = client.models.create_custom(
    name="My Fine-tuned Model",
    key="my-org/custom-model-v1",
    description="Fine-tuned GPT for medical Q&A",
    api_url="https://my-api.example.com/v1",
    max_tokens=4096,
    api_key=os.environ.get("MY_PROVIDER_API_KEY"),  # optional
)
print(f"Created model: {response.model_id}")
```

## Client aliases

For backward compatibility, multiple import names are available:

```python
from layerlens import Stratix          # Primary
from layerlens import AsyncStratix     # Async primary
from layerlens import Client           # Alias for Stratix
from layerlens import AsyncClient      # Alias for AsyncStratix
from layerlens import Atlas            # Legacy alias
from layerlens import AsyncAtlas       # Legacy alias
from layerlens import PublicClient     # Public endpoints
from layerlens import AsyncPublicClient
```

## Configuration

| Environment Variable         | Description               | Default                           |
| ---------------------------- | ------------------------- | --------------------------------- |
| `LAYERLENS_STRATIX_API_KEY`  | Your API key              | (required)                        |
| `LAYERLENS_STRATIX_BASE_URL` | Override the API base URL | `https://api.layerlens.ai/api/v1` |

Legacy env vars (`LAYERLENS_ATLAS_API_KEY`, `LAYERLENS_ATLAS_BASE_URL`) are also supported.

## Error handling

The SDK raises typed exceptions for API errors:

```python
import os
from layerlens import Stratix, StratixError, APIError, BadRequestError, NotFoundError

client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"))

try:
    result = client.models.get_by_id("nonexistent-id")
except NotFoundError as e:
    print(f"Not found (HTTP {e.status_code}): {e.message}")
except BadRequestError as e:
    print(f"Bad request: {e.message}")
except APIError as e:
    print(f"API error: {e.message}")
except StratixError as e:
    print(f"Client error: {e}")
```

Catch the most specific exception first. The hierarchy:

- `StratixError` (base for all SDK errors)
  - `APIError` (base for all API-related errors)
    - `APIConnectionError` (network issues)
      - `APITimeoutError` (request timed out)
    - `APIResponseValidationError` (response didn't match expected schema)
    - `APIStatusError` (HTTP 4xx/5xx)
      - `BadRequestError` (400)
      - `AuthenticationError` (401)
      - `PermissionDeniedError` (403)
      - `NotFoundError` (404)
      - `ConflictError` (409)
      - `UnprocessableEntityError` (422)
      - `RateLimitError` (429)
      - `InternalServerError` (500+)

Note: Only `StratixError`, `APIError`, `BadRequestError`, `AuthenticationError`, and `NotFoundError` are exported from the top-level package. For other exception types, import from `layerlens._exceptions`.

## CLI

The LayerLens CLI lets you manage traces, judges, evaluations, integrations, and more from the terminal.

### Install

```bash
pip install layerlens[cli] --extra-index-url https://sdk.layerlens.ai/package
```

### Configure

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

### Usage

```bash
layerlens --help                   # Show all commands
layerlens trace list               # List traces
layerlens evaluate run \
  --model openai/gpt-4o \
  --benchmark arc-agi-2 --wait     # Run an evaluation and wait for results
layerlens judge create \
  --name "Quality" \
  --goal "Rate response quality" \
  --model-id <MODEL_ID>            # Create a judge
layerlens ci report -o summary.md  # Generate CI report
```

Shell completions are available for bash, zsh, fish, and powershell:

```bash
layerlens completion bash          # Print setup instructions
```

Full CLI docs: [docs/cli/](docs/cli/)

| Guide | Description |
| --- | --- |
| [Getting Started](docs/cli/getting-started.md) | Installation, configuration, first commands |
| [Command Reference](docs/cli/commands.md) | All commands and options |
| [Examples](docs/cli/examples.md) | 15 common workflows as copy-paste shell sessions |

## Requirements

- Python 3.8+
- Dependencies: `httpx`, `pydantic`, `requests`
- CLI extra: `click>=8.0.0`

## Documentation

Full API reference and examples are available in the [docs/](docs/) directory:

- [CLI Guide](docs/cli/) (getting started, command reference, workflow examples)
- [API Reference](docs/api-reference/) (client config, all resource methods, error handling)
- [Code Examples](docs/examples/) (evaluations, judges, traces)
- [Troubleshooting](docs/troubleshooting/) (auth issues, error codes)

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
