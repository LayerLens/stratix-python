# Authentication & Configuration

## API Key Setup

The SDK authenticates using an API key tied to your LayerLens organization. You can obtain your key from the [LayerLens dashboard](https://app.layerlens.ai).

### Environment Variable (Recommended)

Set the `LAYERLENS_STRATIX_API_KEY` environment variable and the client will pick it up automatically:

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

```python
from layerlens import Stratix

# Automatically reads from LAYERLENS_STRATIX_API_KEY
client = Stratix()
```

### Explicit API Key

Pass the key directly when constructing the client:

```python
from layerlens import Stratix

client = Stratix(api_key="your-api-key")
```

### Using a .env File

```bash
# .env (add this file to .gitignore)
LAYERLENS_STRATIX_API_KEY=your-api-key
```

```python
from dotenv import load_dotenv
load_dotenv()

from layerlens import Stratix
client = Stratix()
```

## Configuration Options

| Parameter  | Type                             | Default                           | Description     |
| ---------- | -------------------------------- | --------------------------------- | --------------- |
| `api_key`  | `str \| None`                    | `LAYERLENS_STRATIX_API_KEY` env   | Your API key    |
| `base_url` | `str \| httpx.URL \| None`       | `https://api.layerlens.ai/api/v1` | API base URL    |
| `timeout`  | `float \| httpx.Timeout \| None` | 10 minutes                        | Request timeout |

## Environment Variables

| Variable                     | Description               | Default                           |
| ---------------------------- | ------------------------- | --------------------------------- |
| `LAYERLENS_STRATIX_API_KEY`  | Your API key              | (required)                        |
| `LAYERLENS_STRATIX_BASE_URL` | Override the API base URL | `https://api.layerlens.ai/api/v1` |

Legacy environment variables (`LAYERLENS_ATLAS_API_KEY`, `LAYERLENS_ATLAS_BASE_URL`) are also supported for backward compatibility.

## Organization & Project

On initialization, the client fetches your organization and project IDs automatically using your API key. These are used to scope all API requests.

## Timeout Configuration

```python
# Global timeout (30 seconds)
client = Stratix(timeout=30.0)

# Per-request timeout override
evaluation = client.with_options(timeout=120.0).evaluations.create(
    model=model,
    benchmark=benchmark,
)
```
