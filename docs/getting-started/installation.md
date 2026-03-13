# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from LayerLens Package Registry

```bash
pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```

## Verify Installation

```python
import layerlens
print(layerlens.__version__)
```

## Dependencies

The SDK installs the following dependencies automatically:

- `httpx` - HTTP client for API requests
- `pydantic` - Data validation and serialization
- `requests` - Used for file uploads (presigned S3 URLs)

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade layerlens --extra-index-url https://sdk.layerlens.ai/package
```

## Virtual Environment (Recommended)

It's recommended to install the SDK in a virtual environment to avoid dependency conflicts:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```
