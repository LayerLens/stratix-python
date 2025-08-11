# Installation

The Atlas Python SDK supports Python 3.8 and above. You can install it using pip or your preferred Python package manager.

## Install from PyPI

```bash
pip install atlas
```

## Verify Installation

After installation, verify that the SDK is working correctly:

```python
import atlas
print(atlas.__version__)
```

This should print the version number of the installed SDK.

## System Requirements

- **Python**: 3.8 or higher
- **Operating Systems**: Windows, macOS, Linux
- **Dependencies**: The SDK automatically installs required dependencies:
  - `httpx` - HTTP client library
  - `pydantic` - Data validation and serialization
  - `typing-extensions` - Enhanced type hints for older Python versions

## Virtual Environment (Recommended)

We strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv atlas-env

# Activate it (Linux/macOS)
source atlas-env/bin/activate

# Activate it (Windows)
atlas-env\Scripts\activate

# Install the SDK
pip install atlas
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade atlas
```
