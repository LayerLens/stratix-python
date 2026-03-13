# Common Issues

This guide covers frequently encountered issues when using the LayerLens Python SDK and how to resolve them.

## Installation Issues

### Package Not Found

**Error**: `Could not find a version that satisfies the requirement layerlens`

Make sure to include the custom package index:

```bash
pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```

### Python Version Incompatibility

**Error**: `Requires-Python >=3.8`

The SDK requires Python 3.8 or higher. Check your version:

```bash
python --version
```

## Client Initialization

### Missing API Key

**Error**: `StratixError: The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_STRATIX_API_KEY environment variable`

Set the environment variable or pass the key explicitly:

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

```python
client = Stratix(api_key="your-api-key")
```

### Organization Not Found

**Error**: `NotFoundError` during client initialization

Your API key may be invalid or expired. Verify it in the [LayerLens dashboard](https://app.layerlens.ai).

## Evaluation Issues

### Evaluation Stuck in Progress

If `wait_for_completion` seems to hang, the evaluation may be queued behind other jobs. You can check the status manually:

```python
evaluation = client.evaluations.get(evaluation.id)
print(f"Status: {evaluation.status}")
```

### Model or Benchmark Not Found

When using `get_by_key`, ensure the key matches exactly (case-sensitive):

```python
# Correct
model = client.models.get_by_key("openai/gpt-4o")

# Search by name if unsure
models = client.models.get(type="public", name="gpt-4o")
```

## Trace Upload Issues

### File Too Large

Trace files must be under 50 MB. For larger datasets, split the file into smaller chunks.

### Invalid File Format

Trace uploads accept `.json` and `.jsonl` files only. Ensure your file contains valid JSON.

## Timeout Errors

**Error**: `APITimeoutError`

The default timeout is 10 minutes. For long-running operations, increase it:

```python
# Global timeout
client = Stratix(timeout=1200.0)  # 20 minutes

# Per-request timeout
result = client.with_options(timeout=300.0).evaluations.create(
    model=model,
    benchmark=benchmark,
)
```

## Network Issues

**Error**: `APIConnectionError`

- Verify your internet connection
- Check if `https://api.layerlens.ai` is reachable
- If behind a proxy, configure `httpx` accordingly

## See Also

- [Authentication Problems](authentication.md) for auth-specific issues
- [Error Codes Reference](error-codes.md) for the full exception hierarchy
