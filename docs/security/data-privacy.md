# Data Privacy

This guide covers data privacy considerations when using the LayerLens Stratix SDK.

## Data Sent to the API

When using the SDK, the following data is transmitted to LayerLens servers:

- **API key** - Used for authentication (sent in request headers)
- **Trace data** - Contents of trace files you upload via `client.traces.upload()`
- **Evaluation parameters** - Model and benchmark selections, judge configurations
- **Judge definitions** - Evaluation goals and settings for custom judges

## Data Storage

- Uploaded traces are stored in your organization's project scope and are not shared across organizations
- Evaluation results are associated with your organization and project
- API keys are never logged or stored in SDK-side logs

## Sensitive Data in Traces

If your traces contain sensitive information (PII, credentials, proprietary data), consider:

1. **Sanitize before upload** - Remove or redact sensitive fields from trace files before uploading
2. **Use test data** - Use synthetic or anonymized data for development and testing
3. **Review trace contents** - Inspect trace files before upload to ensure no secrets are included

## Logging

The SDK uses Python's standard `logging` module. Sensitive headers (API keys, authorization tokens) are automatically redacted in log output.

```python
import logging

# Enable SDK logging (API keys will be redacted)
logging.basicConfig(level=logging.DEBUG)
```

## Local Data

The SDK does not write any data to disk. All operations are performed in-memory and transmitted directly to the API. Trace files are read from disk for upload but are not modified or cached.

## Environment Variable Security

Store API keys in environment variables rather than in source code:

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

See [API Key Management](api-key-management.md) and [Environment Variables](environment-variables.md) for detailed guidance.
