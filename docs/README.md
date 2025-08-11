# Atlas Python SDK Documentation

Welcome to the official documentation for the Atlas Python SDK. This library provides convenient access to the LayerLens Atlas REST API from any Python 3.8+ application.

## What is Atlas?

Atlas is LayerLens's evaluation platform that allows you to benchmark AI models against various datasets and metrics. The Python SDK provides a synchronous HTTP client powered by [httpx](https://github.com/encode/httpx) and [Pydantic](https://pydantic.dev/) models for type-safe API interactions.

## Key Features

- **Simple Authentication**: Secure API key-based authentication
- **Type Safety**: Full Pydantic model support for all API responses
- **Comprehensive Error Handling**: Detailed exception hierarchy for different error scenarios
- **Configurable Timeouts**: Fine-grained timeout control for different operations
- **Environment Variable Support**: Easy configuration through environment variables
- **Python 3.8+ Compatibility**: Works with modern Python versions

## Quick Start

```python
import os
from atlas import Atlas

# Initialize the client
client = Atlas(
    api_key=os.environ.get("LAYERLENS_ATLAS_API_KEY"),
    organization_id=os.environ.get("LAYERLENS_ATLAS_ORG_ID"), 
    project_id=os.environ.get("LAYERLENS_ATLAS_PROJECT_ID"),
)

# Create an evaluation
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)

# Get results
if evaluation:
    results = client.results.get(evaluation_id=evaluation.id)
    print(f"Evaluation completed with {len(results)} results")
```

## Navigation

- **[Getting Started](getting-started/)** - Installation, setup, and your first API call
- **[API Reference](api-reference/)** - Complete documentation of all available methods
- **[Code Examples](examples/)** - Practical examples for common use cases
- **[Troubleshooting](troubleshooting/)** - Solutions to common issues
- **[Security](security/)** - Best practices for secure API usage

## Support

- **LayerLens Support**: Contact support through your LayerLens dashboard
- **Documentation**: Visit [docs.layerlens.com](https://docs.layerlens.com) for additional resources
- **API Status**: Check the [LayerLens status page](https://status.layerlens.com) for service updates

## License

This SDK is released under the MIT License.