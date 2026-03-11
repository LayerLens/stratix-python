# Layerlens Python SDK Documentation

Welcome to the official documentation for the Layerlens Python SDK for the Stratix platform. This library provides convenient programmatic to the Stratix platform from any Python 3.8+ application.

## What is Stratix?

Stratix is an evaluation platform that allows you to benchmark AI models against various datasets and metrics. The Python SDK provides two HTTP clients (synchronous and asynchronous) powered by [httpx](https://github.com/encode/httpx) and [Pydantic](https://pydantic.dev/) models for type-safe API interactions.

## Quick Start

### Install LayerLens python sdk

Install the layerlens python sdk using the following command

```bash
pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
```

### Generate an api key on the Stratix platform

Login to your organization at [app.layerlens.ai](https://app.layerlens.ai) to generate an api key. Admin users of organizations can generate a keys in the settings page.

Run this command to add your API key to your environment:

```bash
export LAYERLENS_STRATIX_API_KEY="YOUR_API_KEY"
```

### Running an evaluation on the Stratix platform

Before triggering an evaluation using the sdk, login to your organization at [app.layerlens.ai](https://app.layerlens.ai) to ensure that the model and benchmark you are trying to evaluate has been added to your organizations dashboard.

#### Using synchronous client

```python
from layerlens import Stratix

    # Construct sync client
    client = Stratix()

    # --- Models replace with the model key you want to run
    model = client.models.get_by_key("openai/gpt-4o")

    if not model:
      print("Model not found")

    # --- Benchmarks replace with the benchmark name you want to run
    benchmark = client.benchmarks.get_by_key("aime2024")

    if not benchmark:
      print("benchmark not found")

    # --- Create evaluation
    evaluation = client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )
```

#### Using Async Client

```python
import asyncio
from layerlens import AsyncStratix

async def run_evaluation_async():
    # Construct async client
    client = AsyncStratix()

    # --- Model to use
    model = await client.models.get_by_key("openai/gpt-4o")

    if not model:
        print("Model not found")
        return

    # --- Benchmark to use
    benchmark = await client.benchmarks.get_by_key("aime2024")

    if not benchmark:
        print("benchmark not found")
        return

    # --- Create evaluation
    evaluation = await client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )

if __name__ == "__main__":
    asyncio.run(run_evaluation_async())
```

## Next steps

- **[API Reference](api-reference/)** - Complete documentation of all available methods
- **[Code Examples](examples/)** - Practical examples for common use cases
- **[Troubleshooting](troubleshooting/)** - Solutions to common issues
