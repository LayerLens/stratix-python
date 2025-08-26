# Layerlens Python SDK Documentation

Welcome to the official documentation for the Layerlens Python SDK for the atlas platform. This library provides convenient programmatic to the Atlas platform from any Python 3.8+ application.

## What is Atlas?

Atlas is an evaluation platform that allows you to benchmark AI models against various datasets and metrics. The Python SDK provides two HTTP clients (syncronous and asynchronous) powered by [httpx](https://github.com/encode/httpx) and [Pydantic](https://pydantic.dev/) models for type-safe API interactions.

## Quick Start

### Install LayerLens python sdk
Install the layerlens python sdk using the following command
```bash
pip install layerlens --index-url https://sdk.layerlens.ai
```

### Generate an api key on the atlas platform

Login to your organization at [app.layerlens.ai](https://app.layerlens.ai) to generate an api key. Admin users of organizations can generate a keys in the settings page.

Run this command to add your API key to your environment:

```bash
export LAYERLENS_ATLAS_API_KEY="YOUR_API_KEY"
```

### Running an evaluation on the atlas platform

Before triggering an evaluation using the sdk, login to your organization at [app.layerlens.ai](https://app.layerlens.ai) to ensure that the model and benchmark you are trying to evaluate has been added to your organizations dashboard.

#### Using synchronous client

```python
from atlas import Atlas

    # Construct sync client
    client = Atlas()

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

    # --- Create evaluation
    evaluation = client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )
```


#### Using Async Client

```python
import asyncio
from atlas import AsyncAtlas

async def run_evaluation_async():
    # Construct async client
    client = AsyncAtlas()

    # --- Models replace with the model name you want to run
    models = await client.models.get(type="public",name="gpt-4o")
    print(f"Models found: {models}")

    if not models:
        print("gpt-4o not found, exiting")
        return

    model = models[0]
    # --- Benchmarks replace with the benchmark name you want to run
    benchmarks = await client.benchmarks.get(type="public", name="simpleQA")

    if not benchmarks:
        print("SimpleQA benchmark not found, exiting")
        return

    benchmark = benchmarks[0]

    # --- Create evaluation
    evaluation = await client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Next steps

- **[API Reference](api-reference/)** - Complete documentation of all available methods
- **[Code Examples](examples/)** - Practical examples for common use cases
- **[Troubleshooting](troubleshooting/)** - Solutions to common issues
