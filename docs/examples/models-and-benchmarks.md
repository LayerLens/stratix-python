# Models and Benchmarks

Examples for browsing, filtering, creating, and managing models and benchmarks using the LayerLens Python SDK.

## Filtering Models

> Source: [`examples/get_models.py`](../../examples/get_models.py)

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    # --- Filter by name
    model_name = "gpt-4o"
    models = await client.models.get(name=model_name)
    print(f"Found {len(models)} models with name {model_name}")

    # --- Filter by company
    company_names = ["openai", "anthropic"]
    models = await client.models.get(companies=company_names)
    print(f"Found {len(models)} models with companies {company_names}")

    # --- Filter by region
    region_names = ["usa"]
    models = await client.models.get(regions=region_names)
    print(f"Found {len(models)} models with regions {region_names}")

    # --- Filter by type
    model_type = "public"
    models = await client.models.get(type=model_type)
    print(f"Found {len(models)} models with type {model_type}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Filtering Benchmarks

> Source: [`examples/get_benchmarks.py`](../../examples/get_benchmarks.py)

```python
import asyncio

from layerlens import AsyncStratix


async def main():
    client = AsyncStratix()

    # --- Filter by name
    benchmark_name = "mmlu"
    benchmarks = await client.benchmarks.get(name=benchmark_name)
    print(f"Found {len(benchmarks)} benchmarks with name {benchmark_name}")

    # --- Filter by type
    benchmark_type = "public"
    benchmarks = await client.benchmarks.get(type=benchmark_type)
    print(f"Found {len(benchmarks)} benchmarks with type {benchmark_type}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Creating a Custom Model

> Source: [`examples/create_custom_model.py`](../../examples/create_custom_model.py)

Custom models let you evaluate any model accessible via an OpenAI-compatible chat completions endpoint.

```python
import os

from layerlens import Stratix


def main():
    client = Stratix()

    result = client.models.create_custom(
        name="My Custom Model",
        key="my-org/custom-model-v1",
        description="Custom fine-tuned model served via vLLM",
        api_url="https://my-model-endpoint.example.com/v1",
        api_key=os.environ["MY_PROVIDER_API_KEY"],
        max_tokens=4096,
    )

    if result:
        print(f"Custom model created: {result.model_id}")
    else:
        print("Failed to create custom model")

    # Verify the model was added
    models = client.models.get(type="custom")
    if models:
        print(f"\nCustom models in project ({len(models)}):")
        for m in models:
            print(f"  - {m.name} (id={m.id}, key={m.key})")


if __name__ == "__main__":
    main()
```

## Creating a Custom Benchmark

> Source: [`examples/create_custom_benchmark.py`](../../examples/create_custom_benchmark.py)

Custom benchmarks are created from JSONL files with `input` and `truth` fields.

```python
from layerlens import Stratix


def main():
    client = Stratix()

    # Basic custom benchmark
    result = client.benchmarks.create_custom(
        name="My Custom Benchmark",
        description="A simple test benchmark for QA evaluation",
        file_path="path/to/benchmark.jsonl",
    )

    if result:
        print(f"Custom benchmark created: {result.benchmark_id}")

    # With additional metrics and input type
    result = client.benchmarks.create_custom(
        name="Advanced Benchmark",
        description="Benchmark with toxicity and readability scoring",
        file_path="path/to/benchmark.jsonl",
        additional_metrics=["toxicity", "readability"],
        input_type="messages",
    )

    if result:
        print(f"Advanced benchmark created: {result.benchmark_id}")

    # Verify
    benchmarks = client.benchmarks.get(type="custom")
    if benchmarks:
        print(f"\nCustom benchmarks in project ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  - {b.name} (id={b.id})")


if __name__ == "__main__":
    main()
```

### JSONL File Format

Each line should be a JSON object:

```json
{"input": "What is 2+2?", "truth": "4"}
{"input": "Capital of France?", "truth": "Paris"}
```

Optional field: `subset` (for grouping prompts into categories).

## Creating a Smart Benchmark

> Source: [`examples/create_smart_benchmark.py`](../../examples/create_smart_benchmark.py)

Smart benchmarks use AI to automatically generate benchmark prompts from uploaded documents. Supported file types: `.txt`, `.pdf`, `.html`, `.docx`, `.csv`, `.json`, `.jsonl`, `.parquet`.

```python
from layerlens import Stratix


def main():
    client = Stratix()

    result = client.benchmarks.create_smart(
        name="Product Knowledge Benchmark",
        description="Evaluates model knowledge of our product documentation",
        system_prompt=(
            "Generate question-answer pairs that test understanding of the "
            "product features, capabilities, and limitations described in "
            "the provided documents. Each question should have a clear, "
            "factual answer derived from the source material."
        ),
        file_paths=[
            "path/to/product_docs.pdf",
            "path/to/faq.txt",
        ],
        metrics=["hallucination"],
    )

    if result:
        print(f"Smart benchmark created: {result.benchmark_id}")
        print("The benchmark is being generated asynchronously.")
        print("Check the dashboard for progress.")
    else:
        print("Failed to create smart benchmark")


if __name__ == "__main__":
    main()
```

## Managing Project Models and Benchmarks

> Source: [`examples/manage_project_models_benchmarks.py`](../../examples/manage_project_models_benchmarks.py)

Add and remove public models and benchmarks from your project.

```python
from layerlens import Stratix


def main():
    client = Stratix()

    # --- Add public models to the project
    success = client.models.add("model-id-1", "model-id-2")
    print(f"Add models: {'success' if success else 'failed'}")

    # --- Remove a model from the project
    success = client.models.remove("model-id-1")
    print(f"Remove model: {'success' if success else 'failed'}")

    # --- Add public benchmarks to the project
    success = client.benchmarks.add("benchmark-id-1")
    print(f"Add benchmark: {'success' if success else 'failed'}")

    # --- Remove a benchmark from the project
    success = client.benchmarks.remove("benchmark-id-1")
    print(f"Remove benchmark: {'success' if success else 'failed'}")

    # --- List current models and benchmarks
    models = client.models.get()
    if models:
        print(f"\nModels in project ({len(models)}):")
        for m in models:
            print(f"  - {m.name} (id={m.id})")

    benchmarks = client.benchmarks.get()
    if benchmarks:
        print(f"\nBenchmarks in project ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  - {b.name} (id={b.id})")


if __name__ == "__main__":
    main()
```
