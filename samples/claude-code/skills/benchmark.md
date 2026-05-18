---
name: benchmark
description: Manage models, benchmarks, and comparisons in LayerLens
user_invocable: true
---

You are helping the user manage models, benchmarks, and comparisons in the LayerLens platform using the Python SDK.

## SDK Reference: Models

```python
from layerlens import Stratix
client = Stratix()

# List models (project-scoped or public catalog)
models = client.models.get(
    type="public",          # "public" or "custom"
    name="gpt-4o",          # filter by name
    key="gpt-4o",           # filter by key
    companies=["openai"],   # filter by company
    regions=["us"],         # filter by region
    licenses=["proprietary"], # filter by license
)
# Returns List[Model], each with .id, .name, .key

# Look up a specific model
model = client.models.get_by_id("model_id")
model = client.models.get_by_key("gpt-4o")

# Add public models to your project
client.models.add("model_id_1", "model_id_2")

# Remove models from your project
client.models.remove("model_id_1", "model_id_2")

# Create a custom model endpoint
model = client.models.create_custom(
    name="My Custom Model",
    key="my-custom-model",
    description="Fine-tuned GPT for legal analysis",
    api_url="https://api.example.com/v1/completions",
    max_tokens=4096,
    api_key="sk-...",
)

# Browse the public catalog
public_models = client.public.models.get()
```

## SDK Reference: Benchmarks

```python
# List benchmarks in the project
benchmarks = client.benchmarks.get(
    type="public",     # "public" or "custom"
    name="MMLU",       # filter by name
    key="mmlu",        # filter by key
)
# Returns List[Benchmark], each with .id, .name, .key

# Look up a specific benchmark
benchmark = client.benchmarks.get_by_id("benchmark_id")
benchmark = client.benchmarks.get_by_key("mmlu")

# Add public benchmarks to your project
client.benchmarks.add("benchmark_id_1", "benchmark_id_2")

# Remove benchmarks from your project
client.benchmarks.remove("benchmark_id_1")

# Create a custom benchmark from a JSONL dataset
benchmark = client.benchmarks.create_custom(
    name="My Custom Benchmark",
    description="Domain-specific QA evaluation",
    file_path="path/to/dataset.jsonl",
    additional_metrics=["f1", "bleu"],        # optional extra metrics
    custom_scorer_ids=["scorer_id"],          # optional custom scorers
    input_type="text",                        # input type
)

# Create a smart benchmark (AI-assisted generation)
benchmark = client.benchmarks.create_smart(
    name="Smart Legal Benchmark",
    description="Evaluates legal reasoning capabilities",
    system_prompt="You are a legal expert...",
    file_paths=["reference_doc1.pdf", "reference_doc2.pdf"],
    metrics=["accuracy", "relevance"],
)

# Browse the public catalog
public_benchmarks = client.public.benchmarks.get()
```

## SDK Reference: Comparisons

```python
# Compare two evaluations side-by-side
comparison = client.public.comparisons.compare(
    evaluation_id_1="eval_id_1",
    evaluation_id_2="eval_id_2",
    outcome_filter="disagree",    # optional: filter to disagreements
    search="query text",          # optional: text search
)

# Compare two models on the same benchmark
comparison = client.public.comparisons.compare_models(
    benchmark_id="benchmark_id",
    model_id_1="model_id_1",
    model_id_2="model_id_2",
    outcome_filter="disagree",    # optional
    search="query text",          # optional
)
```

## Instructions

When the user asks to work with models, benchmarks, or comparisons:

### Browsing and adding models/benchmarks
1. Use `client.public.models.get()` or `client.public.benchmarks.get()` to browse the full catalog.
2. Use `client.models.get()` or `client.benchmarks.get()` to see what is already in the project.
3. Use `.add()` to add catalog items to the project and `.remove()` to remove them.

### Creating custom benchmarks
1. For a dataset-based benchmark: prepare a JSONL file and use `client.benchmarks.create_custom()`.
2. For an AI-generated benchmark: use `client.benchmarks.create_smart()` with a system prompt and optional reference files.
3. Sample dataset files are in `samples/data/datasets/` and `samples/data/industry/`.

### Comparing evaluations or models
1. To compare two evaluation runs: use `client.public.comparisons.compare()` with both evaluation IDs.
2. To compare two models on the same benchmark: use `client.public.comparisons.compare_models()`.
3. Use `outcome_filter="disagree"` to focus on cases where the two differ.

See `samples/core/model_benchmark_management.py` for model/benchmark CRUD, `samples/core/run_evaluation.py` for running evaluations, and `samples/core/compare_evaluations.py` for comparisons.
