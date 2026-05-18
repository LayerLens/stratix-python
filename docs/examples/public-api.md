# Public API

Examples for browsing public models, benchmarks, evaluations, and comparing results using the LayerLens Python SDK.

The public API is accessed through `client.public` on a `Stratix` client, or by instantiating `PublicClient` directly. An API key is still required.

```python
from layerlens import Stratix, PublicClient

# Via the main client
client = Stratix()
public = client.public

# Or directly
public = PublicClient()
```

## Public Models

> Source: [`samples/core/public_catalog.py`](../../samples/core/public_catalog.py)

```python
from layerlens import PublicClient


def main():
    client = PublicClient()

    # --- Browse all public models (first page)
    response = client.models.get(page=1, page_size=10)
    print(f"Found {response.total_count} public models (showing first {len(response.models)})")
    for model in response.models:
        print(f"  - {model.name} ({model.company})")

    # --- Search models by query
    response = client.models.get(query="gpt")
    print(f"\nFound {response.total_count} models matching 'gpt'")
    for model in response.models:
        print(f"  - {model.name}")

    # --- Filter by company
    companies = ["OpenAI", "Anthropic"]
    response = client.models.get(companies=companies)
    print(f"\nFound {response.total_count} models from {companies}")
    for model in response.models:
        print(f"  - {model.name} ({model.company})")

    # --- Filter by region
    response = client.models.get(regions=["usa"])
    print(f"\nFound {response.total_count} models in region 'usa'")

    # --- Filter by category
    response = client.models.get(categories=["open-source"])
    print(f"\nFound {response.total_count} open-source models")

    # --- Sort by release date (newest first)
    response = client.models.get(sort_by="releasedAt", order="desc", page_size=5)
    print(f"\nNewest 5 models:")
    for model in response.models:
        print(f"  - {model.name} (released_at={model.released_at})")

    # --- Include deprecated models
    response = client.models.get(include_deprecated=True)
    print(f"\nTotal models (including deprecated): {response.total_count}")

    # --- Discover available filter values
    response = client.models.get(page=1, page_size=1)
    print(f"\nAvailable filter values:")
    print(f"  Categories: {response.categories}")
    print(f"  Companies:  {response.companies}")
    print(f"  Regions:    {response.regions}")
    print(f"  Licenses:   {response.licenses}")
    print(f"  Sizes:      {response.sizes}")


if __name__ == "__main__":
    main()
```

## Public Benchmarks

> Source: [`samples/core/public_catalog.py`](../../samples/core/public_catalog.py)

```python
from layerlens import PublicClient


def main():
    client = PublicClient()

    # --- Browse all public benchmarks
    response = client.benchmarks.get(page=1, page_size=10)
    print(f"Found {response.total_count} public benchmarks (showing first {len(response.datasets)})")
    for benchmark in response.datasets:
        print(f"  - {benchmark.name} (prompts={benchmark.prompt_count}, language={benchmark.language})")

    # --- Filter by language
    response = client.benchmarks.get(languages=["English"])
    print(f"\nFound {response.total_count} English benchmarks")

    # --- Discover available filter values
    if response.categories:
        print(f"\nAvailable categories: {response.categories}")
    if response.languages:
        print(f"Available languages: {response.languages}")

    # --- Search by name
    response = client.benchmarks.get(query="mmlu")
    print(f"\nFound {response.total_count} benchmarks matching 'mmlu'")
    for benchmark in response.datasets:
        print(f"  - {benchmark.name}: {benchmark.description[:80] if benchmark.description else 'N/A'}...")

    # --- Get benchmark prompts (paginated)
    if response.datasets:
        benchmark = response.datasets[0]
        print(f"\nFetching prompts for '{benchmark.name}' (id={benchmark.id})...")

        prompts_response = client.benchmarks.get_prompts(
            benchmark.id,
            page=1,
            page_size=5,
        )

        if prompts_response:
            print(f"Total prompts: {prompts_response.data.count}")
            print(f"Showing first {len(prompts_response.data.prompts)} prompts:")
            for prompt in prompts_response.data.prompts:
                input_preview = str(prompt.input)[:80]
                truth_preview = prompt.truth[:50] if prompt.truth else "N/A"
                print(f"  - Input: {input_preview}...")
                print(f"    Truth: {truth_preview}")

    # --- Get all prompts (auto-paginates)
    if response.datasets:
        benchmark = response.datasets[0]
        print(f"\nFetching ALL prompts for '{benchmark.name}'...")
        all_prompts = client.benchmarks.get_all_prompts(benchmark.id)
        print(f"Retrieved {len(all_prompts)} total prompts")


if __name__ == "__main__":
    main()
```

## Public Evaluations

> Source: [`samples/core/public_catalog.py`](../../samples/core/public_catalog.py)

```python
from layerlens import PublicClient
from layerlens.models import EvaluationStatus


def main():
    client = PublicClient()

    # --- Get a specific evaluation by ID
    evaluation_id = "699f1426c1212b2d9c78e947"
    evaluation = client.evaluations.get_by_id(evaluation_id)
    if evaluation:
        print(f"Evaluation: {evaluation.id}")
        print(f"  Model: {evaluation.model_name} ({evaluation.model_company})")
        print(f"  Benchmark: {evaluation.benchmark_name}")
        print(f"  Status: {evaluation.status.value}")
        print(f"  Accuracy: {evaluation.accuracy:.2f}%")

        if evaluation.summary:
            print(f"  Summary: {evaluation.summary.name}")
            print(f"  Goal: {evaluation.summary.goal}")
            if evaluation.summary.metrics:
                print(f"  Metrics: {', '.join(m.name for m in evaluation.summary.metrics)}")
            if evaluation.summary.performance_details:
                print(f"  Strengths: {evaluation.summary.performance_details.strengths}")
            if evaluation.summary.analysis_summary:
                print(f"  Key takeaways: {evaluation.summary.analysis_summary.key_takeaways}")
    else:
        print(f"Evaluation {evaluation_id} not found")

    # --- List latest evaluations
    response = client.evaluations.get_many(
        page=1,
        page_size=5,
        sort_by="submittedAt",
        order="desc",
    )
    if response:
        print(f"\nLatest evaluations ({response.pagination.total_count} total):")
        for e in response.evaluations:
            print(f"  - {e.id}: {e.model_name} on {e.benchmark_name} -> {e.accuracy:.2f}% ({e.status.value})")

    # --- Filter by status (only successful)
    response = client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
        page_size=5,
    )
    if response:
        print(f"\nTop successful evaluations ({response.pagination.total_count} total):")
        for e in response.evaluations:
            print(f"  - {e.model_name}: {e.accuracy:.2f}%")


if __name__ == "__main__":
    main()
```

## Comparing Evaluations

> Source: [`samples/core/compare_evaluations.py`](../../samples/core/compare_evaluations.py)

Compare how two models perform on the same benchmark, prompt by prompt.

```python
from layerlens import PublicClient


def main():
    client = PublicClient()

    # --- Compare two models on a benchmark
    # The SDK automatically finds the most recent successful evaluation for each model.
    benchmark_id = "682bddc1e014f9fa440f8a91"  # AIME 2025
    model_id_1 = "699f9761e014f9c3072b0513"    # Qwen3.5 27B
    model_id_2 = "699f9761e014f9c3072b0512"    # Qwen3.5 122B A10B

    print(f"Comparing models on benchmark {benchmark_id}...")
    comparison = client.comparisons.compare_models(
        benchmark_id=benchmark_id,
        model_id_1=model_id_1,
        model_id_2=model_id_2,
        page=1,
        page_size=10,
    )

    if comparison:
        print(f"\n=== Comparison Summary ===")
        print(f"Model 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
        print(f"Model 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")
        print(f"Total compared: {comparison.total_count}")

        if comparison.results:
            print(f"\nFirst {len(comparison.results)} results:")
            for result in comparison.results:
                s1 = "Y" if result.score1 and result.score1 > 0.5 else "N"
                s2 = "Y" if result.score2 and result.score2 > 0.5 else "N"
                print(f"  Prompt: {result.prompt[:80]}...")
                print(f"    Model 1: {s1} (score={result.score1})")
                print(f"    Model 2: {s2} (score={result.score2})")

    # --- Filter: where model 1 fails but model 2 succeeds
    comparison = client.comparisons.compare_models(
        benchmark_id=benchmark_id,
        model_id_1=model_id_1,
        model_id_2=model_id_2,
        outcome_filter="reference_fails",
    )

    if comparison:
        print(f"\n=== Where Model 1 Fails but Model 2 Succeeds ===")
        print(f"Found {comparison.total_count} such cases")

    # --- Compare using evaluation IDs directly
    comparison = client.comparisons.compare(
        evaluation_id_1="699f9938a03d70bf6607081f",
        evaluation_id_2="699f991ca782d00ebd666ba1",
        page=1,
        page_size=5,
    )

    if comparison:
        print(f"\n=== Direct Comparison by Evaluation IDs ===")
        print(f"Model 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
        print(f"Model 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")


if __name__ == "__main__":
    main()
```
