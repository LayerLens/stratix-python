"""
Public Catalog -- LayerLens Python SDK Sample
=============================================

Demonstrates the public (unauthenticated) catalog APIs:

  1. Browse and search public models with filters.
  2. Browse and search public benchmarks with filters.
  3. List public evaluations with sorting and status filters.
  4. Download benchmark prompts (paginated and all-at-once).

These endpoints are read-only and do not require a project API key;
however, a valid ``LAYERLENS_STRATIX_API_KEY`` must still be set.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python public_catalog.py
"""

from __future__ import annotations

from layerlens import PublicClient
from layerlens.models import EvaluationStatus


def main() -> None:
    client = PublicClient()

    # ── Public Models ─────────────────────────────────────────────────
    print("=" * 60)
    print("PUBLIC MODELS")
    print("=" * 60)

    # Browse first page
    response = client.models.get(page=1, page_size=10)
    print(f"Total public models: {response.total_count} (showing first {len(response.models)})")
    for model in response.models:
        print(f"  - {model.name} ({model.company})")

    # Search by query
    response = client.models.get(query="gpt")
    print(f"\nModels matching 'gpt': {response.total_count}")
    for model in response.models:
        print(f"  - {model.name}")

    # Filter by company
    companies = ["OpenAI", "Anthropic"]
    response = client.models.get(companies=companies)
    print(f"\nModels from {companies}: {response.total_count}")

    # Filter by region
    response = client.models.get(regions=["usa"])
    print(f"Models in region 'usa': {response.total_count}")

    # Filter by category
    response = client.models.get(categories=["open-source"])
    print(f"Open-source models: {response.total_count}")

    # Sort by release date (newest first)
    response = client.models.get(sort_by="releasedAt", order="desc", page_size=5)
    print(f"\nNewest 5 models:")
    for model in response.models:
        print(f"  - {model.name} (released_at={model.released_at})")

    # Include deprecated models
    response = client.models.get(include_deprecated=True)
    print(f"\nTotal models (including deprecated): {response.total_count}")

    # Discover available filter values
    response = client.models.get(page=1, page_size=1)
    print(f"\nAvailable filter values:")
    print(f"  Categories: {response.categories}")
    print(f"  Companies:  {response.companies}")
    print(f"  Regions:    {response.regions}")
    print(f"  Licenses:   {response.licenses}")
    print(f"  Sizes:      {response.sizes}")

    # ── Public Benchmarks ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PUBLIC BENCHMARKS")
    print("=" * 60)

    # Browse first page
    response = client.benchmarks.get(page=1, page_size=10)
    print(f"Total public benchmarks: {response.total_count} (showing first {len(response.datasets)})")
    for benchmark in response.datasets:
        print(f"  - {benchmark.name} (prompts={benchmark.prompt_count}, language={benchmark.language})")

    # Filter by language
    response = client.benchmarks.get(languages=["English"])
    print(f"\nEnglish benchmarks: {response.total_count}")

    # Show available filter categories
    if response.categories:
        print(f"Available categories: {response.categories}")
    if response.languages:
        print(f"Available languages: {response.languages}")

    # Search by name
    response = client.benchmarks.get(query="mmlu")
    print(f"\nBenchmarks matching 'mmlu': {response.total_count}")
    for benchmark in response.datasets:
        desc = benchmark.description[:80] if benchmark.description else "N/A"
        print(f"  - {benchmark.name}: {desc}...")

    # Download benchmark prompts (paginated)
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

        # Fetch all prompts (auto-paginates)
        print(f"\nFetching ALL prompts for '{benchmark.name}'...")
        all_prompts = client.benchmarks.get_all_prompts(benchmark.id)
        print(f"Retrieved {len(all_prompts)} total prompts")

    # ── Public Evaluations ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PUBLIC EVALUATIONS")
    print("=" * 60)

    # List latest evaluations
    response = client.evaluations.get_many(
        page=1,
        page_size=5,
        sort_by="submittedAt",
        order="desc",
    )
    if response:
        print(f"Latest evaluations ({response.pagination.total_count} total):")
        for e in response.evaluations:
            print(f"  - {e.id}: {e.model_name} on {e.benchmark_name} -> {e.accuracy:.2f}% ({e.status.value})")

    # Filter by status (only successful)
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

    # Get a specific evaluation by ID (if available from the listing)
    if response and response.evaluations:
        eval_id = response.evaluations[0].id
        evaluation = client.evaluations.get_by_id(eval_id)
        if evaluation:
            print(f"\nEvaluation detail: {evaluation.id}")
            print(f"  Model: {evaluation.model_name} ({evaluation.model_company})")
            print(f"  Benchmark: {evaluation.benchmark_name}")
            print(f"  Status: {evaluation.status.value}")
            print(f"  Accuracy: {evaluation.accuracy:.2f}%")
            if evaluation.summary:
                print(f"  Summary: {evaluation.summary.name}")
                print(f"  Goal: {evaluation.summary.goal}")
                if evaluation.summary.metrics:
                    print(f"  Metrics: {', '.join(m.name for m in evaluation.summary.metrics)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
