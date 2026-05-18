"""
Custom & Smart Benchmarks -- LayerLens Python SDK Sample
========================================================

Demonstrates two ways to create project-specific benchmarks:

  1. **Custom benchmark** -- upload a JSONL file of prompt/truth pairs.
  2. **Smart benchmark** -- upload source documents and let AI generate
     evaluation prompts automatically.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python custom_benchmark.py
"""

from __future__ import annotations

from layerlens import Stratix


def main() -> None:
    client = Stratix()

    # ── 1. Create a custom benchmark from a JSONL file ────────────────
    #
    # The JSONL file should have one JSON object per line:
    #   {"input": "What is 2+2?", "truth": "4"}
    #   {"input": "Capital of France?", "truth": "Paris"}
    #
    # Optional fields: "subset" (for grouping prompts)

    print("Creating custom benchmark...")
    result = client.benchmarks.create_custom(
        name="My Custom Benchmark",
        description="A simple test benchmark for QA evaluation",
        file_path="path/to/benchmark.jsonl",
    )

    if result:
        print(f"Custom benchmark created: {result.benchmark_id}")
    else:
        print("Failed to create custom benchmark")

    # Create with additional metrics and input type
    result = client.benchmarks.create_custom(
        name="Advanced Benchmark",
        description="Benchmark with toxicity and readability scoring",
        file_path="path/to/benchmark.jsonl",
        additional_metrics=["toxicity", "readability"],
        input_type="messages",
    )

    if result:
        print(f"Advanced benchmark created: {result.benchmark_id}")

    # ── 2. Create a smart benchmark from source documents ─────────────
    #
    # Smart benchmarks use AI to automatically generate benchmark prompts
    # from your uploaded documents. Supported file types include:
    #   .txt, .pdf, .html, .docx, .csv, .json, .jsonl, .parquet
    #
    # You provide a system prompt that guides how the AI generates
    # evaluation questions from the source material.

    print("\nCreating smart benchmark...")
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

    # ── Verify benchmarks were added to the project ───────────────────
    benchmarks = client.benchmarks.get(type="custom")
    if benchmarks:
        print(f"\nCustom benchmarks in project ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  - {b.name} (id={b.id})")
    else:
        print("\nNo custom benchmarks found in project")


if __name__ == "__main__":
    main()
