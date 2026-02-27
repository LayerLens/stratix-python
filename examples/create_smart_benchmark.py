#!/usr/bin/env -S poetry run python

from layerlens import Stratix


def main():
    # Construct client (API key from env or inline)
    client = Stratix()

    # --- Create a smart benchmark from source files
    #
    # Smart benchmarks use AI to automatically generate benchmark prompts
    # from your uploaded documents. Supported file types include:
    #   .txt, .pdf, .html, .docx, .csv, .json, .jsonl, .parquet
    #
    # You provide a system prompt that guides how the AI generates
    # evaluation questions from the source material.

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

    # --- Verify the benchmark was added to the project
    benchmarks = client.benchmarks.get(type="custom")
    if benchmarks:
        print(f"\nCustom benchmarks in project ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  - {b.name} (id={b.id})")


if __name__ == "__main__":
    main()
