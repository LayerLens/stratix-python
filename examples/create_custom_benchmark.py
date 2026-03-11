#!/usr/bin/env python3

from layerlens import Stratix


def main():
    # Construct client (API key from env or inline)
    client = Stratix()

    # --- Create a custom benchmark from a JSONL file
    #
    # The JSONL file should have one JSON object per line with these fields:
    #   {"input": "What is 2+2?", "truth": "4"}
    #   {"input": "Capital of France?", "truth": "Paris"}
    #
    # Optional fields: "subset" (for grouping prompts)

    result = client.benchmarks.create_custom(
        name="My Custom Benchmark",
        description="A simple test benchmark for QA evaluation",
        file_path="path/to/benchmark.jsonl",
    )

    if result:
        print(f"Custom benchmark created: {result.benchmark_id}")
    else:
        print("Failed to create custom benchmark")

    # --- Create with additional metrics and input type
    result = client.benchmarks.create_custom(
        name="Advanced Benchmark",
        description="Benchmark with toxicity and readability scoring",
        file_path="path/to/benchmark.jsonl",
        additional_metrics=["toxicity", "readability"],
        input_type="messages",
    )

    if result:
        print(f"Advanced benchmark created: {result.benchmark_id}")

    # --- Verify the benchmark was added to the project
    benchmarks = client.benchmarks.get(type="custom")
    if benchmarks:
        print(f"\nCustom benchmarks in project ({len(benchmarks)}):")
        for b in benchmarks:
            print(f"  - {b.name} (id={b.id})")


if __name__ == "__main__":
    main()
