#!/usr/bin/env python3

from layerlens import Stratix


def main():
    # Construct client (API key from env or inline)
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
