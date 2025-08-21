#!/usr/bin/env -S poetry run python

from atlas import Atlas

# Construct sync client (API key from env or inline)
client = Atlas()

# --- Models
models = client.models.get()
print(f"Found {len(models)} models")

# --- Benchmarks
benchmarks = client.benchmarks.get()
print(f"Found {len(benchmarks)} benchmarks")

# --- Create evaluation
evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0],
)
print(f"Created evaluation {evaluation.id}, status={evaluation.status}")

# --- Wait for completion
evaluation = client.evaluations.wait_for_completion(
    evaluation,
    interval_seconds=10,
    timeout_seconds=600,  # 10 minutes
)
print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

# --- Results
if evaluation.is_success:
    results = client.results.get(evaluation=evaluation)
    print("Results:", results)
else:
    print("Evaluation did not succeed, no results to show.")
