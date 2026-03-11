#!/usr/bin/env python3

"""
Judge Optimizations example.

Note: Optimization requires that the judge has at least 10 annotations
(trace evaluation results). Run trace evaluations first to build up
annotation data before attempting optimization.
"""

import time

import layerlens
from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Fetch a model to use for judge creation
models = client.models.get(type="public", name="gpt-4o")
if not models:
    print("No models found, exiting")
    exit(1)
model = models[0]
print(f"Using model: {model.name} ({model.id})")

# --- Create a judge to optimize
judge = client.judges.create(
    name=f"Optimization Demo Judge {int(time.time())}",
    evaluation_goal="Evaluate whether the response is accurate, complete, and well-structured",
    model_id=model.id,
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Estimate the cost of optimization
estimate = client.judge_optimizations.estimate(
    judge_id=judge.id,
    budget="medium",
)
if estimate:
    print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
    print(f"  Annotations: {estimate.annotation_count}")
    print(f"  Budget: {estimate.budget}")
else:
    print("Could not estimate cost")

# --- Create an optimization run
# Requires at least 10 annotations on the judge.
# If the judge doesn't have enough annotations, the API returns a 400 error.
try:
    run = client.judge_optimizations.create(
        judge_id=judge.id,
        budget="medium",
    )
except layerlens.BadRequestError as e:
    print(f"Cannot start optimization: {e}")
    print("Tip: Run trace evaluations with this judge first to build up annotations.")
    # Demonstrate list and clean up even without a successful run
    response = client.judge_optimizations.get_many(judge_id=judge.id)
    if response:
        print(f"Found {response.total} optimization runs for this judge")
    client.judges.delete(judge.id)
    print(f"Cleaned up judge {judge.id}")
    exit(0)

if not run:
    print("Failed to create optimization run")
    client.judges.delete(judge.id)
    exit(1)
print(f"Created optimization run {run.id}, status: {run.status}")

# --- Poll for completion (optimization can take a while)
optimization = None
print("Waiting for optimization to complete...")
for i in range(60):
    optimization = client.judge_optimizations.get(run.id)
    if not optimization:
        print("  Could not fetch optimization run")
        break
    print(f"  [{i * 5}s] Status: {optimization.status}")
    if optimization.status.value in ("success", "failure"):
        print(f"  Baseline accuracy: {optimization.baseline_accuracy}")
        print(f"  Optimized accuracy: {optimization.optimized_accuracy}")
        if optimization.original_goal:
            print(f"  Original goal: {optimization.original_goal[:80]}...")
        if optimization.optimized_goal:
            print(f"  Optimized goal: {optimization.optimized_goal[:80]}...")
        print(f"  Actual cost: ${optimization.actual_cost:.4f}")
        break
    time.sleep(5)

# --- List optimization runs for this judge
response = client.judge_optimizations.get_many(judge_id=judge.id)
if response:
    print(f"Found {response.total} optimization runs for this judge")
    for r in response.optimization_runs:
        print(f"  - {r.id}: {r.status} (budget: {r.budget})")

# --- Apply optimization results (only if optimization succeeded)
if optimization and optimization.status.value == "success":
    result = client.judge_optimizations.apply(run.id)
    if result:
        print(f"Applied optimization to judge {result.judge_id}")
        print(f"  New version: v{result.new_version}")
        print(f"  {result.message}")
    else:
        print("Could not apply optimization result")
else:
    print("Skipping apply (optimization did not succeed)")

# --- Clean up
client.judges.delete(judge.id)
print(f"Cleaned up judge {judge.id}")
