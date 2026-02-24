#!/usr/bin/env -S poetry run python

import time

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

# --- Create a judge to use for evaluations
judge = client.judges.create(
    name=f"Trace Eval Demo Judge {int(time.time())}",
    evaluation_goal="Evaluate whether the response is accurate, complete, and well-structured",
    model_id=model.id,
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Get existing traces to evaluate
traces_response = client.traces.get_many(page_size=3)
if not traces_response or len(traces_response.traces) == 0:
    print("No traces found. Upload some traces first using traces.py")
    # Clean up the judge
    client.judges.delete(judge.id)
    exit(1)

trace_ids = [t.id for t in traces_response.traces]
print(f"Found {len(trace_ids)} traces to evaluate")

# --- Estimate cost before running
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=trace_ids,
    judge_id=judge.id,
)
print(f"Estimated cost: ${estimate.estimated_cost:.4f} for {estimate.trace_count} traces")

# --- Run a judge on the first trace
evaluation = client.trace_evaluations.create(
    trace_id=trace_ids[0],
    judge_id=judge.id,
)
print(f"Created evaluation {evaluation.id}, status: {evaluation.status}")

# --- Wait for evaluation to complete (poll every 2 seconds, up to 60s)
for _ in range(30):
    evaluation = client.trace_evaluations.get(evaluation.id)
    print(f"Evaluation status: {evaluation.status}")
    if evaluation.status.value in ("success", "failure"):
        break
    time.sleep(2)

# --- Get evaluation results (may 404 if still in progress)
try:
    results_response = client.trace_evaluations.get_results(evaluation.id)
    if results_response and results_response.results:
        for result in results_response.results:
            print(f"  Score: {result.score}, Passed: {result.passed}")
            print(f"  Reasoning: {result.reasoning}")
            if result.steps:
                for step in result.steps:
                    print(f"    Step {step.step}: {step.reasoning}")
    else:
        print("  No results returned")
except Exception:
    print("  No results yet (evaluation may still be in progress)")

# --- List all trace evaluations
response = client.trace_evaluations.get_many()
print(f"Found {response.total} trace evaluations")

# --- Clean up
client.judges.delete(judge.id)
print(f"Cleaned up judge {judge.id}")
