#!/usr/bin/env python3

import time

from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Create a judge (no model_id → server uses default model)
judge = client.judges.create(
    name=f"Trace Eval Demo Judge {int(time.time())}",
    evaluation_goal="Evaluate whether the response is accurate, complete, and well-structured",
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Get existing traces to evaluate
traces_response = client.traces.get_many(page_size=3)
if not traces_response or len(traces_response.traces) == 0:
    print("No traces found. Upload some traces first using traces.py")
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

# --- Wait for completion and get results in one call
result = client.trace_evaluations.wait_for_completion(evaluation.id)
if result:
    print(f"  Score: {result.score}, Passed: {result.passed}")
    print(f"  Reasoning: {result.reasoning}")
    if result.steps:
        for step in result.steps:
            print(f"    Tool: {step.tool}, Result: {step.result[:80]}")
else:
    print("  No results returned (evaluation may have failed)")

# --- List all trace evaluations
response = client.trace_evaluations.get_many()
print(f"Found {response.total} trace evaluations")

# --- Clean up
client.judges.delete(judge.id)
print(f"Cleaned up judge {judge.id}")
