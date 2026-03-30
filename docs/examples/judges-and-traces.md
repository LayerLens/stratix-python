# Judges and Traces

Examples for working with judges, traces, and trace evaluations on the Stratix platform using the LayerLens Python SDK.

## Creating and Managing Judges

> Source: [`examples/judges.py`](../../examples/judges.py)

```python
import time

from layerlens import Stratix

client = Stratix()

# Fetch a model to use as the judge's LLM
models = client.models.get(type="public", name="gpt-4o")
model = models[0]
print(f"Using model: {model.name} ({model.id})")

# --- Create a judge
judge = client.judges.create(
    name=f"Code Quality Judge {int(time.time())}",
    evaluation_goal="Evaluate the quality of code output including correctness, readability, and style",
    model_id=model.id,
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Get a judge by ID
judge = client.judges.get(judge.id)
print(f"Judge: {judge.name}, version: {judge.version}")

# --- List all judges
response = client.judges.get_many()
print(f"Found {response.total_count} judges")
for j in response.judges:
    print(f"  - {j.name} (v{j.version}, {j.run_count} runs)")

# --- Update a judge (creates a new version)
updated = client.judges.update(
    judge.id,
    name="Updated Code Quality Judge",
    evaluation_goal="Evaluate code output for correctness, readability, style, and security",
)
print(f"Updated judge {updated.id}")

# --- Delete a judge
deleted = client.judges.delete(judge.id)
print(f"Deleted judge {deleted.id}")
```

## Uploading and Managing Traces

> Source: [`examples/traces.py`](../../examples/traces.py)

```python
import os

from layerlens import Stratix

client = Stratix()

# --- Upload traces from a file
traces_file = os.path.join(os.path.dirname(__file__), "traces.jsonl")
result = client.traces.upload(traces_file)
print(f"Uploaded {len(result.trace_ids)} traces")

# --- List traces
response = client.traces.get_many()
print(f"Found {response.total_count} traces")
for trace in response.traces[:5]:
    print(f"  - {trace.id}: {trace.filename}")

# --- List traces with filters
filtered = client.traces.get_many(
    sort_by="created_at",
    sort_order="desc",
    page_size=10,
)
print(f"Filtered traces: {filtered.count}")

# --- Get a single trace
trace = client.traces.get(result.trace_ids[0])
print(f"Trace {trace.id}: {len(trace.data)} data keys")

# --- Get available sources
sources = client.traces.get_sources()
print(f"Sources: {sources}")

# --- Delete a trace
deleted = client.traces.delete(trace.id)
print(f"Deleted: {deleted}")
```

## Running Trace Evaluations

> Source: [`examples/trace_evaluations.py`](../../examples/trace_evaluations.py)

```python
import time

from layerlens import Stratix

client = Stratix()

# Create a judge (no model_id → server uses default model)
judge = client.judges.create(
    name=f"Trace Eval Demo Judge {int(time.time())}",
    evaluation_goal="Evaluate whether the response is accurate, complete, and well-structured",
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Get existing traces to evaluate
traces_response = client.traces.get_many(page_size=3)
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

# --- Wait for completion and get results
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
```

## Judge Optimizations

> Source: [`examples/judge_optimizations.py`](../../examples/judge_optimizations.py)

Optimization requires that the judge has at least 10 annotations (trace evaluation results). Run trace evaluations first to build up annotation data.

```python
import time

import layerlens
from layerlens import Stratix

client = Stratix()

models = client.models.get(type="public", name="gpt-4o")
model = models[0]

judge = client.judges.create(
    name=f"Optimization Demo Judge {int(time.time())}",
    evaluation_goal="Evaluate whether the response is accurate, complete, and well-structured",
    model_id=model.id,
)

# --- Estimate cost
estimate = client.judge_optimizations.estimate(
    judge_id=judge.id,
    budget="medium",
)
if estimate:
    print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
    print(f"  Annotations: {estimate.annotation_count}, Budget: {estimate.budget}")

# --- Create an optimization run
try:
    run = client.judge_optimizations.create(
        judge_id=judge.id,
        budget="medium",
    )
except layerlens.BadRequestError as e:
    print(f"Cannot start optimization: {e}")
    print("Tip: Run trace evaluations with this judge first to build up annotations.")
    client.judges.delete(judge.id)
    exit(0)

# --- Poll for completion
optimization = None
for i in range(60):
    optimization = client.judge_optimizations.get(run.id)
    if not optimization:
        break
    print(f"  [{i * 5}s] Status: {optimization.status}")
    if optimization.status.value in ("success", "failure"):
        print(f"  Baseline accuracy: {optimization.baseline_accuracy}")
        print(f"  Optimized accuracy: {optimization.optimized_accuracy}")
        break
    time.sleep(5)

# --- List optimization runs
response = client.judge_optimizations.get_many(judge_id=judge.id)
if response:
    print(f"Found {response.total} optimization runs")

# --- Apply optimization results
if optimization and optimization.status.value == "success":
    result = client.judge_optimizations.apply(run.id)
    if result:
        print(f"Applied optimization: new version v{result.new_version}")

client.judges.delete(judge.id)
```

## Async Judges and Traces

> Source: [`examples/async_judges_and_traces.py`](../../examples/async_judges_and_traces.py)

```python
import os
import time
import asyncio

from layerlens import Stratix, AsyncStratix


async def main():
    # Fetch a model using sync client
    sync_client = Stratix()
    models = sync_client.models.get(type="public", name="gpt-4o")
    model = models[0]

    client = AsyncStratix()

    # --- Create a judge
    judge = await client.judges.create(
        name=f"Response Quality Judge {int(time.time())}",
        evaluation_goal="Evaluate whether the response is accurate, helpful, and well-structured",
        model_id=model.id,
    )
    print(f"Created judge {judge.id}: {judge.name}")

    # --- Upload traces
    traces_file = os.path.join(os.path.dirname(__file__), "traces.jsonl")
    result = await client.traces.upload(traces_file)
    print(f"Uploaded {len(result.trace_ids)} traces")

    # --- List traces
    traces_response = await client.traces.get_many(page_size=10)
    trace_ids = [t.id for t in traces_response.traces[:5]]

    # --- Estimate cost
    estimate = await client.trace_evaluations.estimate_cost(
        trace_ids=trace_ids,
        judge_id=judge.id,
    )
    print(f"Estimated cost: ${estimate.estimated_cost:.4f}")

    # --- Run evaluations concurrently
    tasks = [client.trace_evaluations.create(trace_id=tid, judge_id=judge.id) for tid in trace_ids]
    evaluations = await asyncio.gather(*tasks)

    for evaluation in evaluations:
        if evaluation:
            print(f"  Evaluation {evaluation.id}: {evaluation.status}")

    # --- Wait for results concurrently
    result_tasks = [
        client.trace_evaluations.wait_for_completion(e.id)
        for e in evaluations if e
    ]
    results = await asyncio.gather(*result_tasks)
    for result in results:
        if result:
            print(f"  Score: {result.score}, Passed: {result.passed}")
        else:
            print(f"  No results (evaluation may have failed)")

    await client.judges.delete(judge.id)


if __name__ == "__main__":
    asyncio.run(main())
```

## See Also

- [Models and Benchmarks](models-and-benchmarks.md) - Custom models, custom/smart benchmarks, project management
- [Public API](public-api.md) - Public models, benchmarks, evaluations, and comparisons

## Error Handling

```python
from layerlens import Stratix
import layerlens

client = Stratix()

try:
    models = client.models.get(type="public", name="gpt-4o")
    model = models[0]

    judge = client.judges.create(
        name="My Judge",
        evaluation_goal="Evaluate output quality",
        model_id=model.id,
    )

    evaluation = client.trace_evaluations.create(
        trace_id="trace-abc",
        judge_id=judge.id,
    )

except layerlens.AuthenticationError:
    print("Check your API key")
except layerlens.NotFoundError:
    print("Trace or judge not found")
except layerlens.BadRequestError as e:
    print(f"Invalid request: {e}")
except layerlens.APIError as e:
    print(f"API error: {e}")
```
