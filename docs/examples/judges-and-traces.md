# Judges and Traces

Examples for working with judges, traces, and trace evaluations on the Stratix platform using the Layerlens Python SDK.

## Creating and Managing Judges

### Basic Judge CRUD

```python
from layerlens import Stratix

client = Stratix()

# Create a judge
judge = client.judges.create(
    name="Code Quality Judge",
    evaluation_goal="Evaluate the quality of code output including correctness, readability, and style",
)
print(f"Created judge: {judge.name} (v{judge.version})")

# Get a judge by ID
judge = client.judges.get(judge.id)

# List all judges with pagination
response = client.judges.get_many(page=1, page_size=50)
for j in response.judges:
    print(f"  {j.name}: v{j.version}, {j.run_count} runs")

# Update a judge (creates a new version)
client.judges.update(
    judge.id,
    evaluation_goal="Evaluate code for correctness, readability, style, and security",
)

# Delete a judge
client.judges.delete(judge.id)
```

## Uploading and Managing Traces

### Upload Trace Files

```python
from layerlens import Stratix

client = Stratix()

# Upload a JSONL file containing multiple traces
result = client.traces.upload("./traces.jsonl")
print(f"Uploaded {result.count} traces")

# Upload a single JSON trace
result = client.traces.upload("./single-trace.json")
```

### Browse and Filter Traces

```python
from layerlens import Stratix

client = Stratix()

# List all traces
response = client.traces.get_many()
print(f"Total traces: {response.total_count}")

# Filter by source and time range
response = client.traces.get_many(
    source="upload",
    time_range="7d",
    sort_by="created_at",
    sort_order="desc",
)

# Search traces
response = client.traces.get_many(search="authentication")

# Get available sources
sources = client.traces.get_sources()
print(f"Sources: {sources}")
```

### Get Trace Details

```python
from layerlens import Stratix

client = Stratix()

trace = client.traces.get("trace-abc123")
if trace:
    print(f"Filename: {trace.filename}")
    print(f"Created: {trace.created_at}")
    print(f"Data keys: {list(trace.data.keys())}")
```

## Running Trace Evaluations

### Estimate Cost Before Running

```python
from layerlens import Stratix

client = Stratix()

# Get trace IDs to evaluate
traces_response = client.traces.get_many(page_size=10)
trace_ids = [t.id for t in traces_response.traces]

# Estimate cost
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=trace_ids,
    judge_id="judge-123",
)
print(f"Cost for {estimate.trace_count} traces: ${estimate.estimated_cost:.4f}")
print(f"Model: {estimate.model}")
```

### Run a Judge on a Trace

```python
from layerlens import Stratix

client = Stratix()

# Create an evaluation
evaluation = client.trace_evaluations.create(
    trace_id="trace-abc",
    judge_id="judge-xyz",
)
print(f"Evaluation {evaluation.id}: {evaluation.status}")

# Get results
results_response = client.trace_evaluations.get_results(evaluation.id)
if results_response:
    for result in results_response.results:
        print(f"Score: {result.score}, Passed: {result.passed}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Latency: {result.latency_ms}ms, Cost: ${result.total_cost:.4f}")
        for step in result.steps:
            print(f"  Step {step.step}: {step.reasoning}")
```

### Browse Evaluation Results

```python
from layerlens import Stratix

client = Stratix()

# List all evaluations
response = client.trace_evaluations.get_many()
print(f"Total evaluations: {response.total}")

# Filter by judge and outcome
response = client.trace_evaluations.get_many(
    judge_id="judge-123",
    outcome="pass",
    sort_by="created_at",
    sort_order="desc",
)

# Filter by trace
response = client.trace_evaluations.get_many(
    trace_id="trace-abc",
)
```

## Async Workflows

### Run Evaluations Concurrently

```python
import asyncio
from layerlens import AsyncStratix

async def main():
    client = AsyncStratix()

    # Create a judge
    judge = await client.judges.create(
        name="Response Quality Judge",
        evaluation_goal="Evaluate whether the response is accurate and well-structured",
    )

    # Upload traces
    result = await client.traces.upload("./traces.jsonl")
    print(f"Uploaded {result.count} traces")

    # Get traces to evaluate
    traces_response = await client.traces.get_many(page_size=5)
    trace_ids = [t.id for t in traces_response.traces]

    # Run evaluations concurrently
    tasks = [
        client.trace_evaluations.create(trace_id=tid, judge_id=judge.id)
        for tid in trace_ids
    ]
    evaluations = await asyncio.gather(*tasks)

    # Fetch all results concurrently
    result_tasks = [
        client.trace_evaluations.get_results(e.id)
        for e in evaluations if e
    ]
    all_results = await asyncio.gather(*result_tasks)

    for results_response in all_results:
        if results_response:
            for result in results_response.results:
                print(f"Score: {result.score}, Passed: {result.passed}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

```python
from layerlens import Stratix
import layerlens

client = Stratix()

try:
    judge = client.judges.create(
        name="My Judge",
        evaluation_goal="Evaluate output quality",
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
