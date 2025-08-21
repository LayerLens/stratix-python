# Creating Evaluations

Simple examples for creating AI model evaluations with the Atlas Python SDK.

## Quick Start

### Basic Evaluation

```python
from atlas import Atlas

# Initialize client (reads LAYERLENS_ATLAS_API_KEY from environment)
client = Atlas()

# Get available models and benchmarks
models = client.models.get()
benchmarks = client.benchmarks.get()

print(f"Available: {len(models)} models, {len(benchmarks)} benchmarks")

# Create evaluation with first available model and benchmark
evaluation = client.evaluations.create(
    model=models[0],
    benchmark=benchmarks[0]
)

print(f"Created evaluation: {evaluation.id}")
print(f"Status: {evaluation.status}")
```

### Choose Specific Model and Benchmark

```python
from atlas import Atlas

client = Atlas()

# Get all available options
models = client.models.get()
benchmarks = client.benchmarks.get()

# Find specific model and benchmark
gpt4_model = next((m for m in models if "gpt-4" in m.id), None)
mmlu_benchmark = next((b for b in benchmarks if "mmlu" in b.id), None)

if gpt4_model and mmlu_benchmark:
    evaluation = client.evaluations.create(
        model=gpt4_model,
        benchmark=mmlu_benchmark
    )
    print(f"Created: {evaluation.id}")
else:
    print("Model or benchmark not found")
    print(f"Available models: {[m.id for m in models[:5]]}...")
    print(f"Available benchmarks: {[b.id for b in benchmarks[:5]]}...")
```

## Async Version

```python
import asyncio
from atlas import AsyncAtlas

async def create_evaluation():
    client = AsyncAtlas()
    
    models = await client.models.get()
    benchmarks = await client.benchmarks.get()
    
    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0]
    )
    
    print(f"Created evaluation: {evaluation.id}")
    return evaluation

# Run it
asyncio.run(create_evaluation())
```

## Wait for Completion

```python
from atlas import Atlas

client = Atlas()
models = client.models.get()
benchmarks = client.benchmarks.get()

# Create evaluation
evaluation = client.evaluations.create(model=models[0], benchmark=benchmarks[0])
print(f"Created: {evaluation.id}")

# Wait for it to complete (this may take several minutes)
completed_evaluation = client.evaluations.wait_for_completion(
    evaluation,
    interval_seconds=30,  # Check every 30 seconds
    timeout=1800         # 30 minute timeout
)

if completed_evaluation.is_success:
    print("Evaluation completed successfully!")
else:
    print(f"Evaluation failed: {completed_evaluation.status}")
```

## Error Handling

```python
from atlas import Atlas
import atlas

client = Atlas()

try:
    models = client.models.get()
    benchmarks = client.benchmarks.get()
    
    evaluation = client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0]
    )
    print(f"Success: {evaluation.id}")
    
except atlas.AuthenticationError:
    print("Check your API key")
except atlas.NotFoundError:
    print("Model or benchmark not found")
except atlas.APIError as e:
    print(f"API error: {e}")
```

## Multiple Evaluations

```python
from atlas import Atlas

client = Atlas()
models = client.models.get()
benchmarks = client.benchmarks.get()

# Create multiple evaluations
evaluations = []
for model in models[:3]:  # First 3 models
    for benchmark in benchmarks[:2]:  # First 2 benchmarks
        evaluation = client.evaluations.create(model=model, benchmark=benchmark)
        evaluations.append(evaluation)
        print(f"Created: {model.id} + {benchmark.id} = {evaluation.id}")

print(f"Created {len(evaluations)} evaluations total")
```
