# Evaluations

The `evaluations` resource allows you to create and manage AI model evaluations against various benchmarks. This is the core functionality of the Atlas platform.

## Overview

An evaluation runs a specified model against a benchmark dataset and returns comprehensive metrics including accuracy, readability, toxicity, and ethics scores.

## Methods

### `create(model, benchmark, timeout=None)`

Creates a new evaluation for the specified model and benchmark.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | The model identifier to evaluate |
| `benchmark` | `str` | Yes | The benchmark dataset identifier |
| `timeout` | `float \| httpx.Timeout \| None` | No | Override request timeout |

#### Returns

Returns an `Evaluation` object if successful, `None` if the evaluation could not be created.

#### Example

```python
from atlas import Atlas

client = Atlas()

# Create a basic evaluation
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)

if evaluation:
    print(f"Evaluation created: {evaluation.id}")
    print(f"Status: {evaluation.status}")
else:
    print("Failed to create evaluation")
```

#### With Custom Timeout

```python
# Create evaluation with custom timeout (5 minutes)
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu",
    timeout=300.0
)
```

## Response Object

The `create` method returns an `Evaluation` object with the following properties:

### Core Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique evaluation identifier |
| `status` | `str` | Current evaluation status |
| `status_description` | `str` | Detailed status description |
| `submitted_at` | `int` | Unix timestamp when evaluation was submitted |
| `finished_at` | `int` | Unix timestamp when evaluation finished |

### Model Information

| Property | Type | Description |
|----------|------|-------------|
| `model_id` | `str` | Model identifier used in the request |
| `model_name` | `str` | Human-readable model name |
| `model_key` | `str` | Internal model key |
| `model_company` | `str` | Company that created the model |

### Benchmark Information

| Property | Type | Description |
|----------|------|-------------|
| `dataset_id` | `str` | Benchmark identifier used in the request |
| `dataset_name` | `str` | Human-readable benchmark name |

### Performance Metrics

These properties are available once the evaluation is completed:

| Property | Type | Description |
|----------|------|-------------|
| `accuracy` | `float` | Overall accuracy score (0.0 to 1.0) |
| `readability_score` | `float` | Readability assessment score |
| `toxicity_score` | `float` | Toxicity assessment score |
| `ethics_score` | `float` | Ethics assessment score |
| `average_duration` | `int` | Average response time in milliseconds |

## Evaluation Status

The `status` field can have the following values:

| Status | Description |
|--------|-------------|
| `"pending"` | Evaluation queued but not yet started |
| `"running"` | Evaluation currently in progress |
| `"completed"` | Evaluation finished successfully |
| `"failed"` | Evaluation failed due to an error |
| `"cancelled"` | Evaluation was cancelled by user |

## Complete Example

```python
import time
from atlas import Atlas
import atlas

def create_and_monitor_evaluation():
    client = Atlas()
    
    try:
        # Create evaluation
        evaluation = client.evaluations.create(
            model="gpt-3.5-turbo",
            benchmark="mmlu"
        )
        
        if not evaluation:
            print("❌ Failed to create evaluation")
            return None
            
        print(f"✅ Evaluation created: {evaluation.id}")
        print(f"📊 Model: {evaluation.model_name} ({evaluation.model_company})")
        print(f"📋 Benchmark: {evaluation.dataset_name}")
        print(f"⏰ Submitted at: {evaluation.submitted_at}")
        print(f"🔄 Status: {evaluation.status}")
        
        # Note: In practice, you'd use webhooks or polling to check status
        # This is just for demonstration
        if evaluation.status == "completed":
            print(f"\n📈 Results:")
            print(f"   Accuracy: {evaluation.accuracy:.2%}")
            print(f"   Readability: {evaluation.readability_score:.2f}")
            print(f"   Toxicity: {evaluation.toxicity_score:.2f}")
            print(f"   Ethics: {evaluation.ethics_score:.2f}")
            print(f"   Avg Duration: {evaluation.average_duration}ms")
        
        return evaluation
        
    except atlas.AuthenticationError:
        print("❌ Authentication failed - check your API key")
    except atlas.PermissionDeniedError:
        print("❌ Permission denied - check your organization/project access")
    except atlas.NotFoundError:
        print("❌ Model or benchmark not found")
    except atlas.RateLimitError:
        print("❌ Rate limit exceeded - please wait and try again")
    except atlas.APIConnectionError as e:
        print(f"❌ Connection error: {e}")
    except atlas.APIError as e:
        print(f"❌ API error: {e}")
    
    return None

if __name__ == "__main__":
    evaluation = create_and_monitor_evaluation()
```

## Available Models

Common model identifiers include:

- `"gpt-4"` - OpenAI GPT-4
- `"gpt-3.5-turbo"` - OpenAI GPT-3.5 Turbo
- `"claude-3-opus"` - Anthropic Claude 3 Opus
- `"claude-3-sonnet"` - Anthropic Claude 3 Sonnet
- `"llama-2-70b"` - Meta Llama 2 70B
- `"mistral-7b"` - Mistral 7B

> **Note**: Available models may vary based on your organization's access. Check the LayerLens Atlas dashboard for the complete list of available models.

## Available Benchmarks

Common benchmark identifiers include:

- `"mmlu"` - Massive Multitask Language Understanding
- `"hellaswag"` - HellaSwag commonsense reasoning
- `"arc-challenge"` - AI2 Reasoning Challenge
- `"truthfulqa"` - TruthfulQA
- `"winogrande"` - WinoGrande
- `"gsm8k"` - Grade School Math 8K

> **Note**: Available benchmarks may vary based on your organization's access. Check the LayerLens Atlas dashboard for the complete list of available benchmarks.

## Error Handling

### Common Errors

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    evaluation = client.evaluations.create(
        model="nonexistent-model",
        benchmark="mmlu"
    )
except atlas.NotFoundError:
    print("Model 'nonexistent-model' not found")
except atlas.BadRequestError:
    print("Invalid request parameters")
except atlas.UnprocessableEntityError:
    print("Request parameters are valid but cannot be processed")
```

### Timeout Handling

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    evaluation = client.evaluations.create(
        model="gpt-4",
        benchmark="mmlu",
        timeout=30.0  # 30 seconds
    )
except atlas.APITimeoutError:
    print("Request timed out - try increasing timeout or check network")
```

## Best Practices

### 1. Check Return Values
```python
# ✅ Good - always check if evaluation was created
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
if evaluation:
    print(f"Success: {evaluation.id}")
else:
    print("Failed to create evaluation")

# ❌ Bad - assuming success
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
print(f"Success: {evaluation.id}")  # Could raise AttributeError
```

### 2. Handle Long-Running Operations
```python
# ✅ Good - appropriate timeout for evaluation creation
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu",
    timeout=120.0  # 2 minutes
)

# ❌ Bad - timeout too short
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu", 
    timeout=5.0  # Likely to timeout
)
```

### 3. Store Evaluation IDs
```python
# ✅ Good - store evaluation ID for later retrieval
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
if evaluation:
    # Store this ID in your database/system
    evaluation_id = evaluation.id
    print(f"Store this ID: {evaluation_id}")
```

## Next Steps

- Learn how to [retrieve results](results.md) for your evaluations
- Explore [code examples](../examples/creating-evaluations.md) for common patterns
- Understand [error handling](errors.md) for robust applications