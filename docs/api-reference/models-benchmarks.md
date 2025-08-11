# Models & Benchmarks

This page provides reference information about available models and benchmarks in the Atlas platform, along with guidance on selecting appropriate combinations for your evaluations.

## Overview

Atlas evaluations require two key components:
- **Model**: The AI model you want to evaluate
- **Benchmark**: The dataset/test suite to evaluate the model against

The availability of models and benchmarks depends on your organization's access level and the specific Atlas deployment you're using.

## Models

### Model Identification

Models are identified by string IDs that you pass to the `evaluations.create()` method:

```python
from atlas import Atlas

client = Atlas()

# Using model ID
evaluation = client.evaluations.create(
    model="gpt-4",  # Model ID
    benchmark="mmlu"
)
```

### Model Information

When you create an evaluation, the response includes detailed model information:

```python
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

if evaluation:
    print(f"Model ID: {evaluation.model_id}")           # "gpt-4"
    print(f"Model Name: {evaluation.model_name}")       # "GPT-4"
    print(f"Model Key: {evaluation.model_key}")         # Internal key
    print(f"Model Company: {evaluation.model_company}") # "OpenAI"
```

## Benchmarks

### Benchmark Identification

Benchmarks are identified by string IDs representing different evaluation datasets:

```python
from atlas import Atlas

client = Atlas()

evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"  # Benchmark ID
)
```

### Benchmark Information

Evaluation responses include benchmark details:

```python
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

if evaluation:
    print(f"Dataset ID: {evaluation.dataset_id}")       # "mmlu"
    print(f"Dataset Name: {evaluation.dataset_name}")   # "MMLU"
```

### Performance Expectations

Different model-benchmark combinations yield different types of insights:

#### General Intelligence Assessment
```python
# Broad capability assessment
models = ["gpt-4", "claude-3-opus", "llama-2-70b"]
benchmark = "mmlu"

for model in models:
    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    # Compare general intelligence across models
```

#### Specialized Task Performance
```python
# Code generation comparison
models = ["gpt-4", "code-llama-34b", "claude-3-sonnet"]
benchmark = "humaneval"

for model in models:
    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    # Compare coding abilities
```

## Discovery and Validation

### Finding Available Models and Benchmarks

#### Check the Atlas Dashboard
The most reliable way to find available models and benchmarks:

1. Log into your Atlas dashboard
2. Navigate to the evaluation creation page
3. View dropdown lists of available models and benchmarks
4. Note the exact IDs for use in your code

#### Programmatic Discovery

While the SDK doesn't currently provide discovery endpoints, you can validate model/benchmark existence:

```python
import atlas
from atlas import Atlas

def validate_model_benchmark(model_id: str, benchmark_id: str) -> bool:
    """Test if a model/benchmark combination is available"""
    client = Atlas()
    
    try:
        evaluation = client.evaluations.create(
            model=model_id,
            benchmark=benchmark_id
        )
        
        if evaluation:
            print(f"✅ Valid: {model_id} + {benchmark_id}")
            return True
        else:
            print(f"❌ Invalid: {model_id} + {benchmark_id}")
            return False
            
    except atlas.NotFoundError:
        print(f"❌ Not found: {model_id} or {benchmark_id}")
        return False
    except atlas.PermissionDeniedError:
        print(f"❌ No access: {model_id} or {benchmark_id}")
        return False
    except atlas.APIError as e:
        print(f"❌ Error: {e}")
        return False

# Test combinations
combinations = [
    ("gpt-4", "mmlu"),
    ("claude-3-opus", "hellaswag"),
    ("llama-2-70b", "arc-challenge"),
    ("nonexistent-model", "mmlu"),  # Should fail
]

for model, benchmark in combinations:
    validate_model_benchmark(model, benchmark)
```

### Batch Validation

```python
def batch_validate_combinations(model_benchmark_pairs):
    """Validate multiple model/benchmark combinations"""
    client = Atlas()
    results = {}
    
    for model, benchmark in model_benchmark_pairs:
        try:
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            results[(model, benchmark)] = {
                "valid": evaluation is not None,
                "evaluation_id": evaluation.id if evaluation else None,
                "model_name": evaluation.model_name if evaluation else None,
                "dataset_name": evaluation.dataset_name if evaluation else None,
            }
        except atlas.APIError as e:
            results[(model, benchmark)] = {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    return results

# Example usage
combinations = [
    ("gpt-4", "mmlu"),
    ("claude-3-sonnet", "hellaswag"),
    ("llama-2-70b", "gsm8k"),
]

results = batch_validate_combinations(combinations)
for (model, benchmark), result in results.items():
    status = "✅" if result["valid"] else "❌"
    print(f"{status} {model} + {benchmark}: {result}")
```

### Validate Before Production Use

```python
def safe_create_evaluation(model: str, benchmark: str):
    """Create evaluation with validation and error handling"""
    client = Atlas()
    
    # Validate combination first
    if not validate_model_benchmark(model, benchmark):
        return None
    
    try:
        evaluation = client.evaluations.create(model=model, benchmark=benchmark)
        
        if evaluation:
            print(f"✅ Evaluation created successfully:")
            print(f"   ID: {evaluation.id}")
            print(f"   Model: {evaluation.model_name} ({evaluation.model_company})")
            print(f"   Benchmark: {evaluation.dataset_name}")
            return evaluation
        else:
            print(f"❌ Failed to create evaluation")
            return None
            
    except atlas.APIError as e:
        print(f"❌ API error: {e}")
        return None

# Usage
evaluation = safe_create_evaluation("gpt-4", "mmlu")
```

### 4. Document Model and Benchmark Choices

```python
# Document your evaluation strategy
EVALUATION_CONFIGS = {
    "general_intelligence": {
        "models": ["gpt-4", "claude-3-opus", "gemini-pro"],
        "benchmarks": ["mmlu", "arc-challenge", "hellaswag"],
        "description": "Broad cognitive ability assessment"
    },
    "code_generation": {
        "models": ["gpt-4", "code-llama-34b", "claude-3-sonnet"],
        "benchmarks": ["humaneval", "mbpp", "apps"],
        "description": "Programming and code generation capabilities"
    },
    "mathematical_reasoning": {
        "models": ["gpt-4", "claude-3-opus", "minerva-62b"],
        "benchmarks": ["gsm8k", "math", "minerva-math"],
        "description": "Mathematical problem-solving abilities"
    }
}

def run_evaluation_suite(suite_name: str):
    """Run a predefined evaluation suite"""
    if suite_name not in EVALUATION_CONFIGS:
        print(f"Unknown suite: {suite_name}")
        return
    
    config = EVALUATION_CONFIGS[suite_name]
    print(f"Running {suite_name}: {config['description']}")
    
    client = Atlas()
    evaluations = []
    
    for model in config["models"]:
        for benchmark in config["benchmarks"]:
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            if evaluation:
                evaluations.append(evaluation)
                print(f"✅ {model} + {benchmark}: {evaluation.id}")
    
    return evaluations

# Run comprehensive evaluation
evaluations = run_evaluation_suite("general_intelligence")
```

## Troubleshooting

### Model or Benchmark Not Found

```python
try:
    evaluation = client.evaluations.create(
        model="nonexistent-model",
        benchmark="mmlu"
    )
except atlas.NotFoundError:
    print("Model or benchmark not found. Check:")
    print("1. Spelling of model/benchmark ID")
    print("2. Available options in Atlas dashboard")
    print("3. Your organization's access permissions")
```

### Permission Issues

```python
try:
    evaluation = client.evaluations.create(
        model="restricted-model",
        benchmark="private-benchmark"
    )
except atlas.PermissionDeniedError:
    print("Access denied. Possible causes:")
    print("1. Model requires higher permission level")
    print("2. Benchmark is not available to your organization")
    print("3. Project doesn't have access to these resources")
```

### Validation Errors

```python
try:
    evaluation = client.evaluations.create(
        model="",  # Empty string
        benchmark="mmlu"
    )
except atlas.BadRequestError:
    print("Invalid request parameters:")
    print("- Model and benchmark IDs cannot be empty")
    print("- IDs must be valid strings")
```

For more information about available models and benchmarks, consult your Atlas dashboard or contact your LayerLens administrator.