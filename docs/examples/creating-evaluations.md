# Creating Evaluations

This guide provides practical examples for creating evaluations with the Atlas Python SDK.

## Basic Evaluation Creation

### Simple Evaluation

The most straightforward way to create an evaluation:

```python
from atlas import Atlas

# Initialize client
client = Atlas()

# Create evaluation
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)

if evaluation:
    print(f"✅ Evaluation created: {evaluation.id}")
    print(f"   Model: {evaluation.model_name}")
    print(f"   Benchmark: {evaluation.dataset_name}")
    print(f"   Status: {evaluation.status}")
else:
    print("❌ Failed to create evaluation")
```

### With Explicit Configuration

Using explicit client configuration instead of environment variables:

```python
from atlas import Atlas

# Explicit configuration
client = Atlas(api_key="your_api_key_here")

evaluation = client.evaluations.create(
    model="claude-3-opus",
    benchmark="hellaswag"
)

if evaluation:
    print(f"Evaluation ID: {evaluation.id}")
    print(f"Submitted at: {evaluation.submitted_at}")
```

## Batch Evaluation Creation

### Multiple Models on Same Benchmark

Compare multiple models against the same benchmark:

```python
from atlas import Atlas
import time

def compare_models_on_benchmark(models: list, benchmark: str):
    """Create evaluations for multiple models on the same benchmark"""
    client = Atlas()
    evaluations = []

    print(f"🔄 Creating evaluations for {len(models)} models on {benchmark}")

    for model in models:
        try:
            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark
            )

            if evaluation:
                evaluations.append({
                    "model": model,
                    "evaluation_id": evaluation.id,
                    "model_name": evaluation.model_name,
                    "status": evaluation.status
                })
                print(f"✅ {model}: {evaluation.id}")
            else:
                print(f"❌ Failed to create evaluation for {model}")

        except Exception as e:
            print(f"❌ Error creating evaluation for {model}: {e}")

        # Brief pause between requests to avoid rate limits
        time.sleep(0.5)

    return evaluations

# Usage
models_to_compare = [
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
    "llama-2-70b"
]

evaluations = compare_models_on_benchmark(models_to_compare, "mmlu")

# Print summary
print(f"\n📊 Created {len(evaluations)} evaluations:")
for eval_info in evaluations:
    print(f"   {eval_info['model_name']}: {eval_info['evaluation_id']}")
```

### Single Model on Multiple Benchmarks

Evaluate one model across multiple benchmarks:

```python
from atlas import Atlas
import time

def evaluate_model_on_benchmarks(model: str, benchmarks: list):
    """Evaluate a single model across multiple benchmarks"""
    client = Atlas()
    evaluations = []

    print(f"🔄 Evaluating {model} on {len(benchmarks)} benchmarks")

    for benchmark in benchmarks:
        try:
            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark
            )

            if evaluation:
                evaluations.append({
                    "benchmark": benchmark,
                    "evaluation_id": evaluation.id,
                    "dataset_name": evaluation.dataset_name,
                    "status": evaluation.status
                })
                print(f"✅ {benchmark}: {evaluation.id}")
            else:
                print(f"❌ Failed to create evaluation for {benchmark}")

        except Exception as e:
            print(f"❌ Error evaluating on {benchmark}: {e}")

        time.sleep(0.5)

    return evaluations

# Usage
benchmarks_to_test = [
    "mmlu",
    "hellaswag",
    "arc-challenge",
    "truthfulqa",
    "gsm8k"
]

evaluations = evaluate_model_on_benchmarks("gpt-4", benchmarks_to_test)

print(f"\n📊 Created {len(evaluations)} evaluations for GPT-4:")
for eval_info in evaluations:
    print(f"   {eval_info['dataset_name']}: {eval_info['evaluation_id']}")
```

### Full Matrix Evaluation

Create evaluations for all model-benchmark combinations:

```python
from atlas import Atlas
import time
import itertools

def create_evaluation_matrix(models: list, benchmarks: list, delay: float = 1.0):
    """Create evaluations for all model-benchmark combinations"""
    client = Atlas()
    results = {}
    total_combinations = len(models) * len(benchmarks)

    print(f"🔄 Creating {total_combinations} evaluations...")

    for i, (model, benchmark) in enumerate(itertools.product(models, benchmarks), 1):
        print(f"\n[{i}/{total_combinations}] {model} + {benchmark}")

        try:
            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark
            )

            if evaluation:
                if model not in results:
                    results[model] = {}

                results[model][benchmark] = {
                    "evaluation_id": evaluation.id,
                    "model_name": evaluation.model_name,
                    "dataset_name": evaluation.dataset_name,
                    "status": evaluation.status,
                    "success": True
                }
                print(f"✅ Success: {evaluation.id}")
            else:
                print(f"❌ Failed: No evaluation created")

        except Exception as e:
            print(f"❌ Error: {e}")
            if model not in results:
                results[model] = {}
            results[model][benchmark] = {
                "error": str(e),
                "success": False
            }

        # Rate limiting
        if i < total_combinations:
            time.sleep(delay)

    return results

# Usage
test_models = ["gpt-4", "claude-3-opus", "llama-2-70b"]
test_benchmarks = ["mmlu", "hellaswag", "arc-challenge"]

matrix_results = create_evaluation_matrix(test_models, test_benchmarks, delay=2.0)

# Print summary table
print(f"\n📊 Evaluation Matrix Results:")
print("Model".ljust(15), end="")
for benchmark in test_benchmarks:
    print(benchmark.ljust(15), end="")
print()

for model in test_models:
    print(model.ljust(15), end="")
    for benchmark in test_benchmarks:
        if model in matrix_results and benchmark in matrix_results[model]:
            result = matrix_results[model][benchmark]
            status = "✅" if result["success"] else "❌"
            print(status.ljust(15), end="")
        else:
            print("❓".ljust(15), end="")
    print()
```

## Error Handling and Resilience

### Robust Evaluation Creation with Retries

```python
import atlas
from atlas import Atlas
import time
import random

def create_evaluation_with_retry(
    model: str,
    benchmark: str,
    max_retries: int = 3,
    base_delay: float = 1.0
):
    """Create evaluation with exponential backoff retry logic"""
    client = Atlas()

    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries}: Creating evaluation...")

            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark,
                timeout=120.0  # 2-minute timeout
            )

            if evaluation:
                print(f"✅ Success on attempt {attempt + 1}: {evaluation.id}")
                return evaluation
            else:
                print(f"❌ Evaluation creation returned None on attempt {attempt + 1}")

        except atlas.RateLimitError as e:
            retry_after = e.response.headers.get('retry-after', base_delay * (2 ** attempt))
            print(f"⏳ Rate limited, waiting {retry_after}s...")
            time.sleep(float(retry_after))
            continue

        except atlas.InternalServerError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"🔄 Server error, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                print("❌ Server error - max retries exceeded")
                break

        except atlas.APIConnectionError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"🔄 Connection error, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                print("❌ Connection failed - max retries exceeded")
                break

        except atlas.AuthenticationError:
            print("❌ Authentication failed - check your API key")
            break

        except atlas.NotFoundError:
            print(f"❌ Model '{model}' or benchmark '{benchmark}' not found")
            break

        except atlas.PermissionDeniedError:
            print("❌ Permission denied - check your access rights")
            break

        except atlas.APIError as e:
            print(f"❌ API error: {e}")
            break

    return None

# Usage
evaluation = create_evaluation_with_retry(
    model="gpt-4",
    benchmark="mmlu",
    max_retries=3
)

if evaluation:
    print(f"Final result: {evaluation.id}")
else:
    print("Failed to create evaluation after all attempts")
```

### Validation Before Creation

```python
import atlas
from atlas import Atlas

def validate_and_create_evaluation(model: str, benchmark: str):
    """Validate model and benchmark before creating evaluation"""
    client = Atlas()

    # Pre-validation checks
    if not model or not model.strip():
        print("❌ Model cannot be empty")
        return None

    if not benchmark or not benchmark.strip():
        print("❌ Benchmark cannot be empty")
        return None

    print(f"🔍 Validating {model} + {benchmark}...")

    try:
        # Attempt to create the evaluation
        evaluation = client.evaluations.create(
            model=model.strip(),
            benchmark=benchmark.strip()
        )

        if evaluation:
            print(f"✅ Validation successful!")
            print(f"   Evaluation ID: {evaluation.id}")
            print(f"   Model: {evaluation.model_name} ({evaluation.model_company})")
            print(f"   Benchmark: {evaluation.dataset_name}")
            print(f"   Status: {evaluation.status}")
            return evaluation
        else:
            print("❌ Validation failed: No evaluation returned")
            return None

    except atlas.NotFoundError:
        print(f"❌ Validation failed: Model '{model}' or benchmark '{benchmark}' not found")
        print("💡 Suggestions:")
        print("   • Check spelling of model and benchmark IDs")
        print("   • Verify available options in Atlas dashboard")
        print("   • Ensure your organization has access to these resources")
        return None

    except atlas.AuthenticationError:
        print("❌ Authentication failed")
        print("💡 Check your API key configuration")
        return None

    except atlas.PermissionDeniedError:
        print("❌ Permission denied")
        print("💡 Contact your administrator for access")
        return None

    except atlas.APIError as e:
        print(f"❌ Validation failed: {e}")
        return None

# Usage with validation
test_combinations = [
    ("gpt-4", "mmlu"),
    ("claude-3-opus", "hellaswag"),
    ("nonexistent-model", "mmlu"),  # This should fail
    ("gpt-4", "nonexistent-benchmark"),  # This should fail
]

for model, benchmark in test_combinations:
    print(f"\n{'='*50}")
    evaluation = validate_and_create_evaluation(model, benchmark)

    if evaluation:
        print(f"Ready to monitor evaluation: {evaluation.id}")
```

## Custom Timeout Configurations

### Different Timeouts for Different Operations

```python
from atlas import Atlas
import httpx

def create_evaluations_with_custom_timeouts():
    """Demonstrate different timeout configurations"""

    # Quick timeout for testing connectivity
    quick_client = Atlas(timeout=30.0)  # 30 seconds

    # Standard timeout for regular evaluations
    standard_client = Atlas(timeout=300.0)  # 5 minutes

    # Long timeout for complex evaluations
    patient_client = Atlas(
        timeout=httpx.Timeout(
            connect=10.0,   # 10s to connect
            read=1800.0,    # 30min to read response
            write=60.0,     # 1min to send request
            pool=30.0       # 30s for connection pool
        )
    )

    # Test connectivity with quick client
    print("🔍 Testing connectivity...")
    try:
        test_eval = quick_client.evaluations.create(
            model="gpt-3.5-turbo",  # Faster model for testing
            benchmark="arc-easy"     # Smaller benchmark for testing
        )
        print("✅ Connectivity test passed")
    except atlas.APITimeoutError:
        print("❌ Quick connectivity test failed - network issues?")
        return
    except atlas.APIError as e:
        print(f"❌ API error during connectivity test: {e}")
        return

    # Create standard evaluation
    print("\n🔄 Creating standard evaluation...")
    try:
        standard_eval = standard_client.evaluations.create(
            model="gpt-4",
            benchmark="mmlu"
        )
        if standard_eval:
            print(f"✅ Standard evaluation created: {standard_eval.id}")
    except atlas.APITimeoutError:
        print("❌ Standard evaluation timed out")

    # Create complex evaluation with patient timeout
    print("\n🔄 Creating complex evaluation...")
    try:
        complex_eval = patient_client.evaluations.create(
            model="gpt-4",
            benchmark="math"  # Complex benchmark
        )
        if complex_eval:
            print(f"✅ Complex evaluation created: {complex_eval.id}")
    except atlas.APITimeoutError:
        print("❌ Complex evaluation timed out even with extended timeout")

# Run the example
create_evaluations_with_custom_timeouts()
```

### Per-Request Timeout Override

```python
from atlas import Atlas

def create_evaluation_with_override_timeout():
    """Override timeout for specific requests"""
    client = Atlas(timeout=60.0)  # Default 1-minute timeout

    evaluations = []

    # Quick evaluation with short timeout
    print("🔄 Quick evaluation (30s timeout)...")
    try:
        quick_eval = client.with_options(timeout=30.0).evaluations.create(
            model="gpt-3.5-turbo",
            benchmark="arc-easy"
        )
        if quick_eval:
            evaluations.append(("Quick", quick_eval))
            print(f"✅ Quick: {quick_eval.id}")
    except atlas.APITimeoutError:
        print("❌ Quick evaluation timed out")

    # Standard evaluation (uses default timeout)
    print("\n🔄 Standard evaluation (default 60s timeout)...")
    try:
        standard_eval = client.evaluations.create(
            model="gpt-4",
            benchmark="mmlu"
        )
        if standard_eval:
            evaluations.append(("Standard", standard_eval))
            print(f"✅ Standard: {standard_eval.id}")
    except atlas.APITimeoutError:
        print("❌ Standard evaluation timed out")

    # Long evaluation with extended timeout
    print("\n🔄 Long evaluation (5min timeout)...")
    try:
        long_eval = client.with_options(timeout=300.0).evaluations.create(
            model="gpt-4",
            benchmark="math"
        )
        if long_eval:
            evaluations.append(("Long", long_eval))
            print(f"✅ Long: {long_eval.id}")
    except atlas.APITimeoutError:
        print("❌ Long evaluation timed out")

    return evaluations

evaluations = create_evaluation_with_override_timeout()
print(f"\n📊 Created {len(evaluations)} evaluations total")
```

## Monitoring and Logging

### Evaluation Creation with Logging

```python
import logging
from datetime import datetime
from atlas import Atlas
import atlas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('atlas_evaluations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_evaluation_with_logging(model: str, benchmark: str, context: dict = None):
    """Create evaluation with comprehensive logging"""
    client = Atlas()
    context = context or {}

    logger.info(f"Starting evaluation creation: {model} + {benchmark}")
    logger.info(f"Context: {context}")

    start_time = datetime.now()

    try:
        evaluation = client.evaluations.create(
            model=model,
            benchmark=benchmark
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if evaluation:
            logger.info(f"✅ Evaluation created successfully in {duration:.2f}s")
            logger.info(f"   ID: {evaluation.id}")
            logger.info(f"   Model: {evaluation.model_name} ({evaluation.model_company})")
            logger.info(f"   Benchmark: {evaluation.dataset_name}")
            logger.info(f"   Status: {evaluation.status}")
            logger.info(f"   Submitted at: {evaluation.submitted_at}")

            return {
                "success": True,
                "evaluation": evaluation,
                "duration": duration,
                "timestamp": start_time.isoformat()
            }
        else:
            logger.error(f"❌ Evaluation creation failed - returned None")
            return {
                "success": False,
                "error": "No evaluation returned",
                "duration": duration,
                "timestamp": start_time.isoformat()
            }

    except atlas.RateLimitError as e:
        logger.warning(f"⏳ Rate limited - request ID: {getattr(e, 'request_id', 'N/A')}")
        return {"success": False, "error": "rate_limited", "retry_after": e.response.headers.get('retry-after')}

    except atlas.AuthenticationError:
        logger.error("❌ Authentication failed - check API key")
        return {"success": False, "error": "authentication_failed"}

    except atlas.NotFoundError:
        logger.error(f"❌ Model '{model}' or benchmark '{benchmark}' not found")
        return {"success": False, "error": "not_found", "model": model, "benchmark": benchmark}

    except atlas.APIError as e:
        logger.error(f"❌ API error: {e}")
        return {"success": False, "error": str(e), "error_type": type(e).__name__}

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return {"success": False, "error": f"unexpected: {e}"}

# Usage
evaluation_configs = [
    {"model": "gpt-4", "benchmark": "mmlu", "context": {"purpose": "baseline_test"}},
    {"model": "claude-3-opus", "benchmark": "hellaswag", "context": {"purpose": "reasoning_comparison"}},
    {"model": "llama-2-70b", "benchmark": "gsm8k", "context": {"purpose": "math_evaluation"}},
]

results = []
for config in evaluation_configs:
    result = create_evaluation_with_logging(**config)
    results.append(result)

    if not result["success"]:
        logger.error(f"Failed to create evaluation: {config}")

# Summary
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

logger.info(f"📊 Summary: {len(successful)} successful, {len(failed)} failed")
for result in successful:
    logger.info(f"   ✅ {result['evaluation'].id} ({result['duration']:.2f}s)")
for result in failed:
    logger.info(f"   ❌ {result.get('error', 'unknown_error')}")
```

## Advanced Patterns

### Evaluation Factory Pattern

```python
from atlas import Atlas
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import atlas

class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies"""

    @abstractmethod
    def get_model_benchmark_pairs(self) -> List[tuple]:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

class GeneralIntelligenceStrategy(EvaluationStrategy):
    """Strategy for general intelligence assessment"""

    def get_model_benchmark_pairs(self) -> List[tuple]:
        models = ["gpt-4", "claude-3-opus", "llama-2-70b"]
        benchmarks = ["mmlu", "arc-challenge", "hellaswag"]
        return [(m, b) for m in models for b in benchmarks]

    def get_description(self) -> str:
        return "General intelligence assessment across major benchmarks"

class CodeGenerationStrategy(EvaluationStrategy):
    """Strategy for code generation assessment"""

    def get_model_benchmark_pairs(self) -> List[tuple]:
        models = ["gpt-4", "code-llama-34b", "claude-3-sonnet"]
        benchmarks = ["humaneval", "mbpp"]
        return [(m, b) for m in models for b in benchmarks]

    def get_description(self) -> str:
        return "Code generation capability assessment"

class MathReasoningStrategy(EvaluationStrategy):
    """Strategy for mathematical reasoning assessment"""

    def get_model_benchmark_pairs(self) -> List[tuple]:
        models = ["gpt-4", "claude-3-opus", "minerva-62b"]
        benchmarks = ["gsm8k", "math"]
        return [(m, b) for m in models for b in benchmarks]

    def get_description(self) -> str:
        return "Mathematical reasoning and problem-solving assessment"

class EvaluationFactory:
    """Factory for creating evaluations based on strategies"""

    def __init__(self):
        self.client = Atlas()

    def execute_strategy(self, strategy: EvaluationStrategy) -> Dict[str, Any]:
        """Execute an evaluation strategy"""
        pairs = strategy.get_model_benchmark_pairs()
        description = strategy.get_description()

        print(f"🔄 Executing strategy: {description}")
        print(f"📊 Creating {len(pairs)} evaluations...")

        results = {
            "strategy": description,
            "evaluations": [],
            "errors": [],
            "summary": {"total": len(pairs), "successful": 0, "failed": 0}
        }

        for model, benchmark in pairs:
            try:
                evaluation = self.client.evaluations.create(
                    model=model,
                    benchmark=benchmark
                )

                if evaluation:
                    results["evaluations"].append({
                        "model": model,
                        "benchmark": benchmark,
                        "evaluation_id": evaluation.id,
                        "model_name": evaluation.model_name,
                        "dataset_name": evaluation.dataset_name,
                        "status": evaluation.status
                    })
                    results["summary"]["successful"] += 1
                    print(f"✅ {model} + {benchmark}: {evaluation.id}")
                else:
                    results["errors"].append({
                        "model": model,
                        "benchmark": benchmark,
                        "error": "No evaluation returned"
                    })
                    results["summary"]["failed"] += 1
                    print(f"❌ {model} + {benchmark}: Failed")

            except atlas.APIError as e:
                results["errors"].append({
                    "model": model,
                    "benchmark": benchmark,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                results["summary"]["failed"] += 1
                print(f"❌ {model} + {benchmark}: {e}")

        return results

# Usage
factory = EvaluationFactory()

# Run different strategies
strategies = [
    GeneralIntelligenceStrategy(),
    CodeGenerationStrategy(),
    MathReasoningStrategy()
]

all_results = []
for strategy in strategies:
    result = factory.execute_strategy(strategy)
    all_results.append(result)

    print(f"\n📈 Strategy Results: {result['strategy']}")
    print(f"   Successful: {result['summary']['successful']}")
    print(f"   Failed: {result['summary']['failed']}")
    print()

# Overall summary
total_evaluations = sum(r["summary"]["successful"] for r in all_results)
total_errors = sum(r["summary"]["failed"] for r in all_results)

print(f"🎯 Overall Summary:")
print(f"   Total evaluations created: {total_evaluations}")
print(f"   Total errors: {total_errors}")
print(f"   Success rate: {total_evaluations/(total_evaluations+total_errors)*100:.1f}%")
```
