# Working with Timeouts

This guide provides practical examples for configuring and handling timeouts effectively with the Atlas Python SDK.

## Understanding Timeouts

Timeouts in the Atlas SDK control how long to wait for API responses. Different operations may require different timeout configurations based on their expected duration and criticality.

## Basic Timeout Configuration

### Simple Timeout

```python
from atlas import Atlas

# Set a 2-minute timeout for all requests
client = Atlas(timeout=120.0)

# Create evaluation with 2-minute timeout
evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)
```

### Default Timeout Behavior

```python
from atlas import Atlas

# Uses default timeout (10 minutes)
client = Atlas()

print(f"Default timeout: {client.timeout}")  # Should show 10 minutes in seconds
```

## Advanced Timeout Configuration

### Granular Timeout Control

```python
import httpx
from atlas import Atlas

# Configure different timeouts for different operations
client = Atlas(
    timeout=httpx.Timeout(
        connect=10.0,   # 10 seconds to establish connection
        read=300.0,     # 5 minutes to read response
        write=30.0,     # 30 seconds to send request
        pool=60.0       # 1 minute for connection pool operations
    )
)

evaluation = client.evaluations.create(
    model="gpt-4",
    benchmark="mmlu"
)
```

### Per-Request Timeout Override

```python
from atlas import Atlas

# Client with default 1-minute timeout
client = Atlas(timeout=60.0)

# Override timeout for specific operations
try:
    # Quick operation with short timeout
    quick_eval = client.with_options(timeout=30.0).evaluations.create(
        model="gpt-3.5-turbo",
        benchmark="arc-easy"
    )
    
    # Long operation with extended timeout
    complex_eval = client.with_options(timeout=600.0).evaluations.create(
        model="gpt-4",
        benchmark="math"  # Complex benchmark
    )
    
    # Results retrieval with medium timeout
    results = client.with_options(timeout=120.0).results.get(
        evaluation_id=quick_eval.id
    )
    
except Exception as e:
    print(f"Operation failed: {e}")
```

## Timeout Strategies by Use Case

### Development and Testing

```python
from atlas import Atlas
import atlas

def development_client():
    """Client optimized for development with shorter timeouts"""
    return Atlas(
        timeout=30.0  # 30 seconds - fail fast during development
    )

def test_api_connectivity():
    """Quick connectivity test with very short timeout"""
    client = development_client()
    
    try:
        # Use simple, fast operation to test connectivity
        evaluation = client.with_options(timeout=10.0).evaluations.create(
            model="gpt-3.5-turbo",  # Usually faster
            benchmark="arc-easy"    # Smaller benchmark
        )
        
        if evaluation:
            print("✅ API connectivity confirmed")
            return True
        else:
            print("❌ API returned no evaluation")
            return False
            
    except atlas.APITimeoutError:
        print("❌ API timeout - connectivity issues or server overload")
        return False
    except atlas.APIConnectionError:
        print("❌ Connection failed - check network")
        return False
    except atlas.APIError as e:
        print(f"❌ API error: {e}")
        return False

# Usage
if test_api_connectivity():
    print("Proceeding with full evaluation...")
else:
    print("Fix connectivity issues before continuing")
```

### Production Workloads

```python
import httpx
from atlas import Atlas
import atlas

def production_client():
    """Client optimized for production workloads"""
    return Atlas(
        timeout=httpx.Timeout(
            connect=30.0,    # 30s to connect (allows for network delays)
            read=1800.0,     # 30 minutes for complex evaluations
            write=60.0,      # 1 minute to send large requests
            pool=120.0       # 2 minutes for connection pool
        )
    )

def robust_evaluation_creation(model: str, benchmark: str, max_retries: int = 3):
    """Production-ready evaluation creation with timeout handling"""
    client = production_client()
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries}: Creating evaluation...")
            
            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark
            )
            
            if evaluation:
                print(f"✅ Success: {evaluation.id}")
                return evaluation
            else:
                print("❌ No evaluation returned")
                
        except atlas.APITimeoutError:
            print(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                # Increase timeout for retry
                retry_timeout = 1800.0 + (attempt * 600.0)  # Add 10 minutes per retry
                print(f"🔄 Retrying with extended timeout: {retry_timeout/60:.0f} minutes")
                
                try:
                    evaluation = client.with_options(timeout=retry_timeout).evaluations.create(
                        model=model,
                        benchmark=benchmark
                    )
                    if evaluation:
                        print(f"✅ Success on retry: {evaluation.id}")
                        return evaluation
                except atlas.APITimeoutError:
                    print(f"⏰ Extended timeout also failed")
                    continue
            else:
                print("❌ All timeout retry attempts failed")
                
        except atlas.APIError as e:
            print(f"❌ API error: {e}")
            break  # Don't retry API errors
    
    return None

# Usage
evaluation = robust_evaluation_creation("gpt-4", "mmlu")
```

### Batch Operations

```python
from atlas import Atlas
import atlas
import time

def batch_evaluations_with_adaptive_timeout(model_benchmark_pairs: list):
    """Create multiple evaluations with adaptive timeout strategy"""
    client = Atlas(timeout=120.0)  # Start with 2-minute timeout
    
    results = []
    consecutive_timeouts = 0
    current_timeout = 120.0
    
    for i, (model, benchmark) in enumerate(model_benchmark_pairs, 1):
        print(f"\n[{i}/{len(model_benchmark_pairs)}] {model} + {benchmark}")
        print(f"Current timeout: {current_timeout/60:.1f} minutes")
        
        try:
            evaluation = client.with_options(timeout=current_timeout).evaluations.create(
                model=model,
                benchmark=benchmark
            )
            
            if evaluation:
                results.append({
                    "model": model,
                    "benchmark": benchmark,
                    "evaluation_id": evaluation.id,
                    "success": True,
                    "timeout_used": current_timeout
                })
                print(f"✅ Success: {evaluation.id}")
                
                # Reset timeout on success
                consecutive_timeouts = 0
                current_timeout = max(120.0, current_timeout * 0.9)  # Slightly reduce timeout
            else:
                results.append({
                    "model": model,
                    "benchmark": benchmark,
                    "success": False,
                    "error": "no_evaluation_returned"
                })
                
        except atlas.APITimeoutError:
            print(f"⏰ Timeout after {current_timeout/60:.1f} minutes")
            consecutive_timeouts += 1
            
            results.append({
                "model": model,
                "benchmark": benchmark,
                "success": False,
                "error": "timeout",
                "timeout_used": current_timeout
            })
            
            # Increase timeout after consecutive timeouts
            if consecutive_timeouts >= 2:
                current_timeout = min(3600.0, current_timeout * 1.5)  # Max 1 hour
                print(f"🔄 Increased timeout to {current_timeout/60:.1f} minutes")
                consecutive_timeouts = 0  # Reset counter after adjustment
        
        except atlas.APIError as e:
            print(f"❌ API error: {e}")
            results.append({
                "model": model,
                "benchmark": benchmark,
                "success": False,
                "error": str(e)
            })
        
        # Brief pause between requests
        time.sleep(1.0)
    
    # Summary
    successful = [r for r in results if r["success"]]
    timeouts = [r for r in results if r.get("error") == "timeout"]
    
    print(f"\n📊 Batch Summary:")
    print(f"   Total requests: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Timeouts: {len(timeouts)}")
    print(f"   Other errors: {len(results) - len(successful) - len(timeouts)}")
    
    return results

# Usage
pairs = [
    ("gpt-4", "mmlu"),
    ("claude-3-opus", "hellaswag"),
    ("llama-2-70b", "arc-challenge"),
    ("gpt-3.5-turbo", "gsm8k"),
]

batch_results = batch_evaluations_with_adaptive_timeout(pairs)
```

## Error Handling and Recovery

### Timeout-Specific Error Handling

```python
import atlas
from atlas import Atlas
import time

def handle_timeout_gracefully(operation_func, *args, **kwargs):
    """Generic timeout handler for any Atlas operation"""
    max_retries = 3
    base_timeout = 60.0
    
    for attempt in range(max_retries):
        # Calculate timeout for this attempt
        attempt_timeout = base_timeout * (2 ** attempt)  # Exponential increase
        
        print(f"🔄 Attempt {attempt + 1}/{max_retries} (timeout: {attempt_timeout/60:.1f}min)")
        
        try:
            result = operation_func(timeout=attempt_timeout, *args, **kwargs)
            print(f"✅ Operation succeeded on attempt {attempt + 1}")
            return result
            
        except atlas.APITimeoutError:
            print(f"⏰ Timeout on attempt {attempt + 1}")
            
            if attempt == max_retries - 1:
                print("❌ All retry attempts exhausted")
                raise
            else:
                wait_time = 5 * (attempt + 1)  # Progressive wait
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        except atlas.APIError as e:
            print(f"❌ Non-timeout error: {e}")
            raise  # Don't retry non-timeout errors

def create_evaluation_with_timeout_handling(model: str, benchmark: str):
    """Wrapper function for evaluation creation"""
    def operation_func(timeout, *args, **kwargs):
        client = Atlas(timeout=timeout)
        return client.evaluations.create(model=model, benchmark=benchmark)
    
    return handle_timeout_gracefully(operation_func)

def get_results_with_timeout_handling(evaluation_id: str):
    """Wrapper function for results retrieval"""
    def operation_func(timeout, *args, **kwargs):
        client = Atlas(timeout=timeout)
        return client.results.get(evaluation_id=evaluation_id)
    
    return handle_timeout_gracefully(operation_func)

# Usage
try:
    evaluation = create_evaluation_with_timeout_handling("gpt-4", "mmlu")
    if evaluation:
        results = get_results_with_timeout_handling(evaluation.id)
        print(f"📊 Retrieved {len(results) if results else 0} results")
        
except atlas.APITimeoutError:
    print("❌ Operation failed due to persistent timeouts")
except atlas.APIError as e:
    print(f"❌ Operation failed: {e}")
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum
import atlas
from atlas import Atlas

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, don't try
    HALF_OPEN = "half_open" # Testing if recovered

class TimeoutCircuitBreaker:
    """Circuit breaker specifically for timeout management"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_threshold: float = 300.0,  # 5 minutes
                 recovery_timeout: int = 60):       # 1 minute
        self.failure_threshold = failure_threshold
        self.timeout_threshold = timeout_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.current_timeout = 120.0  # Start with 2 minutes
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if (time.time() - self.last_failure_time) < self.recovery_timeout:
                raise atlas.APIConnectionError(
                    message="Circuit breaker is OPEN - too many recent timeouts"
                )
            else:
                self.state = CircuitState.HALF_OPEN
                print("🔄 Circuit breaker transitioning to HALF_OPEN")
        
        try:
            # Use adaptive timeout
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.current_timeout
            
            print(f"🔄 Calling function with {self.current_timeout/60:.1f}min timeout")
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            self.on_success()
            return result
            
        except atlas.APITimeoutError as e:
            self.on_timeout_failure()
            raise
        except atlas.APIError as e:
            # Non-timeout API errors don't affect circuit state
            raise
    
    def on_success(self):
        """Handle successful operation"""
        print("✅ Circuit breaker: Operation succeeded")
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
        # Gradually reduce timeout on success
        self.current_timeout = max(60.0, self.current_timeout * 0.95)
    
    def on_timeout_failure(self):
        """Handle timeout failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        print(f"⏰ Circuit breaker: Timeout failure {self.failure_count}/{self.failure_threshold}")
        
        # Increase timeout for next attempt
        self.current_timeout = min(self.timeout_threshold, self.current_timeout * 1.5)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print("🔴 Circuit breaker: OPEN - too many consecutive timeouts")

# Usage with circuit breaker
def protected_atlas_operations():
    """Example of using circuit breaker with Atlas operations"""
    breaker = TimeoutCircuitBreaker(
        failure_threshold=3,
        timeout_threshold=600.0,  # Max 10 minutes
        recovery_timeout=120      # 2 minute recovery time
    )
    
    def create_evaluation_protected(model: str, benchmark: str):
        def operation(timeout):
            client = Atlas(timeout=timeout)
            return client.evaluations.create(model=model, benchmark=benchmark)
        return breaker.call(operation)
    
    def get_results_protected(evaluation_id: str):
        def operation(timeout):
            client = Atlas(timeout=timeout)
            return client.results.get(evaluation_id=evaluation_id)
        return breaker.call(operation)
    
    # Test with multiple operations
    operations = [
        ("gpt-4", "mmlu"),
        ("claude-3-opus", "hellaswag"),
        ("llama-2-70b", "gsm8k"),
    ]
    
    successful_evaluations = []
    
    for model, benchmark in operations:
        try:
            print(f"\n🔄 Creating evaluation: {model} + {benchmark}")
            evaluation = create_evaluation_protected(model, benchmark)
            
            if evaluation:
                successful_evaluations.append(evaluation)
                print(f"✅ Success: {evaluation.id}")
                
                # Try to get results
                print(f"🔄 Getting results for {evaluation.id}")
                results = get_results_protected(evaluation.id)
                
                if results:
                    print(f"📊 Retrieved {len(results)} results")
                
        except atlas.APIConnectionError as e:
            if "Circuit breaker is OPEN" in str(e):
                print("🔴 Circuit breaker prevented operation")
                print(f"⏳ Waiting {breaker.recovery_timeout}s for recovery...")
                time.sleep(breaker.recovery_timeout)
            else:
                print(f"❌ Connection error: {e}")
                
        except atlas.APITimeoutError:
            print("⏰ Timeout occurred - circuit breaker updated")
            
        except atlas.APIError as e:
            print(f"❌ API error: {e}")
    
    print(f"\n📈 Final Results:")
    print(f"   Circuit state: {breaker.state.value}")
    print(f"   Current timeout: {breaker.current_timeout/60:.1f} minutes")
    print(f"   Successful evaluations: {len(successful_evaluations)}")
    
    return successful_evaluations

# Run protected operations
results = protected_atlas_operations()
```

## Monitoring and Metrics

### Timeout Performance Tracking

```python
import time
from dataclasses import dataclass
from typing import List, Optional
from atlas import Atlas
import atlas

@dataclass
class TimeoutMetrics:
    operation: str
    model: str
    benchmark: str
    timeout_set: float
    actual_duration: float
    success: bool
    error_type: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class TimeoutMonitor:
    """Monitor and analyze timeout patterns"""
    
    def __init__(self):
        self.metrics: List[TimeoutMetrics] = []
    
    def record_operation(self, operation: str, model: str, benchmark: str, 
                        timeout_set: float, start_time: float, success: bool, 
                        error_type: str = None):
        """Record an operation's timeout metrics"""
        actual_duration = time.time() - start_time
        
        metric = TimeoutMetrics(
            operation=operation,
            model=model,
            benchmark=benchmark,
            timeout_set=timeout_set,
            actual_duration=actual_duration,
            success=success,
            error_type=error_type
        )
        
        self.metrics.append(metric)
        
        print(f"📊 Recorded: {operation} took {actual_duration:.1f}s (timeout: {timeout_set:.1f}s)")
    
    def get_timeout_efficiency(self) -> dict:
        """Analyze timeout efficiency"""
        if not self.metrics:
            return {}
        
        successful_ops = [m for m in self.metrics if m.success]
        timeout_ops = [m for m in self.metrics if m.error_type == "timeout"]
        
        analysis = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "timeout_operations": len(timeout_ops),
            "success_rate": len(successful_ops) / len(self.metrics),
            "timeout_rate": len(timeout_ops) / len(self.metrics),
        }
        
        if successful_ops:
            avg_success_duration = sum(m.actual_duration for m in successful_ops) / len(successful_ops)
            avg_success_timeout = sum(m.timeout_set for m in successful_ops) / len(successful_ops)
            
            analysis.update({
                "avg_success_duration": avg_success_duration,
                "avg_success_timeout_set": avg_success_timeout,
                "timeout_efficiency": avg_success_duration / avg_success_timeout if avg_success_timeout > 0 else 0
            })
        
        return analysis
    
    def suggest_optimal_timeouts(self) -> dict:
        """Suggest optimal timeouts based on historical data"""
        if not self.metrics:
            return {"message": "No data available"}
        
        # Group by operation type
        by_operation = {}
        for metric in self.metrics:
            if metric.success:  # Only use successful operations
                key = (metric.operation, metric.model, metric.benchmark)
                if key not in by_operation:
                    by_operation[key] = []
                by_operation[key].append(metric.actual_duration)
        
        suggestions = {}
        for (operation, model, benchmark), durations in by_operation.items():
            # Suggest timeout as 95th percentile + 50% buffer
            durations.sort()
            p95_index = int(len(durations) * 0.95)
            p95_duration = durations[p95_index] if p95_index < len(durations) else durations[-1]
            suggested_timeout = p95_duration * 1.5  # 50% buffer
            
            suggestions[f"{operation}_{model}_{benchmark}"] = {
                "suggested_timeout": suggested_timeout,
                "based_on_operations": len(durations),
                "p95_actual_duration": p95_duration
            }
        
        return suggestions

def monitored_atlas_operations():
    """Example of Atlas operations with timeout monitoring"""
    monitor = TimeoutMonitor()
    client = Atlas()
    
    test_operations = [
        ("gpt-3.5-turbo", "arc-easy", 60.0),    # Should be fast
        ("gpt-4", "mmlu", 180.0),               # Medium complexity
        ("claude-3-opus", "math", 600.0),       # Complex, longer timeout
    ]
    
    for model, benchmark, timeout in test_operations:
        print(f"\n🔄 Testing {model} + {benchmark} (timeout: {timeout/60:.1f}min)")
        
        # Evaluation creation
        start_time = time.time()
        try:
            evaluation = client.with_options(timeout=timeout).evaluations.create(
                model=model,
                benchmark=benchmark
            )
            
            if evaluation:
                monitor.record_operation("create_evaluation", model, benchmark, 
                                       timeout, start_time, True)
                
                # Results retrieval
                start_time = time.time()
                try:
                    results = client.with_options(timeout=timeout).results.get(
                        evaluation_id=evaluation.id
                    )
                    
                    success = results is not None
                    monitor.record_operation("get_results", model, benchmark,
                                           timeout, start_time, success,
                                           None if success else "no_results")
                    
                except atlas.APITimeoutError:
                    monitor.record_operation("get_results", model, benchmark,
                                           timeout, start_time, False, "timeout")
                except atlas.APIError as e:
                    monitor.record_operation("get_results", model, benchmark,
                                           timeout, start_time, False, str(e))
            else:
                monitor.record_operation("create_evaluation", model, benchmark,
                                       timeout, start_time, False, "no_evaluation")
                
        except atlas.APITimeoutError:
            monitor.record_operation("create_evaluation", model, benchmark,
                                   timeout, start_time, False, "timeout")
        except atlas.APIError as e:
            monitor.record_operation("create_evaluation", model, benchmark,
                                   timeout, start_time, False, str(e))
    
    # Analyze results
    print(f"\n📊 Timeout Analysis:")
    efficiency = monitor.get_timeout_efficiency()
    
    for key, value in efficiency.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n💡 Timeout Suggestions:")
    suggestions = monitor.suggest_optimal_timeouts()
    for operation, suggestion in suggestions.items():
        print(f"   {operation}:")
        print(f"     Suggested timeout: {suggestion['suggested_timeout']:.0f}s")
        print(f"     Based on {suggestion['based_on_operations']} successful operations")
        print(f"     95th percentile duration: {suggestion['p95_actual_duration']:.1f}s")

# Run monitoring example
monitored_atlas_operations()
```
