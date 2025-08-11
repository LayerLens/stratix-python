# Rate Limiting

This guide covers how to handle rate limiting when using the Atlas Python SDK, including best practices for avoiding rate limits and properly handling rate limit errors.

## Identifying Rate Limit Errors

### Rate Limit HTTP Response

When you exceed rate limits, the API returns a `429 Too Many Requests` status:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    
    # Making too many requests quickly
    for i in range(100):
        evaluation = client.evaluations.create(
            model="gpt-4", 
            benchmark="mmlu"
        )
        
except atlas.RateLimitError as e:
    print(f"Rate limited: {e}")
    print(f"Status code: {e.status_code}")  # 429
    print(f"Response headers: {dict(e.response.headers)}")
```

### Rate Limit Headers

The API response includes helpful headers:

```python
import atlas
from atlas import Atlas

def inspect_rate_limit_headers(error):
    """Inspect rate limit headers from error response"""
    headers = error.response.headers
    
    # Common rate limit headers
    rate_limit_info = {
        'retry_after': headers.get('retry-after'),
        'x_ratelimit_limit': headers.get('x-ratelimit-limit'),
        'x_ratelimit_remaining': headers.get('x-ratelimit-remaining'),
        'x_ratelimit_reset': headers.get('x-ratelimit-reset'),
    }
    
    print("Rate limit information:")
    for key, value in rate_limit_info.items():
        if value:
            print(f"  {key}: {value}")

try:
    client = Atlas()
    # ... make request that triggers rate limit
    
except atlas.RateLimitError as e:
    inspect_rate_limit_headers(e)
```

## Handling Rate Limits

### Basic Retry with Backoff

```python
import time
import random
import atlas
from atlas import Atlas

def create_evaluation_with_retry(model: str, benchmark: str, max_retries: int = 3):
    """Create evaluation with rate limit retry logic"""
    client = Atlas()
    
    for attempt in range(max_retries):
        try:
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            
            if evaluation:
                print(f"✅ Success on attempt {attempt + 1}")
                return evaluation
                
        except atlas.RateLimitError as e:
            print(f"⏳ Rate limited on attempt {attempt + 1}")
            
            # Check if server provided retry-after header
            retry_after = e.response.headers.get('retry-after')
            
            if retry_after:
                wait_time = int(retry_after)
                print(f"   Server requests waiting {wait_time} seconds")
            else:
                # Exponential backoff with jitter
                base_wait = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait_time = base_wait + jitter
                print(f"   Using exponential backoff: {wait_time:.1f} seconds")
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                print(f"❌ Exhausted all {max_retries} retry attempts")
                raise
                
        except atlas.APIError as e:
            print(f"❌ Non-rate-limit error: {e}")
            raise
    
    return None

# Usage
evaluation = create_evaluation_with_retry("gpt-4", "mmlu")
```

### Advanced Retry Strategies

#### Exponential Backoff with Jitter

```python
import time
import random
import atlas
from atlas import Atlas

class ExponentialBackoffRetry:
    """Implement exponential backoff with jitter for rate limit handling"""
    
    def __init__(self, max_retries=5, base_delay=1.0, max_delay=60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def calculate_delay(self, attempt: int, retry_after: str = None) -> float:
        """Calculate delay before next retry"""
        
        # If server provided retry-after, use that
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        
        # Exponential backoff: 2^attempt * base_delay
        delay = self.base_delay * (2 ** attempt)
        
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * random.uniform(-1, 1)
        delay += jitter
        
        # Cap at maximum delay
        return min(delay, self.max_delay)
    
    def retry_operation(self, operation_func, *args, **kwargs):
        """Retry operation with exponential backoff"""
        
        for attempt in range(self.max_retries):
            try:
                return operation_func(*args, **kwargs)
                
            except atlas.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    # Last attempt - re-raise the error
                    raise
                
                retry_after = e.response.headers.get('retry-after')
                delay = self.calculate_delay(attempt, retry_after)
                
                print(f"⏳ Rate limited (attempt {attempt + 1}/{self.max_retries})")
                print(f"   Waiting {delay:.1f} seconds before retry...")
                
                time.sleep(delay)
                continue
                
            except atlas.APIError as e:
                # Don't retry other API errors
                print(f"❌ Non-retryable error: {e}")
                raise

# Usage
backoff = ExponentialBackoffRetry(max_retries=5, base_delay=2.0, max_delay=120.0)

def create_evaluation():
    client = Atlas()
    return client.evaluations.create(model="gpt-4", benchmark="mmlu")

evaluation = backoff.retry_operation(create_evaluation)
```


## Proactive Rate Limit Management

### Request Throttling

```python
import time
from threading import Lock
from datetime import datetime, timedelta
import atlas
from atlas import Atlas

class ThrottledAtlasClient:
    """Atlas client with built-in request throttling"""
    
    def __init__(self, requests_per_minute=30, **client_kwargs):
        self.client = Atlas(**client_kwargs)
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = None
        self.lock = Lock()
    
    def _wait_for_next_request(self):
        """Wait if necessary to maintain rate limit"""
        with self.lock:
            if self.last_request_time:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    print(f"⏳ Throttling: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            
            self.last_request_time = time.time()
    
    def create_evaluation(self, *args, **kwargs):
        """Create evaluation with throttling"""
        self._wait_for_next_request()
        return self.client.evaluations.create(*args, **kwargs)
    
    def get_results(self, *args, **kwargs):
        """Get results with throttling"""
        self._wait_for_next_request()
        return self.client.results.get(*args, **kwargs)

# Usage
throttled_client = ThrottledAtlasClient(requests_per_minute=20)

# These requests will be automatically throttled
evaluations = []
for i in range(10):
    evaluation = throttled_client.create_evaluation(
        model="gpt-4",
        benchmark="mmlu"
    )
    evaluations.append(evaluation)
```

### Batch Request Management

```python
import time
from typing import List, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import atlas
from atlas import Atlas

class BatchRequestManager:
    """Manage batch requests with rate limiting"""
    
    def __init__(self, requests_per_minute=30, max_concurrent=5):
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self.request_interval = 60.0 / requests_per_minute
        
    def execute_batch(self, operations: List[Tuple[Callable, tuple, dict]], 
                     handle_rate_limits=True) -> List[Any]:
        """Execute a batch of operations with rate limiting"""
        
        results = []
        
        if self.max_concurrent == 1 or not handle_rate_limits:
            # Sequential execution
            for i, (func, args, kwargs) in enumerate(operations):
                if i > 0 and handle_rate_limits:
                    time.sleep(self.request_interval)
                
                try:
                    result = func(*args, **kwargs)
                    results.append({"success": True, "result": result, "index": i})
                except Exception as e:
                    results.append({"success": False, "error": e, "index": i})
        else:
            # Concurrent execution with rate limiting
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_index = {}
                
                for i, (func, args, kwargs) in enumerate(operations):
                    if i > 0 and handle_rate_limits:
                        # Stagger request submissions
                        time.sleep(self.request_interval / self.max_concurrent)
                    
                    future = executor.submit(self._execute_with_retry, func, args, kwargs)
                    future_to_index[future] = i
                
                # Collect results
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append({"success": True, "result": result, "index": index})
                    except Exception as e:
                        results.append({"success": False, "error": e, "index": index})
        
        # Sort results by original order
        results.sort(key=lambda x: x["index"])
        return results
    
    def _execute_with_retry(self, func, args, kwargs, max_retries=3):
        """Execute operation with retry on rate limit"""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except atlas.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                
                retry_after = e.response.headers.get('retry-after', 60)
                wait_time = int(retry_after)
                time.sleep(wait_time)

# Usage
client = Atlas()
batch_manager = BatchRequestManager(requests_per_minute=20, max_concurrent=3)

# Prepare batch operations
operations = []
models = ["gpt-4", "claude-3-opus", "gpt-3.5-turbo"] * 5

for model in models:
    operation = (
        client.evaluations.create,  # function
        (),                         # args
        {"model": model, "benchmark": "mmlu"}  # kwargs
    )
    operations.append(operation)

# Execute batch
print(f"📦 Executing batch of {len(operations)} operations...")
results = batch_manager.execute_batch(operations)

# Process results
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"✅ Successful: {len(successful)}")
print(f"❌ Failed: {len(failed)}")

for result in failed:
    print(f"   Failed operation {result['index']}: {result['error']}")
```

## Monitoring Rate Limits

### Rate Limit Usage Tracking

```python
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List
import atlas
from atlas import Atlas

class RateLimitMonitor:
    """Monitor and track rate limit usage"""
    
    def __init__(self, window_minutes=60):
        self.window_minutes = window_minutes
        self.request_times = deque()
        self.rate_limit_events = []
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
    def record_request(self, operation: str):
        """Record a successful request"""
        now = datetime.now()
        self.request_times.append(now)
        self.operation_counts[operation] += 1
        self._cleanup_old_data(now)
    
    def record_rate_limit(self, operation: str, retry_after: int = None):
        """Record a rate limit event"""
        event = {
            'timestamp': datetime.now(),
            'operation': operation,
            'retry_after': retry_after
        }
        self.rate_limit_events.append(event)
        self.error_counts['rate_limit'] += 1
    
    def _cleanup_old_data(self, current_time: datetime):
        """Remove data outside monitoring window"""
        cutoff = current_time - timedelta(minutes=self.window_minutes)
        
        # Clean request times
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        
        # Clean rate limit events
        self.rate_limit_events = [
            event for event in self.rate_limit_events
            if event['timestamp'] > cutoff
        ]
    
    def get_current_rate(self) -> float:
        """Get current requests per minute"""
        self._cleanup_old_data(datetime.now())
        
        if not self.request_times:
            return 0.0
        
        # Calculate rate over actual time window
        time_span = (datetime.now() - self.request_times[0]).total_seconds() / 60
        return len(self.request_times) / max(time_span, 1)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive rate limit statistics"""
        self._cleanup_old_data(datetime.now())
        
        recent_rate_limits = len(self.rate_limit_events)
        total_requests = len(self.request_times)
        
        return {
            'current_rate_per_minute': self.get_current_rate(),
            'total_requests_in_window': total_requests,
            'rate_limit_events': recent_rate_limits,
            'rate_limit_percentage': (recent_rate_limits / max(total_requests, 1)) * 100,
            'operation_breakdown': dict(self.operation_counts),
            'last_rate_limit': max([e['timestamp'] for e in self.rate_limit_events], 
                                 default=None)
        }
    
    def should_slow_down(self, threshold_percentage=5) -> bool:
        """Check if we should slow down requests based on rate limits"""
        stats = self.get_statistics()
        return stats['rate_limit_percentage'] > threshold_percentage

class MonitoredAtlasClient:
    """Atlas client with rate limit monitoring"""
    
    def __init__(self, **client_kwargs):
        self.client = Atlas(**client_kwargs)
        self.monitor = RateLimitMonitor()
    
    def create_evaluation(self, *args, **kwargs):
        """Create evaluation with monitoring"""
        try:
            result = self.client.evaluations.create(*args, **kwargs)
            self.monitor.record_request('create_evaluation')
            
            # Adaptive slowdown
            if self.monitor.should_slow_down():
                print("⚠️ High rate limit percentage detected, slowing down...")
                time.sleep(2)
            
            return result
            
        except atlas.RateLimitError as e:
            retry_after = e.response.headers.get('retry-after')
            self.monitor.record_rate_limit('create_evaluation', retry_after)
            raise
    
    def get_results(self, *args, **kwargs):
        """Get results with monitoring"""
        try:
            result = self.client.results.get(*args, **kwargs)
            self.monitor.record_request('get_results')
            return result
            
        except atlas.RateLimitError as e:
            retry_after = e.response.headers.get('retry-after')
            self.monitor.record_rate_limit('get_results', retry_after)
            raise
    
    def print_statistics(self):
        """Print current rate limit statistics"""
        stats = self.monitor.get_statistics()
        
        print("📊 Rate Limit Statistics (last hour):")
        print(f"   Current rate: {stats['current_rate_per_minute']:.1f} requests/min")
        print(f"   Total requests: {stats['total_requests_in_window']}")
        print(f"   Rate limit events: {stats['rate_limit_events']}")
        print(f"   Rate limit percentage: {stats['rate_limit_percentage']:.1f}%")
        
        if stats['operation_breakdown']:
            print("   Operations:")
            for op, count in stats['operation_breakdown'].items():
                print(f"     {op}: {count}")
        
        if stats['last_rate_limit']:
            print(f"   Last rate limit: {stats['last_rate_limit']}")

# Usage
monitored_client = MonitoredAtlasClient()

# Make requests and monitor
for i in range(20):
    try:
        evaluation = monitored_client.create_evaluation(
            model="gpt-4",
            benchmark="mmlu"
        )
        print(f"✅ Evaluation {i+1} created")
        
        if i % 5 == 0:  # Print stats every 5 requests
            monitored_client.print_statistics()
            
    except atlas.RateLimitError:
        print(f"⏳ Rate limited on request {i+1}")
        time.sleep(30)  # Wait before continuing

# Final statistics
monitored_client.print_statistics()
```

## Best Practices Summary

### 1. Implement Proper Retry Logic
```python
# ✅ Good: Exponential backoff with jitter
def robust_request(operation_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation_func()
        except atlas.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            # Use server-suggested wait time if available
            retry_after = e.response.headers.get('retry-after', 2 ** attempt)
            wait_time = int(retry_after) + random.uniform(0, 1)
            time.sleep(wait_time)
```

### 2. Respect Server Headers
```python
# ✅ Good: Check retry-after header
except atlas.RateLimitError as e:
    retry_after = e.response.headers.get('retry-after')
    if retry_after:
        time.sleep(int(retry_after))
```

### 3. Monitor Your Usage
```python
# ✅ Good: Track your rate limit usage
monitor = RateLimitMonitor()
# ... use monitor to adjust request patterns
```

### 4. Use Appropriate Request Rates
```python
# ✅ Good: Conservative request rate
throttled_client = ThrottledAtlasClient(requests_per_minute=20)

# ❌ Bad: Aggressive request rate
# aggressive_client = ThrottledAtlasClient(requests_per_minute=1000)
```

### 5. Handle Rate Limits Gracefully
```python
# ✅ Good: Graceful handling
try:
    result = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.RateLimitError:
    # Log the event, wait, and potentially retry
    logger.warning("Rate limit hit, backing off")
    time.sleep(60)
```
