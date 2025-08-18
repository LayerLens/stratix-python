# Error Codes Reference

This reference guide provides detailed information about all error codes and exceptions in the Atlas Python SDK.

## Exception Hierarchy

```
AtlasError (Base exception)
├── APIError (Base for API-related errors)
│   ├── APIConnectionError (Network/connection issues)
│   │   └── APITimeoutError (Request timeouts)
│   ├── APIResponseValidationError (Invalid response format)
│   └── APIStatusError (HTTP status errors)
│       ├── BadRequestError (400)
│       ├── AuthenticationError (401)
│       ├── PermissionDeniedError (403)
│       ├── NotFoundError (404)
│       ├── ConflictError (409)
│       ├── UnprocessableEntityError (422)
│       ├── RateLimitError (429)
│       └── InternalServerError (500+)
```

## HTTP Status Code Errors

### 400 - Bad Request (`BadRequestError`)

**When it occurs**:

- Invalid request parameters
- Missing required fields
- Malformed request data

**Common causes**:

```python
# Empty or invalid parameters
client.evaluations.create(model="", benchmark="")  # Empty strings
client.evaluations.create(model=None, benchmark="mmlu")  # None values

# Invalid parameter types
client.evaluations.create(model=123, benchmark="mmlu")  # Wrong type
```

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="", benchmark="mmlu")
except atlas.BadRequestError as e:
    print(f"Bad request: {e}")
    print(f"Status code: {e.status_code}")  # 400
    print(f"Response body: {e.body}")
```

**Solutions**:

1. **Validate parameters before making requests**:

   ```python
   def validate_evaluation_params(model, benchmark):
       if not model or not isinstance(model, str):
           raise ValueError("Model must be a non-empty string")
       if not benchmark or not isinstance(benchmark, str):
           raise ValueError("Benchmark must be a non-empty string")
       return True

   if validate_evaluation_params(model, benchmark):
       evaluation = client.evaluations.create(model=model, benchmark=benchmark)
   ```

2. **Check parameter format requirements**:

   ```python
   # Ensure parameters meet expected format
   model = model.strip() if model else ""
   benchmark = benchmark.strip() if benchmark else ""

   if len(model) < 2 or len(benchmark) < 2:
       raise ValueError("Model and benchmark names must be at least 2 characters")
   ```

### 401 - Unauthorized (`AuthenticationError`)

**When it occurs**:

- Missing API key
- Invalid or expired API key
- API key format issues

**Common causes**:

```python
# Missing API key
client = Atlas(api_key=None)

# Invalid API key format
client = Atlas(api_key="invalid-key")

# Expired API key (need to regenerate)
client = Atlas(api_key="sk-old-expired-key")
```

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas(api_key="invalid-key")
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print(f"Status code: {e.status_code}")  # 401
    print(f"Request ID: {e.request_id}")
```

**Solutions**:

1. **Verify API key configuration**:

   ```python
   import os

   api_key = os.getenv('LAYERLENS_ATLAS_API_KEY')
   if not api_key:
       print("❌ API key not found in environment variables")
   elif len(api_key) < 10:
       print("⚠️ API key seems too short")
   else:
       print("✅ API key found and looks valid")
   ```

2. **Regenerate API key**:

   - Log into Atlas dashboard
   - Go to Settings > API Keys
   - Generate new API key
   - Update environment variables

3. **Test authentication separately**:

   ```python
   def test_authentication(api_key):
       try:
           client = Atlas(api_key=api_key)
           # Try minimal operation to test auth
           client.evaluations.create(model="test", benchmark="test")
       except atlas.AuthenticationError:
           return False, "Invalid API key"
       except atlas.NotFoundError:
           return True, "Authentication successful (test resources not found is expected)"
       except Exception as e:
           return False, f"Unexpected error: {e}"

   is_valid, message = test_authentication(your_api_key)
   print(f"Authentication test: {message}")
   ```

### 403 - Forbidden (`PermissionDeniedError`)

**When it occurs**:

- Valid API key but insufficient permissions
- No access to specific models or benchmarks
- Organization/project access issues

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="restricted-model", benchmark="mmlu")
except atlas.PermissionDeniedError as e:
    print(f"Permission denied: {e}")
    print(f"Status code: {e.status_code}")  # 403
    print(f"Response body: {e.body}")
```

**Solutions**:

1. **Test access to different resources**:

   ```python
   def test_resource_access(models, benchmarks):
       client = Atlas()
       access_matrix = {}

       for model in models:
           access_matrix[model] = {}
           for benchmark in benchmarks:
               try:
                   evaluation = client.evaluations.create(model=model, benchmark=benchmark)
                   access_matrix[model][benchmark] = "✅ Access granted"
               except atlas.PermissionDeniedError:
                   access_matrix[model][benchmark] = "❌ Permission denied"
               except atlas.NotFoundError:
                   access_matrix[model][benchmark] = "❓ Resource not found"
               except Exception as e:
                   access_matrix[model][benchmark] = f"❓ {type(e).__name__}"

       return access_matrix

   # Test common resources
   models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]
   benchmarks = ["mmlu", "hellaswag", "arc-easy"]

   access = test_resource_access(models, benchmarks)
   ```

2. **Contact administrator for access**:
   - Request access to specific models or benchmarks
   - Verify project membership
   - Check organization-level permissions

### 404 - Not Found (`NotFoundError`)

**When it occurs**:

- Model ID doesn't exist
- Benchmark ID doesn't exist
- Evaluation ID not found (for results)
- Resource doesn't exist in your organization

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="nonexistent-model", benchmark="mmlu")
except atlas.NotFoundError as e:
    print(f"Resource not found: {e}")
    print(f"Status code: {e.status_code}")  # 404
```

**Solutions**:

1. **Verify resource names**:

   ```python
   def find_available_models():
       """Try common model names to find available ones"""
       client = Atlas()

       common_models = [
           "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo",
           "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
           "llama-2-70b", "llama-2-13b", "mistral-7b"
       ]

       available_models = []

       for model in common_models:
           try:
               # Test with common benchmark
               evaluation = client.evaluations.create(model=model, benchmark="mmlu")
               if evaluation:
                   available_models.append(model)
           except atlas.NotFoundError:
               # Model or benchmark not found
               continue
           except atlas.PermissionDeniedError:
               # Model exists but no permission
               available_models.append(f"{model} (no permission)")
           except Exception:
               # Other errors - model might exist
               available_models.append(f"{model} (unknown status)")

       return available_models

   available = find_available_models()
   print(f"Available models: {available}")
   ```

2. **Check spelling and case sensitivity**:

   ```python
   # Common mistakes
   correct_names = {
       "GPT-4": "gpt-4",
       "GPT4": "gpt-4",
       "MMLU": "mmlu",
       "HellaSwag": "hellaswag",
       "arc_challenge": "arc-challenge"  # Underscore vs hyphen
   }
   ```

3. **Use exact names from Atlas dashboard**:
   - Log into Atlas dashboard
   - Check available models and benchmarks
   - Copy exact names (case-sensitive)

### 409 - Conflict (`ConflictError`)

**When it occurs**:

- Resource already exists
- Conflicting operation in progress
- State conflict (e.g., trying to modify completed evaluation)

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    # Some operation that conflicts with current state
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.ConflictError as e:
    print(f"Conflict error: {e}")
    print(f"Status code: {e.status_code}")  # 409
```

**Solutions**:

1. **Check current resource state**
2. **Wait for ongoing operations to complete**
3. **Use different resource identifiers**

### 422 - Unprocessable Entity (`UnprocessableEntityError`)

**When it occurs**:

- Valid request format but business logic prevents processing
- Parameter combinations that don't make sense
- Resource constraints exceeded

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="invalid-benchmark")
except atlas.UnprocessableEntityError as e:
    print(f"Unprocessable entity: {e}")
    print(f"Status code: {e.status_code}")  # 422
    print(f"Response details: {e.body}")
```

**Solutions**:

1. **Check business logic constraints**
2. **Verify parameter combinations are valid**
3. **Review API documentation for limitations**

### 429 - Rate Limited (`RateLimitError`)

**When it occurs**:

- Too many requests in short time period
- API rate limits exceeded
- Organization-level quotas reached

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()

    # Making too many requests quickly
    for i in range(100):
        evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

except atlas.RateLimitError as e:
    print(f"Rate limited: {e}")
    print(f"Status code: {e.status_code}")  # 429
    print(f"Retry after: {e.response.headers.get('retry-after', 'not specified')}")
```

**Solutions**:

1. **Implement retry with backoff**:

   ```python
   import time
   import atlas
   from atlas import Atlas

   def create_evaluation_with_rate_limit_handling(model, benchmark, max_retries=3):
       client = Atlas()

       for attempt in range(max_retries):
           try:
               return client.evaluations.create(model=model, benchmark=benchmark)

           except atlas.RateLimitError as e:
               retry_after = e.response.headers.get('retry-after')

               if retry_after:
                   wait_time = int(retry_after)
                   print(f"Rate limited. Waiting {wait_time}s as requested...")
               else:
                   wait_time = (2 ** attempt) * 60  # Exponential backoff
                   print(f"Rate limited. Waiting {wait_time}s...")

               if attempt < max_retries - 1:
                   time.sleep(wait_time)
               else:
                   raise  # Re-raise on final attempt

       return None

   evaluation = create_evaluation_with_rate_limit_handling("gpt-4", "mmlu")
   ```

2. **Add delays between requests**:

   ```python
   import time

   evaluations = []
   models = ["gpt-4", "claude-3-opus", "llama-2-70b"]

   for model in models:
       evaluation = client.evaluations.create(model=model, benchmark="mmlu")
       evaluations.append(evaluation)

       # Wait between requests to avoid rate limits
       time.sleep(2)  # 2-second delay
   ```

3. **Monitor rate limit headers**:
   ```python
   def monitor_rate_limits(client):
       """Monitor rate limit status"""
       # This would require SDK modification to expose headers
       # Check with LayerLens documentation for rate limit details
       pass
   ```

### 500+ - Server Errors (`InternalServerError`)

**When it occurs**:

- Atlas API server errors
- Temporary service unavailability
- Infrastructure issues

**Example error**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.InternalServerError as e:
    print(f"Server error: {e}")
    print(f"Status code: {e.status_code}")  # 500, 502, 503, etc.
    print(f"Request ID: {e.request_id}")  # Include in support requests
```

**Solutions**:

1. **Implement retry logic**:

   ```python
   import time
   import atlas
   from atlas import Atlas

   def create_evaluation_with_server_error_handling(model, benchmark):
       client = Atlas()
       max_retries = 3
       base_delay = 5  # seconds

       for attempt in range(max_retries):
           try:
               return client.evaluations.create(model=model, benchmark=benchmark)

           except atlas.InternalServerError as e:
               print(f"Server error on attempt {attempt + 1}: {e}")

               if attempt < max_retries - 1:
                   # Exponential backoff with jitter
                   delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                   print(f"Retrying in {delay:.1f}s...")
                   time.sleep(delay)
               else:
                   print(f"All {max_retries} attempts failed. Request ID: {e.request_id}")
                   raise

       return None
   ```

2. **Check service status**:

   - Visit LayerLens status page
   - Check for ongoing incidents
   - Monitor Atlas service announcements

3. **Report persistent issues**:
   - Include request ID from error
   - Provide timestamp and error details
   - Contact LayerLens support

## Connection Errors

### `APIConnectionError`

**When it occurs**:

- Network connectivity issues
- DNS resolution failures
- Firewall blocking requests
- Proxy configuration problems

**Example**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas(timeout=10.0)
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIConnectionError as e:
    print(f"Connection error: {e}")
    print(f"Request URL: {e.request.url}")
```

**Solutions**:

1. **Test basic connectivity**:

   ```bash
   ping api.layerlens.com
   curl -I https://api.layerlens.com
   ```

2. **Check proxy/firewall settings**
3. **Verify DNS resolution**

### `APITimeoutError`

**When it occurs**:

- Request takes longer than configured timeout
- Network latency issues
- Server processing delays

**Example**:

```python
import atlas
from atlas import Atlas

try:
    client = Atlas(timeout=30.0)  # 30-second timeout
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APITimeoutError as e:
    print(f"Request timed out: {e}")
```

**Solutions**:

1. **Increase timeout**:

   ```python
   client = Atlas(timeout=600.0)  # 10 minutes
   ```

2. **Use appropriate timeouts for operation type**:

   ```python
   # Quick operations
   quick_client = Atlas(timeout=60.0)

   # Long-running evaluations
   patient_client = Atlas(timeout=1800.0)  # 30 minutes
   ```

## Error Handling Best Practices

### Comprehensive Error Handling

```python
import atlas
from atlas import Atlas
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_create_evaluation(model: str, benchmark: str):
    """Create evaluation with comprehensive error handling"""
    client = Atlas()

    try:
        evaluation = client.evaluations.create(model=model, benchmark=benchmark)

        if evaluation:
            logger.info(f"✅ Evaluation created: {evaluation.id}")
            return evaluation
        else:
            logger.warning("⚠️ Evaluation creation returned None")
            return None

    except atlas.BadRequestError as e:
        logger.error(f"❌ Bad request - check parameters: {e}")
        logger.error(f"   Model: '{model}', Benchmark: '{benchmark}'")
        return None

    except atlas.AuthenticationError as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("   Check API key configuration")
        return None

    except atlas.PermissionDeniedError as e:
        logger.error(f"❌ Permission denied: {e}")
        logger.error(f"   No access to model '{model}' or benchmark '{benchmark}'")
        return None

    except atlas.NotFoundError as e:
        logger.error(f"❌ Resource not found: {e}")
        logger.error(f"   Model '{model}' or benchmark '{benchmark}' doesn't exist")
        return None

    except atlas.RateLimitError as e:
        retry_after = e.response.headers.get('retry-after', 60)
        logger.warning(f"⏳ Rate limited - retry after {retry_after}s")
        return None  # Could implement retry logic here

    except atlas.InternalServerError as e:
        logger.error(f"❌ Server error: {e}")
        logger.error(f"   Request ID: {e.request_id} (include in support requests)")
        return None

    except atlas.APITimeoutError as e:
        logger.error(f"⏰ Request timed out: {e}")
        logger.error("   Consider increasing timeout or checking network")
        return None

    except atlas.APIConnectionError as e:
        logger.error(f"🔌 Connection error: {e}")
        logger.error("   Check network connectivity and proxy settings")
        return None

    except atlas.APIError as e:
        logger.error(f"❌ Unexpected API error: {e}")
        logger.error(f"   Type: {type(e).__name__}")
        return None

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(f"   Type: {type(e).__name__}")
        return None

# Usage
evaluation = robust_create_evaluation("gpt-4", "mmlu")
```

### Error Recovery Patterns

```python
import atlas
from atlas import Atlas
import time
import random

class AtlasErrorRecovery:
    """Implement various error recovery patterns"""

    def __init__(self, client: Atlas):
        self.client = client

    def exponential_backoff_retry(self, operation, max_retries=3, base_delay=1):
        """Retry with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return operation()
            except (atlas.InternalServerError, atlas.APIConnectionError, atlas.APITimeoutError) as e:
                if attempt == max_retries - 1:
                    raise  # Last attempt - re-raise the error

                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

    def circuit_breaker(self, operation, failure_threshold=5, recovery_time=60):
        """Implement circuit breaker pattern"""
        # This would be a more complex implementation
        # See advanced-usage.md for full implementation
        pass

    def fallback_strategy(self, primary_operation, fallback_operation):
        """Try primary operation, fall back to alternative"""
        try:
            return primary_operation()
        except atlas.APIError as e:
            print(f"Primary operation failed: {e}")
            print("Trying fallback...")
            return fallback_operation()

# Usage
client = Atlas()
recovery = AtlasErrorRecovery(client)

def create_evaluation():
    return client.evaluations.create(model="gpt-4", benchmark="mmlu")

# Retry with exponential backoff
evaluation = recovery.exponential_backoff_retry(create_evaluation)
```
