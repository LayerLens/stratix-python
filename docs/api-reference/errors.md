# Error Handling

The Atlas Python SDK provides a comprehensive exception hierarchy to help you handle different error conditions gracefully. This guide covers all available exception types and best practices for error handling.

## Exception Hierarchy

All Atlas exceptions inherit from the base `AtlasError` class:

```
AtlasError
├── APIError
│   ├── APIConnectionError
│   │   └── APITimeoutError
│   ├── APIResponseValidationError
│   └── APIStatusError
│       ├── BadRequestError (400)
│       ├── AuthenticationError (401)
│       ├── PermissionDeniedError (403)
│       ├── NotFoundError (404)
│       ├── ConflictError (409)
│       ├── UnprocessableEntityError (422)
│       ├── RateLimitError (429)
│       └── InternalServerError (500+)
```

## Exception Types

### Base Exceptions

#### `AtlasError`

Base exception for all Atlas-related errors.

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.AtlasError as e:
    print(f"Atlas error occurred: {e}")
```

#### `APIError`

Base exception for all API-related errors. Contains additional context about the request.

**Properties:**

- `message`: Error message
- `request`: The HTTP request that caused the error
- `body`: Response body (if available)

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIError as e:
    print(f"API error: {e.message}")
    print(f"Request URL: {e.request.url}")
    print(f"Response body: {e.body}")
```

### Connection Errors

#### `APIConnectionError`

Raised when the client cannot connect to the API server.

**Common causes:**

- Network connectivity issues
- DNS resolution problems
- Server is down
- Firewall blocking requests

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIConnectionError as e:
    print("Connection failed - check your network connection")
    print(f"Error details: {e}")
```

#### `APITimeoutError`

Raised when a request times out.

```python
import atlas

try:
    client = atlas.Atlas(timeout=0.2)  # Very short timeout
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APITimeoutError:
    print("Request timed out - try increasing timeout or check network")
```

### HTTP Status Errors

All HTTP status errors inherit from `APIStatusError` and include additional properties:

**Properties:**

- `status_code`: HTTP status code
- `response`: Full HTTP response object
- `request_id`: Request ID for tracking (if provided by server)

#### `BadRequestError` (400)

Request was malformed or contained invalid parameters.

```python
import atlas

try:
    client = atlas.Atlas()
    # Invalid parameters
    evaluation = client.evaluations.create(model="", benchmark="")
except atlas.BadRequestError as e:
    print(f"Bad request: {e}")
    print(f"Status code: {e.status_code}")
```

#### `AuthenticationError` (401)

API key is missing, invalid, or expired.

```python
import atlas

try:
    client = atlas.Atlas(api_key="invalid_key")
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.AuthenticationError:
    print("Authentication failed - check your API key")
    print("Make sure LAYERLENS_ATLAS_API_KEY is set correctly")
```

#### `PermissionDeniedError` (403)

Valid API key but insufficient permissions for the requested operation.

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="restricted-model", benchmark="mmlu")
except atlas.PermissionDeniedError:
    print("Permission denied - check your organization/project access")
    print("Contact your administrator for access to this resource")
```

#### `NotFoundError` (404)

Requested resource (model, benchmark, evaluation) does not exist.

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="nonexistent-model", benchmark="mmlu")
except atlas.NotFoundError:
    print("Model or benchmark not found")
    print("Check available models and benchmarks in the Atlas dashboard")
```

#### `ConflictError` (409)

Request conflicts with current resource state.

```python
import atlas

try:
    client = atlas.Atlas()
    # Some operation that conflicts with current state
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.ConflictError:
    print("Request conflicts with current state")
```

#### `UnprocessableEntityError` (422)

Request parameters are valid but cannot be processed.

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="invalid-benchmark")
except atlas.UnprocessableEntityError as e:
    print(f"Cannot process request: {e}")
    print("Parameters are valid but operation cannot be completed")
```

#### `RateLimitError` (429)

Too many requests sent in a given time period.

```python
import atlas
import time

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.RateLimitError as e:
    print("Rate limit exceeded")
    # Extract retry-after header if available
    retry_after = e.response.headers.get('retry-after')
    if retry_after:
        print(f"Retry after {retry_after} seconds")
        time.sleep(int(retry_after))
    else:
        print("Waiting 60 seconds before retry...")
        time.sleep(60)
```

#### `InternalServerError` (500+)

Server-side error occurred.

```python
import atlas

try:
    client = atlas.Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.InternalServerError as e:
    print(f"Server error: {e.status_code}")
    print("This is a server-side issue - try again later")
    print(f"Request ID: {e.request_id}")  # For support tickets
```

## Best Practices

### 1. Handle Specific Exceptions

```python
import atlas
import time
from atlas import Atlas

def robust_create_evaluation(model: str, benchmark: str, max_retries: int = 3):
    client = Atlas()

    for attempt in range(max_retries):
        try:
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            return evaluation

        except atlas.AuthenticationError:
            print("❌ Authentication failed - check your API key")
            break  # Don't retry auth errors

        except atlas.PermissionDeniedError:
            print("❌ Permission denied - contact your administrator")
            break  # Don't retry permission errors

        except atlas.NotFoundError:
            print(f"❌ Model '{model}' or benchmark '{benchmark}' not found")
            break  # Don't retry not found errors

        except atlas.RateLimitError as e:
            retry_after = e.response.headers.get('retry-after', 60)
            print(f"⏳ Rate limited - waiting {retry_after} seconds...")
            time.sleep(int(retry_after))
            continue  # Retry after waiting

        except atlas.InternalServerError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"🔄 Server error - retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
            else:
                print("❌ Server error - max retries exceeded")
                break

        except atlas.APIConnectionError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"🔄 Connection error - retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
            else:
                print("❌ Connection failed - check your network")
                break

        except atlas.APIError as e:
            print(f"❌ Unexpected API error: {e}")
            break

    return None
```

### 2. Graceful Degradation

```python
import atlas
from atlas import Atlas

def get_evaluation_results_with_fallback(evaluation_id: str):
    client = Atlas()

    try:
        results = client.results.get(evaluation_id=evaluation_id)

        if results:
            return {"success": True, "data": results, "message": "Results retrieved successfully"}
        else:
            return {"success": False, "data": None, "message": "No results found"}

    except atlas.NotFoundError:
        return {"success": False, "data": None, "message": "Evaluation not found"}

    except atlas.AuthenticationError:
        return {"success": False, "data": None, "message": "Authentication required"}

    except atlas.APIConnectionError:
        return {"success": False, "data": None, "message": "Service temporarily unavailable"}

    except atlas.APIError as e:
        return {"success": False, "data": None, "message": f"Service error: {e}"}

# Usage
result = get_evaluation_results_with_fallback("eval_123")
if result["success"]:
    process_results(result["data"])
else:
    print(f"Could not get results: {result['message']}")
```

### 3. Logging and Monitoring

```python
import logging
import atlas
from atlas import Atlas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_api_call():
    client = Atlas()

    try:
        logger.info("Creating evaluation...")
        evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

        if evaluation:
            logger.info(f"Evaluation created successfully: {evaluation.id}")
            return evaluation
        else:
            logger.warning("Evaluation creation returned None")
            return None

    except atlas.RateLimitError as e:
        logger.warning(f"Rate limited - request ID: {e.request_id}")
        raise

    except atlas.AuthenticationError:
        logger.error("Authentication failed - check API key configuration")
        raise

    except atlas.APIConnectionError:
        logger.error("Network connection failed")
        raise

    except atlas.InternalServerError as e:
        logger.error(f"Server error: {e.status_code} - request ID: {e.request_id}")
        raise

    except atlas.APIError as e:
        logger.error(f"Unexpected API error: {e} - request ID: {getattr(e, 'request_id', 'N/A')}")
        raise
```

### 4. Context Managers for Resource Management

```python
import atlas
from contextlib import contextmanager
from atlas import Atlas

@contextmanager
def atlas_client():
    """Context manager for Atlas client with error handling"""
    client = None
    try:
        client = Atlas()
        yield client
    except atlas.AuthenticationError:
        print("Authentication failed")
        raise
    except atlas.APIConnectionError:
        print("Connection failed")
        raise
    finally:
        # Cleanup if needed
        pass

# Usage
try:
    with atlas_client() as client:
        evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
        results = client.results.get(evaluation_id=evaluation.id)
except atlas.AtlasError:
    print("Atlas operation failed")
```

## Error Response Details

### Status Error Properties

```python
import atlas
from atlas import Atlas

try:
    client = Atlas()
    evaluation = client.evaluations.create(model="invalid", benchmark="invalid")
except atlas.APIStatusError as e:
    print(f"Status Code: {e.status_code}")
    print(f"Request ID: {e.request_id}")
    print(f"Response Headers: {dict(e.response.headers)}")
    print(f"Response Body: {e.body}")
    print(f"Request URL: {e.request.url}")
    print(f"Request Method: {e.request.method}")
```

### Extracting Useful Information

```python
import atlas
from atlas import Atlas

def extract_error_info(error: atlas.APIError):
    info = {
        "type": type(error).__name__,
        "message": str(error),
        "request_url": error.request.url if hasattr(error, 'request') else None,
        "request_method": error.request.method if hasattr(error, 'request') else None,
    }

    if hasattr(error, 'status_code'):
        info["status_code"] = error.status_code

    if hasattr(error, 'request_id'):
        info["request_id"] = error.request_id

    if hasattr(error, 'response'):
        info["response_headers"] = dict(error.response.headers)

    return info

# Usage
try:
    client = Atlas()
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIError as e:
    error_info = extract_error_info(e)
    print(f"Error details: {error_info}")
```

## Testing Error Handling

```python
import pytest
import atlas
from unittest.mock import Mock, patch
from atlas import Atlas

def test_authentication_error_handling():
    """Test that authentication errors are handled properly"""
    with patch('atlas.Atlas') as mock_atlas:
        mock_atlas.side_effect = atlas.AuthenticationError(
            "Invalid API key",
            request=Mock(),
            response=Mock()
        )

        with pytest.raises(atlas.AuthenticationError):
            client = Atlas()
            client.evaluations.create(model="gpt-4", benchmark="mmlu")

def test_rate_limit_retry():
    """Test that rate limit errors trigger appropriate retry logic"""
    # Your retry logic test here
    pass
```

## Common Error Scenarios

### Invalid Configuration

```python
# Missing API key
try:
    client = Atlas(api_key=None)
except atlas.AtlasError as e:
    print(f"Configuration error: {e}")
```

### Network Issues

```python
# Connection timeout
try:
    client = Atlas(timeout=0.1)  # Very short timeout
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APITimeoutError:
    print("Request timed out")

# Network connectivity
try:
    # Simulate network issues
    evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")
except atlas.APIConnectionError:
    print("Network connectivity issue")
```

## Error Recovery Strategies

### Exponential Backoff

```python
import time
import random
import atlas
from atlas import Atlas

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except (atlas.InternalServerError, atlas.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s...")
            time.sleep(delay)

# Usage
def create_evaluation():
    client = Atlas()
    return client.evaluations.create(model="gpt-4", benchmark="mmlu")

evaluation = exponential_backoff_retry(create_evaluation)
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum
from atlas import Atlas
import atlas

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.timeout:
                raise atlas.APIConnectionError(message="Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except (atlas.InternalServerError, atlas.APIConnectionError) as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
breaker = CircuitBreaker()
client = Atlas()

try:
    evaluation = breaker.call(
        client.evaluations.create,
        model="gpt-4",
        benchmark="mmlu"
    )
except atlas.APIError as e:
    print(f"Circuit breaker prevented call or operation failed: {e}")
```
