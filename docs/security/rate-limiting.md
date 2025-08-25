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
