import httpx

# default timeout is 10 minutes
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)
