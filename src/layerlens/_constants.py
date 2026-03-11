import httpx

# default timeout is 10 minutes
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)

DEFAULT_BASE_URL = "https://api.layerlens.ai/api/v1"
