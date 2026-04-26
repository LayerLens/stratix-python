"""
Langfuse API Client

HTTP client for the Langfuse REST API using stdlib urllib.
Supports Basic auth, pagination, and exponential backoff.
"""

from __future__ import annotations

import json
import time
import base64
import logging
import contextlib
from typing import Any
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Langfuse API rate limit: 429 responses trigger backoff
_DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 16.0
_REQUEST_TIMEOUT_S = 30


class LangfuseAPIError(Exception):
    """Raised when a Langfuse API call fails."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LangfuseAPIClient:
    """
    HTTP client for the Langfuse REST API.

    Uses Basic auth with base64(public_key:secret_key).
    No external dependencies — built on stdlib urllib.request.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: int = _REQUEST_TIMEOUT_S,
    ) -> None:
        self._host = host.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout

        # Basic auth header
        credentials = f"{public_key}:{secret_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self._auth_header = f"Basic {encoded}"

    # --- Public API ---

    def health_check(self) -> dict[str, Any]:
        """Check Langfuse API health."""
        return self._request("GET", "/api/public/health")

    def list_traces(
        self,
        page: int = 1,
        limit: int = 50,
        order_by: str = "timestamp",
        order: str = "DESC",
        name: str | None = None,
        tags: list[str] | None = None,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        List traces with pagination and filtering.

        Returns dict with 'data' (list of trace objects) and 'meta' (pagination info).
        """
        params: dict[str, Any] = {
            "page": page,
            "limit": limit,
            "orderBy": order_by,
            "order": order,
        }
        if name:
            params["name"] = name
        if tags:
            for tag in tags:
                params.setdefault("tags", []).append(tag)
        if from_timestamp:
            params["fromTimestamp"] = from_timestamp.isoformat()
        if to_timestamp:
            params["toTimestamp"] = to_timestamp.isoformat()

        return self._request("GET", "/api/public/traces", params=params)

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """Get a single trace with all observations."""
        return self._request("GET", f"/api/public/traces/{trace_id}")

    def list_observations(
        self,
        trace_id: str | None = None,
        page: int = 1,
        limit: int = 50,
        type: str | None = None,
    ) -> dict[str, Any]:
        """List observations for a trace."""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if trace_id:
            params["traceId"] = trace_id
        if type:
            params["type"] = type
        return self._request("GET", "/api/public/observations", params=params)

    def create_trace(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new trace in Langfuse."""
        return self._request(
            "POST",
            "/api/public/ingestion",
            body={
                "batch": [
                    {
                        "id": trace_data.get("id", ""),
                        "type": "trace-create",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "body": trace_data,
                    }
                ],
            },
        )

    def create_generation(self, generation_data: dict[str, Any]) -> dict[str, Any]:
        """Create a generation observation."""
        return self._request(
            "POST",
            "/api/public/ingestion",
            body={
                "batch": [
                    {
                        "id": generation_data.get("id", ""),
                        "type": "generation-create",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "body": generation_data,
                    }
                ],
            },
        )

    def create_span(self, span_data: dict[str, Any]) -> dict[str, Any]:
        """Create a span observation."""
        return self._request(
            "POST",
            "/api/public/ingestion",
            body={
                "batch": [
                    {
                        "id": span_data.get("id", ""),
                        "type": "span-create",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "body": span_data,
                    }
                ],
            },
        )

    def ingestion_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Send a batch of ingestion events."""
        return self._request("POST", "/api/public/ingestion", body={"batch": events})

    def get_all_traces(
        self,
        limit: int = 50,
        tags: list[str] | None = None,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all traces with automatic pagination.

        Yields all pages until exhausted.
        """
        all_traces: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self.list_traces(
                page=page,
                limit=limit,
                tags=tags,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
            data = result.get("data", [])
            if not data:
                break
            all_traces.extend(data)
            meta = result.get("meta", {})
            total_pages = meta.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1
        return all_traces

    # --- Internal ---

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry and backoff."""
        url = f"{self._host}{path}"
        if params:
            # Handle list params (e.g., tags)
            query_parts = []
            for k, v in params.items():
                if isinstance(v, list):
                    for item in v:
                        query_parts.append(f"{k}={item}")
                else:
                    query_parts.append(f"{k}={v}")
            url = f"{url}?{'&'.join(query_parts)}"

        headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = json.dumps(body).encode() if body else None

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                req = Request(url, data=data, headers=headers, method=method)
                with urlopen(req, timeout=self._timeout) as resp:
                    resp_body = resp.read().decode()
                    if not resp_body:
                        return {}
                    return json.loads(resp_body)  # type: ignore[no-any-return]

            except HTTPError as e:
                status = e.code
                error_body = ""
                with contextlib.suppress(Exception):
                    error_body = e.read().decode()

                if status == 429 or status >= 500:
                    last_error = LangfuseAPIError(
                        f"HTTP {status}: {error_body}", status_code=status, body=error_body
                    )
                    if attempt < self._max_retries:
                        delay = min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_MAX_S)
                        logger.debug(
                            "Langfuse API %s %s returned %d, retrying in %.1fs (attempt %d/%d)",
                            method,
                            path,
                            status,
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                        time.sleep(delay)
                        continue
                raise LangfuseAPIError(  # noqa: B904
                    f"HTTP {status}: {error_body}", status_code=status, body=error_body
                )

            except URLError as e:
                last_error = LangfuseAPIError(f"Connection error: {e}")
                if attempt < self._max_retries:
                    delay = min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_MAX_S)
                    logger.debug(
                        "Langfuse API connection error, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise last_error  # noqa: B904

        raise last_error or LangfuseAPIError("Max retries exceeded")
