"""Langfuse API Client.

HTTP client for the Langfuse REST API using stdlib urllib.

Hardened in PR ``feat/instrument-importer-hardening`` to use the shared
importer base helpers for:

* Rate-limit-aware retry with exponential backoff and full jitter
  (:func:`layerlens.instrument.adapters._base.importer.retry_with_backoff`).
* Rate-limit header parsing
  (:func:`layerlens.instrument.adapters._base.importer.parse_rate_limit_headers`)
  — Langfuse Cloud emits ``X-RateLimit-Remaining`` /
  ``X-RateLimit-Reset`` / ``Retry-After`` and we honour them rather than
  retrying blindly into a 429 storm.
* Cursor-based pagination
  (:func:`layerlens.instrument.adapters._base.importer.paginate`).
"""

from __future__ import annotations

import json
import base64
import logging
import contextlib
from typing import Any
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from layerlens.instrument.adapters._base.importer import (
    RateLimitInfo,
    RetryableHTTPError,
    paginate,
    retry_with_backoff,
    parse_rate_limit_headers,
)

logger = logging.getLogger(__name__)

# Langfuse API rate limit: 429 responses trigger backoff
_DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 16.0
_REQUEST_TIMEOUT_S = 30


class LangfuseAPIError(Exception):
    """Raised when a Langfuse API call fails (terminal — not retryable)."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LangfuseAPIClient:
    """HTTP client for the Langfuse REST API.

    Uses Basic auth with base64(public_key:secret_key).
    No external dependencies — built on stdlib ``urllib.request`` and
    the shared importer-base retry/rate-limit helpers.
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

        # Most recent rate-limit observation (set after every successful
        # request and after every retryable HTTPError). Exposed via
        # :attr:`last_rate_limit` so the importer can surface warnings
        # at the orchestration layer.
        self._last_rate_limit: RateLimitInfo | None = None

    # --- Public API ---

    @property
    def last_rate_limit(self) -> RateLimitInfo | None:
        """Most recent rate-limit info observed on a response.

        Returns ``None`` when no request has been made or when the
        upstream did not include rate-limit headers.
        """
        return self._last_rate_limit

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
        """List traces with pagination and filtering.

        Returns dict with 'data' (list of trace objects) and 'meta'
        (pagination info, including ``totalPages``).
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
        """Fetch all traces with automatic pagination.

        Iterates the Langfuse paged ``/api/public/traces`` endpoint via
        the shared :func:`paginate` helper. The Langfuse API uses
        page-number pagination (``page`` query parameter) rather than
        opaque cursors; we adapt by treating the next page number as
        the "cursor" — :func:`paginate` only requires an opaque token.
        """

        def _fetch(cursor: str | None) -> dict[str, Any]:
            page_num = int(cursor) if cursor is not None else 1
            response = self.list_traces(
                page=page_num,
                limit=limit,
                tags=tags,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
            meta = response.get("meta") or {}
            total_pages = meta.get("totalPages", 1)
            current_page = meta.get("page", page_num)
            # Surface a "next cursor" only when more pages remain.
            if current_page < total_pages:
                response["__next_page"] = str(current_page + 1)
            return response

        return list(paginate(_fetch, cursor_field="__next_page", data_field="data"))

    # --- Internal ---

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        """Build the full URL with optional query string."""
        url = f"{self._host}{path}"
        if not params:
            return url
        # Handle list params (e.g., tags) — repeat key=value per item.
        query_parts: list[str] = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    query_parts.append(f"{k}={item}")
            else:
                query_parts.append(f"{k}={v}")
        return f"{url}?{'&'.join(query_parts)}"

    def _read_response_headers(self, resp: Any) -> dict[str, str]:
        """Extract a flat header dict from an ``http.client`` response.

        ``urlopen`` returns an ``http.client.HTTPResponse`` whose
        ``.headers`` is an ``http.client.HTTPMessage`` (a list-like).
        We surface :func:`parse_rate_limit_headers`-friendly access.
        """
        headers = getattr(resp, "headers", None)
        if headers is None:
            return {}
        out: dict[str, str] = {}
        for key in headers.keys():
            out[key] = headers.get(key, "")
        return out

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with rate-limit-aware retry + backoff.

        Translates transient HTTP errors (429, 5xx) into
        :class:`RetryableHTTPError` so the shared
        :func:`retry_with_backoff` driver can apply exponential backoff
        with full jitter, and honours ``Retry-After`` /
        ``X-RateLimit-Reset`` to sleep until the explicit deadline
        rather than retrying blindly.

        Terminal errors (4xx other than 429) are wrapped in
        :class:`LangfuseAPIError` and raised without retry.
        """
        url = self._build_url(path, params)
        headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = json.dumps(body).encode() if body else None

        def _attempt() -> dict[str, Any]:
            try:
                req = Request(url, data=data, headers=headers, method=method)
                with urlopen(req, timeout=self._timeout) as resp:
                    response_headers = self._read_response_headers(resp)
                    self._last_rate_limit = parse_rate_limit_headers(response_headers)
                    self._warn_if_throttle_imminent()

                    resp_body = resp.read().decode()
                    if not resp_body:
                        return {}
                    parsed = json.loads(resp_body)
                    if not isinstance(parsed, dict):
                        return {"data": parsed}
                    return parsed
            except HTTPError as e:
                status = e.code
                error_body = ""
                with contextlib.suppress(Exception):
                    error_body = e.read().decode()
                # HTTPError exposes headers on .headers — parse them so
                # rate-limit info from a 429 drives the next sleep.
                try:
                    self._last_rate_limit = parse_rate_limit_headers(e.headers)
                except Exception:  # pragma: no cover — defensive only
                    self._last_rate_limit = None

                if status == 429 or status >= 500:
                    raise RetryableHTTPError(
                        f"Langfuse {method} {path} returned HTTP {status}: {error_body}",
                        status_code=status,
                        rate_limit=self._last_rate_limit,
                    ) from e
                # 4xx other than 429 → terminal (caller bug, not transient).
                raise LangfuseAPIError(
                    f"HTTP {status}: {error_body}",
                    status_code=status,
                    body=error_body,
                ) from e
            except URLError as e:
                # Connection-level errors are transient.
                raise RetryableHTTPError(
                    f"Langfuse {method} {path} connection error: {e}",
                ) from e

        try:
            return retry_with_backoff(
                _attempt,
                max_retries=self._max_retries,
                base_delay=_BACKOFF_BASE_S,
                max_delay=_BACKOFF_MAX_S,
                jitter=True,
            )
        except RetryableHTTPError as exc:
            # Exhausted retries — convert to terminal LangfuseAPIError so
            # callers continue to see the same exception type they did
            # before this refactor.
            raise LangfuseAPIError(
                str(exc),
                status_code=exc.status_code,
            ) from exc

    def _warn_if_throttle_imminent(self) -> None:
        """Log a warning when ``usage_ratio`` exceeds 80 %.

        Mirrors Agentforce's ``_check_rate_limit`` (auth.py:204-230) so
        operators are alerted BEFORE the upstream actually throttles.
        """
        info = self._last_rate_limit
        if info is None:
            return
        if info.usage_ratio is not None and info.usage_ratio >= 0.8:
            logger.warning(
                "Langfuse API rate-limit warning: %.0f%% of quota consumed (remaining=%s, limit=%s)",
                info.usage_ratio * 100.0,
                info.remaining,
                info.limit,
            )
