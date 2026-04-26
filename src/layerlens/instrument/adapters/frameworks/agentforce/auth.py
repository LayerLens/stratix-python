"""
Salesforce OAuth 2.0 JWT Bearer Authentication

Implements the JWT Bearer flow for server-to-server authentication
with Salesforce Data Cloud. Includes retry with exponential backoff,
timeouts, and credential masking.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Timeout defaults (seconds)
_AUTH_TIMEOUT = 30
_QUERY_TIMEOUT = 60

# Retry defaults
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_MAX_DELAY = 30.0  # seconds

# Salesforce access token lifetime (conservative; actual is ~2 hours)
_TOKEN_LIFETIME_S = 3600

# Rate limit warning threshold (percentage of API limit consumed)
_RATE_LIMIT_WARN_THRESHOLD = 0.8


class SalesforceAuthError(Exception):
    """Raised when Salesforce authentication fails."""

    def __init__(self, message: str, status_code: int | None = None, endpoint: str = "") -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(message)


class SalesforceQueryError(Exception):
    """Raised when a SOQL query fails."""

    def __init__(self, message: str, status_code: int | None = None, soql: str = "") -> None:
        self.status_code = status_code
        self.soql = soql
        super().__init__(message)


class NormalizationError(Exception):
    """Raised when normalization of AgentForce records fails."""

    pass


@dataclass
class SalesforceCredentials:
    """Salesforce connection credentials."""

    client_id: str
    username: str
    private_key: str  # PEM-encoded private key or env var name
    instance_url: str = "https://login.salesforce.com"
    access_token: str | None = None
    token_expiry: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.token_expiry

    def resolve_private_key(self) -> str:
        """Resolve the private key from env var, file path, or raw PEM string."""
        key = self.private_key
        # Check env var reference
        if key.startswith("$") or key.startswith("env:"):
            env_name = key.lstrip("$").removeprefix("env:")
            resolved = os.environ.get(env_name, "")
            if not resolved:
                raise SalesforceAuthError(
                    f"Environment variable '{env_name}' not set for private key"
                )
            return resolved
        # Check file path
        if os.path.isfile(key):
            with open(key) as f:
                return f.read()
        # Assume raw PEM
        return key

    def __repr__(self) -> str:
        return (
            f"SalesforceCredentials("
            f"client_id='{self.client_id[:8]}...', "
            f"username='{self.username}', "
            f"instance_url='{self.instance_url}', "
            f"private_key='***REDACTED***', "
            f"access_token={'***REDACTED***' if self.access_token else 'None'}, "
            f"is_expired={self.is_expired})"
        )


@dataclass
class SalesforceConnection:
    """Active Salesforce connection with retry and timeout support."""

    credentials: SalesforceCredentials
    instance_url: str = ""
    api_version: str = "v60.0"
    auth_timeout: int = _AUTH_TIMEOUT
    query_timeout: int = _QUERY_TIMEOUT
    max_retries: int = _MAX_RETRIES

    def authenticate(self) -> None:
        """Authenticate using JWT Bearer flow with retry."""
        import jwt
        import requests  # type: ignore[import-untyped,unused-ignore]

        resolved_key = self.credentials.resolve_private_key()

        # Build JWT
        now = int(time.time())
        payload = {
            "iss": self.credentials.client_id,
            "sub": self.credentials.username,
            "aud": self.credentials.instance_url,
            "exp": now + 300,
        }
        token = jwt.encode(payload, resolved_key, algorithm="RS256")

        endpoint = f"{self.credentials.instance_url}/services/oauth2/token"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": token,
                    },
                    timeout=self.auth_timeout,
                )
                response.raise_for_status()
                data = response.json()

                self.credentials.access_token = data["access_token"]
                self.instance_url = data["instance_url"]
                self.credentials.token_expiry = now + _TOKEN_LIFETIME_S
                logger.info("Authenticated with Salesforce: %s", self.instance_url)
                return
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    "Salesforce auth timeout (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                # Don't retry 4xx (client errors) except 429 (rate limit)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise SalesforceAuthError(
                        f"Salesforce authentication failed (HTTP {status}). "
                        f"Check credentials and re-authenticate using `stratix agentforce connect`."
                        f" "
                        f"Endpoint: {endpoint}",
                        status_code=status,
                        endpoint=endpoint,
                    ) from e
                last_error = e
                logger.warning(
                    "Salesforce auth HTTP error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(
                    "Salesforce auth request error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = min(
                    _RETRY_BASE_DELAY * (2**attempt),
                    _RETRY_MAX_DELAY,
                )
                time.sleep(delay)

        raise SalesforceAuthError(
            f"Salesforce authentication failed after {self.max_retries} attempts. "
            f"Last error: {last_error}. "
            f"Re-authenticate using `stratix agentforce connect`. "
            f"Endpoint: {endpoint}",
            endpoint=endpoint,
        )

    @staticmethod
    def _check_rate_limit(response_headers: dict[str, Any]) -> None:
        """Parse Sforce-Limit-Info header and warn if approaching limits.

        Salesforce returns ``Sforce-Limit-Info: api-usage=25/15000`` on every
        API response.  We log a warning when usage exceeds the configured
        threshold so operators can react before hitting hard limits.
        """
        limit_info = response_headers.get("Sforce-Limit-Info", "")
        if not limit_info:
            return
        try:
            # Format: "api-usage=USED/LIMIT"
            usage_part = limit_info.split("=", 1)[1] if "=" in limit_info else ""
            if "/" in usage_part:
                used_str, total_str = usage_part.split("/", 1)
                used, total = int(used_str), int(total_str)
                if total > 0 and used / total >= _RATE_LIMIT_WARN_THRESHOLD:
                    logger.warning(
                        "Salesforce API rate limit warning: %d/%d (%.0f%%) consumed",
                        used,
                        total,
                        (used / total) * 100,
                    )
        except (ValueError, IndexError):
            # Malformed header — ignore silently
            pass

    def query(self, soql: str) -> list[dict[str, Any]]:
        """Execute a SOQL query with retry, timeout, and pagination."""
        if self.credentials.is_expired:
            self.authenticate()

        import requests

        url = f"{self.instance_url}/services/data/{self.api_version}/query"
        headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Content-Type": "application/json",
        }

        records: list[dict[str, Any]] = []
        params: dict[str, str] | None = {"q": soql}

        while True:
            last_error: Exception | None = None
            success = False

            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=self.query_timeout,
                    )
                    response.raise_for_status()

                    # Check Salesforce API rate limits
                    self._check_rate_limit(response.headers)  # type: ignore[arg-type]

                    data = response.json()

                    records.extend(data.get("records", []))

                    # Handle pagination
                    next_url = data.get("nextRecordsUrl")
                    if next_url:
                        url = f"{self.instance_url}{next_url}"
                        params = None  # Pagination URL includes query params
                    success = True
                    break

                except requests.exceptions.Timeout as e:
                    last_error = e
                    logger.warning(
                        "Salesforce query timeout (attempt %d/%d)",
                        attempt + 1,
                        self.max_retries,
                    )
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status is not None and 400 <= status < 500 and status != 429:
                        raise SalesforceQueryError(
                            f"SOQL query failed (HTTP {status})",
                            status_code=status,
                            soql=soql[:200],
                        ) from e
                    last_error = e
                    logger.warning(
                        "Salesforce query HTTP error (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                except requests.exceptions.RequestException as e:
                    last_error = e
                    logger.warning(
                        "Salesforce query request error (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        e,
                    )

                if attempt < self.max_retries - 1:
                    delay = min(
                        _RETRY_BASE_DELAY * (2**attempt),
                        _RETRY_MAX_DELAY,
                    )
                    time.sleep(delay)

            if not success:
                raise SalesforceQueryError(
                    f"SOQL query failed after {self.max_retries} attempts. "
                    f"Last error: {last_error}",
                    soql=soql[:200],
                )

            # If no next page, we're done
            if not data.get("nextRecordsUrl"):
                break

        return records
