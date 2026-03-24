"""Credential storage, token refresh, and CLI login via backend."""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional
from pathlib import Path

import httpx

from .._constants import DEFAULT_BASE_URL, AUTH_CLI_LOGIN_PATH, AUTH_CLI_CONFIG_PATH

CREDENTIALS_DIR = Path.home() / ".layerlens"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials"

# How many seconds before expiry to trigger a proactive refresh
TOKEN_REFRESH_MARGIN = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Credential storage (JSON, atomic writes)
# ---------------------------------------------------------------------------


def _ensure_dir() -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_DIR.chmod(0o700)


def load_credentials() -> Optional[Dict[str, Any]]:
    """Load stored credentials from ~/.layerlens/credentials."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def save_credentials(creds: Dict[str, Any]) -> None:
    """Persist credentials to ~/.layerlens/credentials with restrictive perms."""
    _ensure_dir()
    tmp = CREDENTIALS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(creds, indent=2))
    tmp.chmod(0o600)
    tmp.rename(CREDENTIALS_FILE)


def clear_credentials() -> None:
    """Remove stored credentials."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


# ---------------------------------------------------------------------------
# Auth config discovery
# ---------------------------------------------------------------------------

_cached_auth_config: Optional[Dict[str, str]] = None


def _get_base_url() -> str:
    return (
        os.environ.get("LAYERLENS_STRATIX_BASE_URL") or os.environ.get("LAYERLENS_ATLAS_BASE_URL") or DEFAULT_BASE_URL
    )


def fetch_auth_config(base_url: Optional[str] = None) -> Dict[str, str]:
    """Fetch Cognito OAuth2 config from the backend discovery endpoint.

    Returns dict with keys: region, client_id, domain, scopes.
    Caches the result in memory for the process lifetime.
    """
    global _cached_auth_config

    if _cached_auth_config is not None:
        return _cached_auth_config

    # Also check if stored credentials already have the config cached
    creds = load_credentials()
    if creds and "auth_config" in creds:
        _cached_auth_config = creds["auth_config"]
        return _cached_auth_config

    url = (base_url or _get_base_url()).rstrip("/") + AUTH_CLI_CONFIG_PATH
    resp = httpx.get(url, timeout=15)
    resp.raise_for_status()
    config = resp.json()

    _cached_auth_config = config
    return config


def _cognito_base_url(config: Dict[str, str]) -> str:
    return f"https://{config['domain']}.auth.{config['region']}.amazoncognito.com"


def _token_url(config: Dict[str, str]) -> str:
    return f"{_cognito_base_url(config)}/oauth2/token"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def is_token_expired(creds: Dict[str, Any]) -> bool:
    """Check whether the access token is expired or about to expire."""
    expires_at = creds.get("expires_at", 0)
    return time.time() >= (expires_at - TOKEN_REFRESH_MARGIN)


def refresh_access_token(creds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use the refresh_token to obtain a new access token from Cognito.

    Returns updated credentials dict or ``None`` on failure.
    """
    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        return None

    auth_config = creds.get("auth_config")
    if not auth_config:
        try:
            auth_config = fetch_auth_config()
        except httpx.HTTPError:
            return None

    try:
        resp = httpx.post(
            _token_url(auth_config),
            data={
                "grant_type": "refresh_token",
                "client_id": auth_config["client_id"],
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        return None

    body = resp.json()
    creds["access_token"] = body["access_token"]
    creds["id_token"] = body.get("id_token", creds.get("id_token"))
    creds["expires_at"] = time.time() + body.get("expires_in", 3600)
    # Cognito does not rotate refresh tokens by default
    if "refresh_token" in body:
        creds["refresh_token"] = body["refresh_token"]
    save_credentials(creds)
    return creds


def get_valid_token() -> Optional[str]:
    """Return a valid ID token for Bearer auth, refreshing transparently if needed.

    The backend JWT middleware expects claims (email, cognito:groups) that are
    present in the Cognito **ID token**, not the access token.

    Falls back to ``LAYERLENS_API_KEY`` env-var for CI/CD.
    """
    # 1. Env-var fallback (CI/CD)
    env_key = os.environ.get("LAYERLENS_API_KEY")
    if env_key:
        return env_key

    creds = load_credentials()
    if creds is None:
        return None

    if is_token_expired(creds):
        creds = refresh_access_token(creds)
        if creds is None:
            return None

    # Prefer id_token (has email + cognito:groups claims needed by backend JWT middleware)
    return creds.get("id_token") or creds.get("access_token")


# ---------------------------------------------------------------------------
# Email/password login via backend /auth/cli-login
# ---------------------------------------------------------------------------


class LoginError(Exception):
    """Raised when login fails."""


def cli_login(email: str, password: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Authenticate with email/password via the backend CLI login endpoint.

    Returns the stored credentials dict.
    """
    url = (base_url or _get_base_url()).rstrip("/") + AUTH_CLI_LOGIN_PATH

    resp = httpx.post(
        url,
        json={"email": email, "password": password},
        timeout=30,
    )

    if resp.status_code == 401:
        raise LoginError("Invalid email or password.")
    if resp.status_code == 400:
        raise LoginError("Email and password are required.")
    resp.raise_for_status()

    body = resp.json()

    # Also fetch auth config for future token refresh
    try:
        auth_config = fetch_auth_config(base_url)
    except httpx.HTTPError:
        auth_config = None

    creds: Dict[str, Any] = {
        "access_token": body["access_token"],
        "id_token": body.get("id_token"),
        "refresh_token": body.get("refresh_token"),
        "expires_at": time.time() + body.get("expires_in", 3600),
        "token_type": body.get("token_type", "Bearer"),
        "user": body.get("user"),
        "base_url": base_url or _get_base_url(),
    }
    if auth_config:
        creds["auth_config"] = auth_config

    save_credentials(creds)
    return creds


# ---------------------------------------------------------------------------
# User-info helper
# ---------------------------------------------------------------------------


def get_user_info(access_token: str) -> Optional[Dict[str, Any]]:
    """Return user info from stored credentials or Cognito userInfo endpoint."""
    creds = load_credentials()

    # First try stored user info from login response
    if creds and creds.get("user"):
        return creds["user"]

    # Fall back to Cognito userInfo endpoint
    auth_config = (creds or {}).get("auth_config")
    if not auth_config:
        try:
            auth_config = fetch_auth_config()
        except httpx.HTTPError:
            return None

    userinfo_url = f"{_cognito_base_url(auth_config)}/oauth2/userInfo"
    try:
        resp = httpx.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None
