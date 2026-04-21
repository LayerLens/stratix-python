"""Tests for CLI authentication: credential storage, token refresh, login flow."""
# ruff: noqa: ARG002  # creds_dir fixture is used for its monkeypatch side effect

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from layerlens.cli._app import cli
from layerlens.cli._auth import (
    is_token_expired,
    load_credentials,
    save_credentials,
    clear_credentials,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


@pytest.fixture
def creds_dir(tmp_path: Path, monkeypatch):
    """Redirect credential storage to a temp directory."""
    creds_dir = tmp_path / ".layerlens"
    creds_file = creds_dir / "credentials"
    monkeypatch.setattr("layerlens.cli._auth.CREDENTIALS_DIR", creds_dir)
    monkeypatch.setattr("layerlens.cli._auth.CREDENTIALS_FILE", creds_file)
    return creds_dir


@pytest.fixture(autouse=True)
def clear_config_cache(monkeypatch):
    """Reset the in-memory auth config cache between tests."""
    monkeypatch.setattr("layerlens.cli._auth._cached_auth_config", None)


SAMPLE_AUTH_CONFIG = {
    "region": "us-east-1",
    "client_id": "test-client-id",
    "domain": "atlas-test",
    "scopes": "openid profile email",
}


@pytest.fixture
def sample_creds():
    return {
        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test",
        "id_token": "id-tok",
        "refresh_token": "refresh-tok",
        "expires_at": time.time() + 3600,
        "token_type": "Bearer",
        "auth_config": SAMPLE_AUTH_CONFIG,
        "user": {"email": "test@example.com", "given_name": "Test"},
    }


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------


class TestCredentialStorage:
    def test_save_and_load(self, creds_dir, sample_creds):
        save_credentials(sample_creds)
        loaded = load_credentials()
        assert loaded is not None
        assert loaded["access_token"] == sample_creds["access_token"]

    def test_load_returns_none_when_missing(self, creds_dir):
        assert load_credentials() is None

    def test_clear_credentials(self, creds_dir, sample_creds):
        save_credentials(sample_creds)
        assert load_credentials() is not None
        clear_credentials()
        assert load_credentials() is None

    def test_clear_when_nothing_stored(self, creds_dir):
        clear_credentials()

    def test_file_permissions(self, creds_dir, sample_creds):
        save_credentials(sample_creds)
        creds_file = creds_dir / "credentials"
        mode = creds_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_dir_permissions(self, creds_dir, sample_creds):
        save_credentials(sample_creds)
        mode = creds_dir.stat().st_mode & 0o777
        assert mode == 0o700

    def test_load_corrupted_file(self, creds_dir):
        creds_dir.mkdir(parents=True, exist_ok=True)
        (creds_dir / "credentials").write_text("not json{{{")
        assert load_credentials() is None


# ---------------------------------------------------------------------------
# Auth config discovery
# ---------------------------------------------------------------------------


class TestAuthConfigDiscovery:
    def test_fetch_from_api(self, creds_dir):
        from layerlens.cli._auth import fetch_auth_config

        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_AUTH_CONFIG
        mock_resp.raise_for_status = MagicMock()

        with patch("layerlens.cli._auth.httpx.get", return_value=mock_resp) as mock_get:
            result = fetch_auth_config("https://api.example.com/api/v1")

        assert result["client_id"] == "test-client-id"
        assert result["domain"] == "atlas-test"
        mock_get.assert_called_once()

    def test_uses_cached_from_credentials(self, creds_dir, sample_creds):
        from layerlens.cli._auth import fetch_auth_config

        save_credentials(sample_creds)

        with patch("layerlens.cli._auth.httpx.get") as mock_get:
            result = fetch_auth_config()

        mock_get.assert_not_called()
        assert result["client_id"] == "test-client-id"

    def test_caches_in_memory(self, creds_dir):
        from layerlens.cli._auth import fetch_auth_config

        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_AUTH_CONFIG
        mock_resp.raise_for_status = MagicMock()

        with patch("layerlens.cli._auth.httpx.get", return_value=mock_resp) as mock_get:
            fetch_auth_config("https://api.example.com/api/v1")
            fetch_auth_config("https://api.example.com/api/v1")

        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Token expiry
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    def test_not_expired(self):
        creds = {"expires_at": time.time() + 3600}
        assert not is_token_expired(creds)

    def test_expired(self):
        creds = {"expires_at": time.time() - 10}
        assert is_token_expired(creds)

    def test_within_margin(self):
        creds = {"expires_at": time.time() + 240}
        assert is_token_expired(creds)

    def test_missing_expires_at(self):
        assert is_token_expired({})


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


class TestTokenRefresh:
    def test_refresh_success(self, creds_dir, sample_creds):
        from layerlens.cli._auth import refresh_access_token

        save_credentials(sample_creds)
        sample_creds["expires_at"] = time.time() - 10

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "new-access-token",
            "id_token": "new-id-token",
            "expires_in": 3600,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("layerlens.cli._auth.httpx.post", return_value=mock_resp):
            result = refresh_access_token(sample_creds)

        assert result is not None
        assert result["access_token"] == "new-access-token"
        loaded = load_credentials()
        assert loaded["access_token"] == "new-access-token"

    def test_refresh_no_refresh_token(self, creds_dir):
        from layerlens.cli._auth import refresh_access_token

        result = refresh_access_token({"access_token": "tok"})
        assert result is None

    def test_refresh_http_error(self, creds_dir, sample_creds):
        import httpx as _httpx

        from layerlens.cli._auth import refresh_access_token

        with patch("layerlens.cli._auth.httpx.post", side_effect=_httpx.HTTPError("fail")):
            result = refresh_access_token(sample_creds)

        assert result is None


# ---------------------------------------------------------------------------
# get_valid_token
# ---------------------------------------------------------------------------


class TestGetValidToken:
    def test_env_var_takes_precedence(self, creds_dir, monkeypatch):
        from layerlens.cli._auth import get_valid_token

        monkeypatch.setenv("LAYERLENS_API_KEY", "env-key-123")
        assert get_valid_token() == "env-key-123"

    def test_returns_stored_token(self, creds_dir, sample_creds, monkeypatch):
        from layerlens.cli._auth import get_valid_token

        monkeypatch.delenv("LAYERLENS_API_KEY", raising=False)
        save_credentials(sample_creds)
        # get_valid_token prefers id_token (has email + cognito:groups for backend JWT middleware)
        assert get_valid_token() == sample_creds["id_token"]

    def test_returns_none_when_no_creds(self, creds_dir, monkeypatch):
        from layerlens.cli._auth import get_valid_token

        monkeypatch.delenv("LAYERLENS_API_KEY", raising=False)
        assert get_valid_token() is None

    def test_refreshes_expired_token(self, creds_dir, sample_creds, monkeypatch):
        from layerlens.cli._auth import get_valid_token

        monkeypatch.delenv("LAYERLENS_API_KEY", raising=False)
        sample_creds["expires_at"] = time.time() - 10
        save_credentials(sample_creds)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "refreshed-token",
            "id_token": "refreshed-id-token",
            "expires_in": 3600,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("layerlens.cli._auth.httpx.post", return_value=mock_resp):
            token = get_valid_token()

        # get_valid_token prefers id_token
        assert token == "refreshed-id-token"


# ---------------------------------------------------------------------------
# cli_login
# ---------------------------------------------------------------------------


class TestCLILogin:
    def test_login_success(self, creds_dir):
        from layerlens.cli._auth import cli_login

        login_resp = MagicMock()
        login_resp.status_code = 200
        login_resp.json.return_value = {
            "access_token": "access-tok",
            "id_token": "id-tok",
            "refresh_token": "refresh-tok",
            "expires_in": 3600,
            "token_type": "Bearer",
            "user": {"email": "user@example.com", "given_name": "Test"},
        }
        login_resp.raise_for_status = MagicMock()

        config_resp = MagicMock()
        config_resp.json.return_value = SAMPLE_AUTH_CONFIG
        config_resp.raise_for_status = MagicMock()

        with patch("layerlens.cli._auth.httpx.post", return_value=login_resp):
            with patch("layerlens.cli._auth.httpx.get", return_value=config_resp):
                result = cli_login("user@example.com", "pass123", base_url="https://api.test.com/api/v1")

        assert result["access_token"] == "access-tok"
        assert result["user"]["email"] == "user@example.com"

        loaded = load_credentials()
        assert loaded["access_token"] == "access-tok"

    def test_login_invalid_credentials(self, creds_dir):
        from layerlens.cli._auth import LoginError, cli_login

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("layerlens.cli._auth.httpx.post", return_value=mock_resp):
            with pytest.raises(LoginError, match="Invalid email or password"):
                cli_login("bad@example.com", "wrong")

    def test_login_missing_fields(self, creds_dir):
        from layerlens.cli._auth import LoginError, cli_login

        mock_resp = MagicMock()
        mock_resp.status_code = 400

        with patch("layerlens.cli._auth.httpx.post", return_value=mock_resp):
            with pytest.raises(LoginError, match="required"):
                cli_login("", "")


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestLoginCommand:
    def test_login_success(self, runner, creds_dir):
        mock_creds = {
            "access_token": "tok",
            "expires_at": time.time() + 3600,
            "user": {"email": "user@test.com", "given_name": "Test"},
        }

        with patch("layerlens.cli._auth.cli_login", return_value=mock_creds):
            with patch("layerlens.cli._auth.load_credentials", return_value=None):
                result = runner.invoke(cli, ["login"], input="user@test.com\nsecret\n")

        assert result.exit_code == 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "logged in" in combined.lower()

    def test_login_already_logged_in_decline(self, runner, creds_dir):
        existing = {"access_token": "old-tok"}

        with patch("layerlens.cli._auth.load_credentials", return_value=existing):
            result = runner.invoke(cli, ["login"], input="n\n")

        assert result.exit_code == 0

    def test_login_error(self, runner, creds_dir):
        from layerlens.cli._auth import LoginError

        with patch("layerlens.cli._auth.load_credentials", return_value=None):
            with patch("layerlens.cli._auth.cli_login", side_effect=LoginError("Invalid email or password.")):
                result = runner.invoke(cli, ["login"], input="bad@test.com\nwrong\n")

        assert result.exit_code != 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "invalid" in combined.lower()


class TestLogoutCommand:
    def test_logout_success(self, runner, creds_dir, sample_creds):
        save_credentials(sample_creds)
        result = runner.invoke(cli, ["logout"])
        assert result.exit_code == 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "logged out" in combined.lower()
        assert load_credentials() is None

    def test_logout_not_logged_in(self, runner, creds_dir):
        result = runner.invoke(cli, ["logout"])
        assert result.exit_code == 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "not currently" in combined.lower()


class TestWhoamiCommand:
    def test_whoami_with_env_var(self, runner, monkeypatch):
        monkeypatch.setenv("LAYERLENS_API_KEY", "env-key")
        result = runner.invoke(cli, ["whoami"])
        assert result.exit_code == 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "LAYERLENS_API_KEY" in combined

    def test_whoami_not_logged_in(self, runner, creds_dir, monkeypatch):
        monkeypatch.delenv("LAYERLENS_API_KEY", raising=False)
        result = runner.invoke(cli, ["whoami"])
        assert result.exit_code != 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "not logged in" in combined.lower()

    def test_whoami_shows_user_info(self, runner, creds_dir, sample_creds, monkeypatch):
        monkeypatch.delenv("LAYERLENS_API_KEY", raising=False)
        save_credentials(sample_creds)

        with patch("layerlens.cli._auth.get_valid_token", return_value="tok"):
            with patch(
                "layerlens.cli._auth.get_user_info",
                return_value={"email": "user@example.com", "name": "Test User", "sub": "abc-123"},
            ):
                result = runner.invoke(cli, ["whoami"])

        assert result.exit_code == 0
        combined = (result.output or "") + (getattr(result, "stderr", "") or "")
        assert "user@example.com" in combined
        assert "Test User" in combined
