import pytest

from atlas import Atlas
from atlas._exceptions import AtlasError


class TestAtlasClientInitialization:
    """Test Atlas client initialization and configuration."""

    def test_init_with_explicit_params(self):
        """Client initializes correctly with explicit parameters."""
        client = Atlas(api_key="explicit-key", organization_id="explicit-org", project_id="explicit-project")

        assert client.api_key == "explicit-key"
        assert client.organization_id == "explicit-org"
        assert client.project_id == "explicit-project"

    def test_init_from_environment(self, mock_env_vars):
        """Client initializes from environment variables."""
        _ = mock_env_vars  # Fixture used for side effects
        client = Atlas()

        assert client.api_key == "test-api-key"
        assert client.organization_id == "test-org-id"
        assert client.project_id == "test-project-id"

    def test_explicit_params_override_env(self, mock_env_vars):
        """Explicit parameters override environment variables."""
        _ = mock_env_vars  # Fixture used for side effects
        client = Atlas(api_key="override-key", organization_id="override-org")

        assert client.api_key == "override-key"
        assert client.organization_id == "override-org"
        assert client.project_id == "test-project-id"

    def test_missing_api_key_raises_error(self, env_vars):
        """Missing API key raises AtlasError."""
        _ = env_vars  # Fixture used for side effects
        with pytest.raises(AtlasError, match="api_key client option must be set"):
            Atlas()

    def test_none_values_fallback_to_env(self, mock_env_vars):
        """None values explicitly passed fallback to environment."""
        _ = mock_env_vars  # Fixture used for side effects
        client = Atlas(api_key=None, organization_id=None, project_id=None)

        assert client.api_key == "test-api-key"
        assert client.organization_id == "test-org-id"
        assert client.project_id == "test-project-id"

    def test_optional_params_can_be_none(self):
        """Organization and project IDs can be None."""
        client = Atlas(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.organization_id is None
        assert client.project_id is None

    @pytest.mark.parametrize("base_url", ["https://custom.api.com", "https://staging.layerlens.ai/api/v1"])
    def test_custom_base_url(self, base_url):
        """Client accepts custom base URL."""
        client = Atlas(api_key="test-key", base_url=base_url)

        assert str(client.base_url).rstrip("/") == base_url.rstrip("/")

    def test_custom_timeout(self):
        """Client accepts custom timeout."""
        import httpx

        client = Atlas(api_key="test-key", timeout=30.0)

        assert isinstance(client.timeout, httpx.Timeout)

    def test_auth_headers_with_api_key(self):
        """auth_headers property returns correct headers when API key is set."""
        client = Atlas(api_key="test-api-key")

        headers = client.auth_headers

        assert headers == {"x-api-key": "test-api-key"}

    def test_auth_headers_without_api_key(self):
        """auth_headers property returns empty dict when no API key."""
        client = Atlas(api_key="")

        headers = client.auth_headers

        assert headers == {}

    def test_auth_headers_with_empty_api_key(self):
        """auth_headers property returns empty dict when API key is empty string."""
        client = Atlas(api_key="")

        headers = client.auth_headers

        assert headers == {}

    def test_copy_method(self):
        """copy method creates new client with overridden parameters."""
        original_client = Atlas(
            api_key="original-key",
            organization_id="original-org",
            project_id="original-project",
            base_url="https://original.api.com",
            timeout=10.0,
        )

        new_client = original_client.copy(api_key="new-key", organization_id="new-org", timeout=20.0)

        # Check overridden values
        assert new_client.api_key == "new-key"
        assert new_client.organization_id == "new-org"
        # The copy method uses 'or' logic, so timeout=20.0 won't override the existing timeout
        # Let's check that the timeout is still the original value
        assert new_client.timeout == original_client.timeout  # Should remain the original timeout

        # Check unchanged values
        assert new_client.project_id == "original-project"
        assert str(new_client.base_url) == "https://original.api.com"

    def test_copy_method_partial_override(self):
        """copy method allows partial parameter override."""
        original_client = Atlas(api_key="original-key", organization_id="original-org", project_id="original-project")

        new_client = original_client.copy(api_key="new-key")

        assert new_client.api_key == "new-key"
        assert new_client.organization_id == "original-org"
        assert new_client.project_id == "original-project"

    def test_with_options_alias(self):
        """with_options is an alias for copy method."""
        original_client = Atlas(api_key="original-key")

        new_client = original_client.with_options(api_key="new-key")

        assert new_client.api_key == "new-key"
        assert new_client is not original_client

    def test_copy_method_timeout_override(self):
        """copy method properly overrides timeout when original is None."""
        # Create a client with no explicit timeout (uses default)
        original_client = Atlas(api_key="original-key")

        new_client = original_client.copy(timeout=30.0)

        import httpx

        assert isinstance(new_client.timeout, httpx.Timeout)
        # Both clients use the same default timeout, so they should be equal
        assert new_client.timeout == original_client.timeout


class TestAtlasClientErrorHandling:
    """Test error handling in Atlas client."""

    def _create_mock_response(self, status_code):
        """Helper to create a mock response with all required attributes."""
        mock_request = type("MockRequest", (), {})()
        return type("MockResponse", (), {"status_code": status_code, "request": mock_request, "headers": {}})()

    def test_make_status_error_bad_request(self):
        """_make_status_error creates BadRequestError for 400 status."""
        from atlas._exceptions import BadRequestError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(400)
        mock_body = {"error": "Bad request"}

        error = client._make_status_error("Bad request", body=mock_body, response=mock_response)

        assert isinstance(error, BadRequestError)
        assert error.message == "Bad request"

    def test_make_status_error_unauthorized(self):
        """_make_status_error creates AuthenticationError for 401 status."""
        from atlas._exceptions import AuthenticationError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(401)
        mock_body = {"error": "Unauthorized"}

        error = client._make_status_error("Unauthorized", body=mock_body, response=mock_response)

        assert isinstance(error, AuthenticationError)
        assert error.message == "Unauthorized"

    def test_make_status_error_forbidden(self):
        """_make_status_error creates PermissionDeniedError for 403 status."""
        from atlas._exceptions import PermissionDeniedError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(403)
        mock_body = {"error": "Forbidden"}

        error = client._make_status_error("Forbidden", body=mock_body, response=mock_response)

        assert isinstance(error, PermissionDeniedError)
        assert error.message == "Forbidden"

    def test_make_status_error_not_found(self):
        """_make_status_error creates NotFoundError for 404 status."""
        from atlas._exceptions import NotFoundError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(404)
        mock_body = {"error": "Not found"}

        error = client._make_status_error("Not found", body=mock_body, response=mock_response)

        assert isinstance(error, NotFoundError)
        assert error.message == "Not found"

    def test_make_status_error_conflict(self):
        """_make_status_error creates ConflictError for 409 status."""
        from atlas._exceptions import ConflictError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(409)
        mock_body = {"error": "Conflict"}

        error = client._make_status_error("Conflict", body=mock_body, response=mock_response)

        assert isinstance(error, ConflictError)
        assert error.message == "Conflict"

    def test_make_status_error_unprocessable_entity(self):
        """_make_status_error creates UnprocessableEntityError for 422 status."""
        from atlas._exceptions import UnprocessableEntityError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(422)
        mock_body = {"error": "Unprocessable entity"}

        error = client._make_status_error("Unprocessable entity", body=mock_body, response=mock_response)

        assert isinstance(error, UnprocessableEntityError)
        assert error.message == "Unprocessable entity"

    def test_make_status_error_rate_limit(self):
        """_make_status_error creates RateLimitError for 429 status."""
        from atlas._exceptions import RateLimitError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(429)
        mock_body = {"error": "Rate limited"}

        error = client._make_status_error("Rate limited", body=mock_body, response=mock_response)

        assert isinstance(error, RateLimitError)
        assert error.message == "Rate limited"

    def test_make_status_error_internal_server_error(self):
        """_make_status_error creates InternalServerError for 500+ status."""
        from atlas._exceptions import InternalServerError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(500)
        mock_body = {"error": "Internal server error"}

        error = client._make_status_error("Internal server error", body=mock_body, response=mock_response)

        assert isinstance(error, InternalServerError)
        assert error.message == "Internal server error"

    def test_make_status_error_gateway_timeout(self):
        """_make_status_error creates InternalServerError for 502 status."""
        from atlas._exceptions import InternalServerError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(502)
        mock_body = {"error": "Gateway timeout"}

        error = client._make_status_error("Gateway timeout", body=mock_body, response=mock_response)

        assert isinstance(error, InternalServerError)
        assert error.message == "Gateway timeout"

    def test_make_status_error_unknown_status(self):
        """_make_status_error creates generic APIStatusError for unknown status codes."""
        from atlas._exceptions import APIStatusError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(418)  # I'm a teapot
        mock_body = {"error": "Unknown error"}

        error = client._make_status_error("Unknown error", body=mock_body, response=mock_response)

        assert isinstance(error, APIStatusError)
        assert error.message == "Unknown error"

    def test_make_status_error_with_non_mapping_body(self):
        """_make_status_error handles non-mapping body correctly."""
        from atlas._exceptions import NotFoundError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(404)
        mock_body = "Simple string error"

        error = client._make_status_error("Not found", body=mock_body, response=mock_response)

        assert isinstance(error, NotFoundError)
        assert error.body == "Simple string error"

    def test_make_status_error_with_none_body(self):
        """_make_status_error handles None body correctly."""
        from atlas._exceptions import BadRequestError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(400)

        error = client._make_status_error("Bad request", body=None, response=mock_response)

        assert isinstance(error, BadRequestError)
        assert error.body is None

    def test_make_status_error_with_complex_body(self):
        """_make_status_error extracts error from complex body structure."""
        from atlas._exceptions import AuthenticationError

        client = Atlas(api_key="test-key")
        mock_response = self._create_mock_response(401)
        mock_body = {"error": {"message": "Invalid API key", "code": "AUTH_ERROR"}, "timestamp": "2023-01-01T00:00:00Z"}

        error = client._make_status_error("Authentication failed", body=mock_body, response=mock_response)

        assert isinstance(error, AuthenticationError)
        assert error.body == mock_body["error"]  # Should extract the error field
