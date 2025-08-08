import pytest

from atlas import Atlas
from atlas._exceptions import AtlasError


class TestAtlasClientInitialization:
    """Test Atlas client initialization and configuration."""

    def test_init_with_explicit_params(self):
        """Client initializes correctly with explicit parameters."""
        client = Atlas(
            api_key="explicit-key",
            organization_id="explicit-org",
            project_id="explicit-project"
        )
        
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
        client = Atlas(
            api_key="override-key",
            organization_id="override-org"
        )
        
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
        client = Atlas(
            api_key=None,
            organization_id=None,
            project_id=None
        )
        
        assert client.api_key == "test-api-key"
        assert client.organization_id == "test-org-id"
        assert client.project_id == "test-project-id"

    def test_optional_params_can_be_none(self):
        """Organization and project IDs can be None."""
        client = Atlas(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.organization_id is None
        assert client.project_id is None

    @pytest.mark.parametrize("base_url", [
        "https://custom.api.com",
        "https://staging.layerlens.ai/api/v1"
    ])
    def test_custom_base_url(self, base_url):
        """Client accepts custom base URL."""
        client = Atlas(api_key="test-key", base_url=base_url)
        
        assert str(client.base_url).rstrip('/') == base_url.rstrip('/')

    def test_custom_timeout(self):
        """Client accepts custom timeout."""
        import httpx
        client = Atlas(api_key="test-key", timeout=30.0)
        
        assert isinstance(client.timeout, httpx.Timeout)