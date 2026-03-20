from __future__ import annotations

from unittest.mock import Mock

import pytest

from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.models.integration import Integration
from layerlens.resources.integrations.integrations import Integrations


class TestIntegrations:
    """Test Integrations resource API methods."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def integrations_resource(self, mock_client):
        return Integrations(mock_client)

    @pytest.fixture
    def sample_integration_data(self):
        return {
            "id": "int-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "name": "Langfuse Prod",
            "type": "langfuse",
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
        }

    def test_base_url_org_level(self, integrations_resource):
        """Base URL is at organization level (no project_id)."""
        url = integrations_resource._base_url()
        assert url == "/organizations/org-123/integrations"
        assert "project" not in url

    def test_get_success(self, integrations_resource, sample_integration_data):
        """get returns Integration on success."""
        integrations_resource._get.return_value = sample_integration_data

        result = integrations_resource.get("int-123")

        assert isinstance(result, Integration)
        assert result.id == "int-123"
        assert result.name == "Langfuse Prod"

    def test_get_with_envelope(self, integrations_resource, sample_integration_data):
        """get handles {status, data} envelope."""
        integrations_resource._get.return_value = {"status": "success", "data": sample_integration_data}

        result = integrations_resource.get("int-123")

        assert isinstance(result, Integration)

    def test_get_not_found(self, integrations_resource):
        """get returns None when not found."""
        integrations_resource._get.return_value = None

        result = integrations_resource.get("nonexistent")

        assert result is None

    def test_get_many_success(self, integrations_resource, sample_integration_data):
        """get_many returns IntegrationsResponse."""
        integrations_resource._get.return_value = {
            "status": "success",
            "data": {"integrations": [sample_integration_data], "count": 1, "total_count": 1},
        }

        result = integrations_resource.get_many()

        assert result is not None
        assert len(result.integrations) == 1
        assert result.integrations[0].name == "Langfuse Prod"
        assert result.count == 1

    def test_get_many_empty(self, integrations_resource):
        """get_many returns empty list."""
        integrations_resource._get.return_value = {
            "status": "success",
            "data": {"integrations": [], "count": 0, "total_count": 0},
        }

        result = integrations_resource.get_many()

        assert result is not None
        assert len(result.integrations) == 0

    def test_get_many_pagination(self, integrations_resource, sample_integration_data):
        """get_many passes pagination parameters."""
        integrations_resource._get.return_value = {
            "status": "success",
            "data": {"integrations": [sample_integration_data], "count": 1, "total_count": 10},
        }

        integrations_resource.get_many(page=2, page_size=5)

        integrations_resource._get.assert_called_once_with(
            "/organizations/org-123/integrations",
            params={"page": "2", "page_size": "5"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_test_integration_success(self, integrations_resource):
        """test returns TestIntegrationResponse."""
        integrations_resource._post.return_value = {
            "status": "success",
            "data": {"success": True, "message": "Connection OK"},
        }

        result = integrations_resource.test("int-123")

        assert result is not None
        assert result.success is True
        assert result.message == "Connection OK"

    def test_test_integration_failure(self, integrations_resource):
        """test returns failure result."""
        integrations_resource._post.return_value = {
            "status": "success",
            "data": {"success": False, "message": "Connection refused"},
        }

        result = integrations_resource.test("int-123")

        assert result is not None
        assert result.success is False
