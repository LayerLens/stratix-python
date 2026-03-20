from __future__ import annotations

from unittest.mock import Mock

import pytest

from layerlens.models.evaluation_space import EvaluationSpace
from layerlens.resources.evaluation_spaces.evaluation_spaces import EvaluationSpaces


class TestEvaluationSpaces:
    """Test EvaluationSpaces resource API methods."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.put_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def spaces_resource(self, mock_client):
        return EvaluationSpaces(mock_client)

    @pytest.fixture
    def sample_space_data(self):
        return {
            "id": "sp-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "name": "Q1 Comparison",
            "description": "Compare models for Q1",
            "visibility": "private",
            "owner": "admin@test.com",
            "created_at": "2026-01-01T00:00:00Z",
        }

    def test_base_url(self, spaces_resource):
        """Base URL includes org and project."""
        assert spaces_resource._base_url() == "/organizations/org-123/projects/proj-456/evaluation-spaces"

    def test_get_success(self, spaces_resource, sample_space_data):
        """get returns EvaluationSpace on success."""
        spaces_resource._get.return_value = {"status": "success", "data": sample_space_data}

        result = spaces_resource.get("sp-123")

        assert isinstance(result, EvaluationSpace)
        assert result.name == "Q1 Comparison"

    def test_get_not_found(self, spaces_resource):
        """get returns None when not found."""
        spaces_resource._get.return_value = None

        result = spaces_resource.get("nonexistent")

        assert result is None

    def test_get_many_success(self, spaces_resource, sample_space_data):
        """get_many returns EvaluationSpacesResponse."""
        spaces_resource._get.return_value = {
            "status": "success",
            "data": {"evaluation_spaces": [sample_space_data], "count": 1, "total_count": 1},
        }

        result = spaces_resource.get_many()

        assert result is not None
        assert len(result.evaluation_spaces) == 1
        assert result.evaluation_spaces[0].name == "Q1 Comparison"

    def test_get_many_pagination(self, spaces_resource):
        """get_many passes pagination and sort parameters."""
        spaces_resource._get.return_value = {
            "status": "success",
            "data": {"evaluation_spaces": [], "count": 0, "total_count": 0},
        }

        spaces_resource.get_many(page=2, page_size=10, sort_by="created_at", order="desc")

        call_params = spaces_resource._get.call_args[1]["params"]
        assert call_params["page"] == "2"
        assert call_params["page_size"] == "10"
        assert call_params["sort_by"] == "created_at"
        assert call_params["order"] == "desc"

    def test_create_success(self, spaces_resource, sample_space_data):
        """create returns EvaluationSpace."""
        spaces_resource._post.return_value = {"status": "success", "data": sample_space_data}

        result = spaces_resource.create(name="Q1 Comparison", description="Compare models for Q1")

        assert result is not None
        assert result.name == "Q1 Comparison"

    def test_create_request_body(self, spaces_resource):
        """create sends correct body."""
        spaces_resource._post.return_value = {"status": "success", "data": {"name": "Test"}}

        spaces_resource.create(name="Test", description="Desc", visibility="public")

        call_body = spaces_resource._post.call_args[1]["body"]
        assert call_body["name"] == "Test"
        assert call_body["description"] == "Desc"
        assert call_body["visibility"] == "public"

    def test_delete_success(self, spaces_resource):
        """delete returns True on success."""
        spaces_resource._delete.return_value = {}

        result = spaces_resource.delete("sp-123")

        assert result is True

    def test_delete_failure(self, spaces_resource):
        """delete returns False on exception."""
        spaces_resource._delete.side_effect = Exception("error")

        result = spaces_resource.delete("sp-123")

        assert result is False
