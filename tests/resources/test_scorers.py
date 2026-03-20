from __future__ import annotations

from unittest.mock import Mock

import pytest

from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.models.scorer import Scorer
from layerlens.resources.scorers.scorers import Scorers, _normalize_keys, _pascal_to_snake


class TestPascalToSnake:
    """Test PascalCase to snake_case conversion."""

    def test_simple(self):
        assert _pascal_to_snake("Name") == "name"

    def test_two_words(self):
        assert _pascal_to_snake("ModelId") == "model_id"

    def test_three_words(self):
        assert _pascal_to_snake("ModelCompany") == "model_company"

    def test_consecutive_caps(self):
        assert _pascal_to_snake("ModelID") == "model_id"

    def test_already_snake(self):
        assert _pascal_to_snake("model_id") == "model_id"

    def test_single_char(self):
        assert _pascal_to_snake("A") == "a"


class TestNormalizeKeys:
    """Test dict key normalization."""

    def test_pascal_keys(self):
        d = {"Name": "test", "ModelId": "m-1", "ModelCompany": "OpenAI"}
        result = _normalize_keys(d)
        assert result["name"] == "test"
        assert result["model_id"] == "m-1"
        assert result["model_company"] == "OpenAI"

    def test_snake_keys_passthrough(self):
        d = {"name": "test", "model_id": "m-1"}
        result = _normalize_keys(d)
        assert result is d  # Same object, not copied

    def test_empty_dict(self):
        result = _normalize_keys({})
        assert result == {}


class TestScorers:
    """Test Scorers resource API methods."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.patch_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def scorers_resource(self, mock_client):
        return Scorers(mock_client)

    @pytest.fixture
    def sample_scorer_data(self):
        return {
            "id": "s-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "name": "Quality Scorer",
            "description": "Rates quality",
            "model_id": "m-1",
            "model_name": "GPT-4",
            "model_key": "openai/gpt-4",
            "model_company": "OpenAI",
            "prompt": "Rate quality",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }

    def test_base_url(self, scorers_resource):
        """Base URL includes org and project."""
        assert scorers_resource._base_url() == "/organizations/org-123/projects/proj-456/scorers"

    def test_get_success(self, scorers_resource, sample_scorer_data):
        """get returns Scorer on success."""
        scorers_resource._get.return_value = sample_scorer_data

        result = scorers_resource.get("s-123")

        assert isinstance(result, Scorer)
        assert result.name == "Quality Scorer"

    def test_get_with_envelope(self, scorers_resource, sample_scorer_data):
        """get handles {status, data} envelope."""
        scorers_resource._get.return_value = {"status": "success", "data": sample_scorer_data}

        result = scorers_resource.get("s-123")

        assert isinstance(result, Scorer)
        assert result.id == "s-123"

    def test_get_not_found(self, scorers_resource):
        """get returns None when not found."""
        scorers_resource._get.return_value = None

        result = scorers_resource.get("nonexistent")

        assert result is None

    def test_get_many_success(self, scorers_resource, sample_scorer_data):
        """get_many returns ScorersResponse."""
        scorers_resource._get.return_value = {
            "status": "success",
            "data": {"scorers": [sample_scorer_data], "count": 1, "total_count": 1},
        }

        result = scorers_resource.get_many()

        assert result is not None
        assert len(result.scorers) == 1
        assert result.scorers[0].name == "Quality Scorer"

    def test_get_many_empty(self, scorers_resource):
        """get_many returns empty list when no scorers."""
        scorers_resource._get.return_value = {
            "status": "success",
            "data": {"scorers": [], "count": 0, "total_count": 0},
        }

        result = scorers_resource.get_many()

        assert result is not None
        assert len(result.scorers) == 0

    def test_create_with_pascal_response(self, scorers_resource):
        """create handles PascalCase API response."""
        scorers_resource._post.return_value = {
            "status": "success",
            "data": {
                "Name": "New Scorer",
                "Description": "Desc",
                "ModelID": "m-1",
                "ModelName": "GPT-4",
                "ModelCompany": "OpenAI",
                "ModelKey": "openai/gpt-4",
                "Prompt": "Rate it",
                "CreatedAt": "2026-01-01",
                "UpdatedAt": "2026-01-01",
            },
        }

        result = scorers_resource.create(name="New Scorer", description="Desc", model_id="m-1", prompt="Rate it")

        assert result is not None
        assert result.name == "New Scorer"
        assert result.model_name == "GPT-4"

    def test_create_request_parameters(self, scorers_resource):
        """create sends correct body."""
        scorers_resource._post.return_value = {"status": "success", "data": {"Name": "X", "Prompt": "Y"}}

        scorers_resource.create(name="X", description="D", model_id="m-1", prompt="Y")

        scorers_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/scorers",
            body={"name": "X", "description": "D", "model_id": "m-1", "prompt": "Y"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_delete_success(self, scorers_resource):
        """delete returns True on success."""
        scorers_resource._delete.return_value = {}

        result = scorers_resource.delete("s-123")

        assert result is True

    def test_delete_failure(self, scorers_resource):
        """delete returns False on exception."""
        scorers_resource._delete.side_effect = Exception("not found")

        result = scorers_resource.delete("s-123")

        assert result is False

    def test_update_sends_patch(self, scorers_resource):
        """update sends PATCH with only provided fields."""
        scorers_resource._patch.return_value = {}

        result = scorers_resource.update("s-123", name="Updated")

        assert result is True
        scorers_resource._patch.assert_called_once()
        call_body = scorers_resource._patch.call_args[1]["body"]
        assert call_body == {"name": "Updated"}
