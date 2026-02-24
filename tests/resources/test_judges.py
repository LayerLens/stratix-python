from unittest.mock import Mock

import pytest

from layerlens.models import Judge, JudgesResponse, DeleteJudgeResponse, UpdateJudgeResponse
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.judges.judges import Judges


class TestJudges:
    """Test Judges resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.patch_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def judges_resource(self, mock_client):
        """Judges resource instance."""
        return Judges(mock_client)

    @pytest.fixture
    def sample_judge_data(self):
        """Sample judge data for testing."""
        return {
            "id": "judge-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "name": "Code Quality Judge",
            "evaluation_goal": "Evaluate the quality of code output including correctness and style",
            "model_id": "model-789",
            "model_name": "GPT-4",
            "model_company": "OpenAI",
            "version": 1,
            "run_count": 5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "versions": [],
        }

    def test_judges_initialization(self, mock_client):
        """Judges resource initializes correctly."""
        judges = Judges(mock_client)

        assert judges._client is mock_client
        assert judges._get is mock_client.get_cast
        assert judges._post is mock_client.post_cast
        assert judges._patch is mock_client.patch_cast
        assert judges._delete is mock_client.delete_cast

    def test_create_judge_success(self, judges_resource, sample_judge_data):
        """create method returns Judge on success."""
        judges_resource._post.return_value = {"id": "judge-123"}
        judges_resource._get.return_value = sample_judge_data

        result = judges_resource.create(
            name="Code Quality Judge",
            evaluation_goal="Evaluate the quality of code output including correctness and style",
            model_id="model-789",
        )

        assert isinstance(result, Judge)
        assert result.id == "judge-123"
        assert result.name == "Code Quality Judge"

    def test_create_judge_request_parameters(self, judges_resource, sample_judge_data):
        """create method makes correct API request."""
        judges_resource._post.return_value = {"id": "judge-123"}
        judges_resource._get.return_value = sample_judge_data

        judges_resource.create(
            name="Test Judge",
            evaluation_goal="Test evaluation goal for testing",
            model_id="model-789",
        )

        judges_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judges",
            body={
                "name": "Test Judge",
                "evaluation_goal": "Test evaluation goal for testing",
                "model_id": "model-789",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_create_judge_without_model_id(self, judges_resource, sample_judge_data):
        """create method works without model_id."""
        judges_resource._post.return_value = {"id": "judge-123"}
        judges_resource._get.return_value = sample_judge_data

        judges_resource.create(
            name="Test Judge",
            evaluation_goal="Test evaluation goal for testing",
        )

        call_args = judges_resource._post.call_args
        assert "model_id" not in call_args.kwargs["body"]

    def test_create_judge_none_response(self, judges_resource):
        """create method returns None when response is not a dict."""
        judges_resource._post.return_value = None

        result = judges_resource.create(
            name="Test Judge",
            evaluation_goal="Test evaluation goal for testing",
        )

        assert result is None

    def test_get_judge_success(self, judges_resource, sample_judge_data):
        """get method returns Judge on success."""
        judges_resource._get.return_value = sample_judge_data

        result = judges_resource.get("judge-123")

        assert isinstance(result, Judge)
        assert result.id == "judge-123"
        assert result.name == "Code Quality Judge"

    def test_get_judge_request_parameters(self, judges_resource, sample_judge_data):
        """get method makes correct API request."""
        judges_resource._get.return_value = sample_judge_data

        judges_resource.get("judge-123")

        judges_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judges/judge-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_judge_none_response(self, judges_resource):
        """get method returns None when response is invalid."""
        judges_resource._get.return_value = None

        result = judges_resource.get("judge-123")

        assert result is None

    def test_get_many_judges_success(self, judges_resource, sample_judge_data):
        """get_many returns JudgesResponse on success."""
        judges_resource._get.return_value = {
            "judges": [sample_judge_data],
            "count": 1,
            "total_count": 1,
        }

        result = judges_resource.get_many()

        assert isinstance(result, JudgesResponse)
        assert len(result.judges) == 1
        assert result.judges[0].id == "judge-123"
        assert result.count == 1
        assert result.total_count == 1

    def test_get_many_judges_request_parameters(self, judges_resource, sample_judge_data):
        """get_many makes correct API request with pagination."""
        judges_resource._get.return_value = {
            "judges": [sample_judge_data],
            "count": 1,
            "total_count": 10,
        }

        judges_resource.get_many(page=2, page_size=50)

        judges_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judges",
            params={
                "page": "2",
                "pageSize": "50",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_many_judges_default_pagination(self, judges_resource, sample_judge_data):
        """get_many uses default pagination when not specified."""
        judges_resource._get.return_value = {
            "judges": [sample_judge_data],
            "count": 1,
            "total_count": 1,
        }

        judges_resource.get_many()

        call_args = judges_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["pageSize"] == "100"

    def test_get_many_judges_none_response(self, judges_resource):
        """get_many returns None when response is invalid."""
        judges_resource._get.return_value = None

        result = judges_resource.get_many()

        assert result is None

    def test_get_many_judges_empty_list(self, judges_resource):
        """get_many returns empty list when no judges exist."""
        judges_resource._get.return_value = {
            "judges": [],
            "count": 0,
            "total_count": 0,
        }

        result = judges_resource.get_many()

        assert isinstance(result, JudgesResponse)
        assert len(result.judges) == 0
        assert result.total_count == 0

    def test_update_judge_success(self, judges_resource):
        """update method returns UpdateJudgeResponse on success."""
        judges_resource._patch.return_value = {
            "organization_id": "org-123",
            "project_id": "proj-456",
            "id": "judge-123",
        }

        result = judges_resource.update(
            "judge-123",
            name="Updated Judge",
            evaluation_goal="Updated evaluation goal for the judge",
        )

        assert isinstance(result, UpdateJudgeResponse)
        assert result.id == "judge-123"

    def test_update_judge_request_parameters(self, judges_resource):
        """update method makes correct API request."""
        judges_resource._patch.return_value = {
            "organization_id": "org-123",
            "project_id": "proj-456",
            "id": "judge-123",
        }

        judges_resource.update(
            "judge-123",
            name="Updated Name",
            evaluation_goal="Updated goal text for testing",
            model_id="new-model",
        )

        judges_resource._patch.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judges/judge-123",
            body={
                "name": "Updated Name",
                "evaluation_goal": "Updated goal text for testing",
                "model_id": "new-model",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_update_judge_partial(self, judges_resource):
        """update method only sends provided fields."""
        judges_resource._patch.return_value = {
            "organization_id": "org-123",
            "project_id": "proj-456",
            "id": "judge-123",
        }

        judges_resource.update("judge-123", name="New Name Only")

        call_args = judges_resource._patch.call_args
        body = call_args.kwargs["body"]
        assert body == {"name": "New Name Only"}

    def test_update_judge_none_response(self, judges_resource):
        """update method returns None when response is invalid."""
        judges_resource._patch.return_value = None

        result = judges_resource.update("judge-123", name="Updated")

        assert result is None

    def test_delete_judge_success(self, judges_resource):
        """delete method returns DeleteJudgeResponse on success."""
        judges_resource._delete.return_value = {
            "organization_id": "org-123",
            "project_id": "proj-456",
            "id": "judge-123",
        }

        result = judges_resource.delete("judge-123")

        assert isinstance(result, DeleteJudgeResponse)
        assert result.id == "judge-123"

    def test_delete_judge_request_parameters(self, judges_resource):
        """delete method makes correct API request."""
        judges_resource._delete.return_value = {
            "organization_id": "org-123",
            "project_id": "proj-456",
            "id": "judge-123",
        }

        judges_resource.delete("judge-123")

        judges_resource._delete.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judges/judge-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_delete_judge_none_response(self, judges_resource):
        """delete method returns None when response is invalid."""
        judges_resource._delete.return_value = None

        result = judges_resource.delete("judge-123")

        assert result is None


class TestJudgesErrorHandling:
    """Test error handling in Judges resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.patch_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def judges_resource(self, mock_client):
        """Judges resource instance."""
        return Judges(mock_client)

    def test_get_judge_handles_not_found(self, judges_resource):
        """get method propagates not found errors."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        judges_resource._get.side_effect = NotFoundError("Judge not found", response=mock_response, body=None)

        with pytest.raises(NotFoundError):
            judges_resource.get("nonexistent-judge")

    def test_create_judge_handles_bad_request(self, judges_resource):
        """create method propagates bad request errors."""
        from layerlens._exceptions import BadRequestError

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        judges_resource._post.side_effect = BadRequestError("Invalid judge name", response=mock_response, body=None)

        with pytest.raises(BadRequestError):
            judges_resource.create(name="ab", evaluation_goal="too short")

    def test_update_judge_handles_auth_error(self, judges_resource):
        """update method propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        judges_resource._patch.side_effect = AuthenticationError("Unauthorized", response=mock_response, body=None)

        with pytest.raises(AuthenticationError):
            judges_resource.update("judge-123", name="Updated")

    def test_delete_judge_handles_permission_error(self, judges_resource):
        """delete method propagates permission errors."""
        from layerlens._exceptions import PermissionDeniedError

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        judges_resource._delete.side_effect = PermissionDeniedError("Access denied", response=mock_response, body=None)

        with pytest.raises(PermissionDeniedError):
            judges_resource.delete("judge-123")


class TestJudgesURLConstruction:
    """Test URL construction in Judges resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.organization_id = "custom-org"
        client.project_id = "custom-proj"
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.patch_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def judges_resource(self, mock_client):
        """Judges resource instance."""
        return Judges(mock_client)

    def test_base_url_construction(self, judges_resource):
        """Base URL uses correct organization and project IDs."""
        assert judges_resource._base_url() == "/organizations/custom-org/projects/custom-proj/judges"

    def test_get_url_includes_judge_id(self, judges_resource):
        """Get URL includes judge ID."""
        judges_resource._get.return_value = None

        judges_resource.get("judge-abc")

        call_args = judges_resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judges/judge-abc"

    def test_delete_url_includes_judge_id(self, judges_resource):
        """Delete URL includes judge ID."""
        judges_resource._delete.return_value = None

        judges_resource.delete("judge-xyz")

        call_args = judges_resource._delete.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judges/judge-xyz"

    def test_update_url_includes_judge_id(self, judges_resource):
        """Update URL includes judge ID."""
        judges_resource._patch.return_value = None

        judges_resource.update("judge-xyz", name="Updated")

        call_args = judges_resource._patch.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judges/judge-xyz"
