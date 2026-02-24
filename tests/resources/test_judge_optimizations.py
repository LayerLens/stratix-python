from unittest.mock import Mock

import pytest

from layerlens.models import (
    JudgeOptimizationRun,
    JudgeOptimizationRunsResponse,
    CreateJudgeOptimizationRunResponse,
    ApplyJudgeOptimizationResultResponse,
    EstimateJudgeOptimizationCostResponse,
)
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.judge_optimizations.judge_optimizations import JudgeOptimizations


class TestJudgeOptimizations:
    """Test JudgeOptimizations resource API methods."""

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
    def resource(self, mock_client):
        """JudgeOptimizations resource instance."""
        return JudgeOptimizations(mock_client)

    @pytest.fixture
    def sample_run_data(self):
        """Sample optimization run data."""
        return {
            "id": "run-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "judge_id": "judge-789",
            "user_id": "user-001",
            "status": "pending",
            "status_description": None,
            "judge_snapshot": None,
            "annotation_count": 50,
            "budget": "medium",
            "baseline_accuracy": None,
            "optimized_accuracy": None,
            "original_goal": None,
            "optimized_goal": None,
            "logs": None,
            "estimated_cost": 7.5,
            "actual_cost": 0.0,
            "created_at": "2024-06-01T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "applied_at": None,
            "applied_version": None,
        }

    @pytest.fixture
    def sample_completed_run_data(self):
        """Sample completed optimization run data."""
        return {
            "id": "run-456",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "judge_id": "judge-789",
            "user_id": "user-001",
            "status": "success",
            "status_description": None,
            "judge_snapshot": {
                "name": "Code Quality Judge",
                "version": 1,
                "evaluationGoal": "Evaluate code quality",
                "modelId": None,
                "modelName": None,
                "modelCompany": None,
            },
            "annotation_count": 50,
            "budget": "medium",
            "baseline_accuracy": 0.75,
            "optimized_accuracy": 0.82,
            "original_goal": "Evaluate code quality",
            "optimized_goal": "Evaluate code quality with emphasis on correctness, style, and maintainability",
            "logs": "[INFO] Starting optimization...\n[METRIC] Baseline: 0.75\n[METRIC] Optimized: 0.82",
            "estimated_cost": 7.5,
            "actual_cost": 5.25,
            "created_at": "2024-06-01T00:00:00Z",
            "started_at": "2024-06-01T00:01:00Z",
            "finished_at": "2024-06-01T00:10:00Z",
            "applied_at": None,
            "applied_version": None,
        }

    # --- estimate ---

    def test_estimate_success(self, resource):
        """estimate returns cost estimate on success."""
        resource._post.return_value = {"estimated_cost": 7.5, "annotation_count": 50, "budget": "medium"}

        result = resource.estimate(judge_id="judge-789", budget="medium")

        assert isinstance(result, EstimateJudgeOptimizationCostResponse)
        assert result.estimated_cost == 7.5
        assert result.annotation_count == 50
        assert result.budget == "medium"

    def test_estimate_request_parameters(self, resource):
        """estimate makes correct API request."""
        resource._post.return_value = {"estimated_cost": 7.5, "annotation_count": 50, "budget": "medium"}

        resource.estimate(judge_id="judge-789", budget="heavy")

        resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judge-optimizations/estimate",
            body={"judge_id": "judge-789", "budget": "heavy"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_estimate_default_budget(self, resource):
        """estimate uses medium budget by default."""
        resource._post.return_value = {"estimated_cost": 7.5, "annotation_count": 50, "budget": "medium"}

        resource.estimate(judge_id="judge-789")

        call_args = resource._post.call_args
        assert call_args.kwargs["body"]["budget"] == "medium"

    def test_estimate_none_response(self, resource):
        """estimate returns None when response is invalid."""
        resource._post.return_value = None

        result = resource.estimate(judge_id="judge-789")

        assert result is None

    # --- create ---

    def test_create_success(self, resource):
        """create returns CreateJudgeOptimizationRunResponse on success."""
        resource._post.return_value = {
            "id": "run-123",
            "judge_id": "judge-789",
            "budget": "medium",
            "status": "pending",
        }

        result = resource.create(judge_id="judge-789", budget="medium")

        assert isinstance(result, CreateJudgeOptimizationRunResponse)
        assert result.id == "run-123"
        assert result.judge_id == "judge-789"
        assert result.status == "pending"

    def test_create_request_parameters(self, resource):
        """create makes correct API request."""
        resource._post.return_value = {"id": "run-123", "judge_id": "judge-789", "budget": "light", "status": "pending"}

        resource.create(judge_id="judge-789", budget="light")

        resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judge-optimizations",
            body={"judge_id": "judge-789", "budget": "light"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_create_none_response(self, resource):
        """create returns None when response is invalid."""
        resource._post.return_value = None

        result = resource.create(judge_id="judge-789")

        assert result is None

    # --- get ---

    def test_get_success(self, resource, sample_run_data):
        """get returns JudgeOptimizationRun on success."""
        resource._get.return_value = sample_run_data

        result = resource.get("run-123")

        assert isinstance(result, JudgeOptimizationRun)
        assert result.id == "run-123"
        assert result.judge_id == "judge-789"
        assert result.status.value == "pending"

    def test_get_completed_run(self, resource, sample_completed_run_data):
        """get returns completed run with results."""
        resource._get.return_value = sample_completed_run_data

        result = resource.get("run-456")

        assert isinstance(result, JudgeOptimizationRun)
        assert result.status.value == "success"
        assert result.baseline_accuracy == 0.75
        assert result.optimized_accuracy == 0.82
        assert result.optimized_goal is not None

    def test_get_request_parameters(self, resource, sample_run_data):
        """get makes correct API request."""
        resource._get.return_value = sample_run_data

        resource.get("run-123")

        resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judge-optimizations/run-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_none_response(self, resource):
        """get returns None when response is invalid."""
        resource._get.return_value = None

        result = resource.get("run-123")

        assert result is None

    # --- get_many ---

    def test_get_many_success(self, resource, sample_run_data):
        """get_many returns JudgeOptimizationRunsResponse on success."""
        resource._get.return_value = {
            "optimization_runs": [sample_run_data],
            "count": 1,
            "total": 1,
        }

        result = resource.get_many()

        assert isinstance(result, JudgeOptimizationRunsResponse)
        assert len(result.optimization_runs) == 1
        assert result.optimization_runs[0].id == "run-123"
        assert result.count == 1
        assert result.total == 1

    def test_get_many_with_judge_id(self, resource, sample_run_data):
        """get_many filters by judge_id."""
        resource._get.return_value = {
            "optimization_runs": [sample_run_data],
            "count": 1,
            "total": 1,
        }

        resource.get_many(judge_id="judge-789", page=1, page_size=10)

        resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judge-optimizations",
            params={
                "page": "1",
                "page_size": "10",
                "judge_id": "judge-789",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_many_default_pagination(self, resource, sample_run_data):
        """get_many uses default pagination when not specified."""
        resource._get.return_value = {
            "optimization_runs": [sample_run_data],
            "count": 1,
            "total": 1,
        }

        resource.get_many()

        call_args = resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["page_size"] == "20"

    def test_get_many_none_response(self, resource):
        """get_many returns None when response is invalid."""
        resource._get.return_value = None

        result = resource.get_many()

        assert result is None

    def test_get_many_empty_list(self, resource):
        """get_many returns empty list when no runs exist."""
        resource._get.return_value = {
            "optimization_runs": [],
            "count": 0,
            "total": 0,
        }

        result = resource.get_many()

        assert isinstance(result, JudgeOptimizationRunsResponse)
        assert len(result.optimization_runs) == 0
        assert result.total == 0

    # --- apply ---

    def test_apply_success(self, resource):
        """apply returns ApplyJudgeOptimizationResultResponse on success."""
        resource._post.return_value = {
            "judge_id": "judge-789",
            "new_version": 3,
            "message": "Optimization result applied successfully",
        }

        result = resource.apply("run-456")

        assert isinstance(result, ApplyJudgeOptimizationResultResponse)
        assert result.judge_id == "judge-789"
        assert result.new_version == 3

    def test_apply_request_parameters(self, resource):
        """apply makes correct API request."""
        resource._post.return_value = {
            "judge_id": "judge-789",
            "new_version": 3,
            "message": "Optimization result applied successfully",
        }

        resource.apply("run-456")

        resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/judge-optimizations/run-456/apply",
            body={},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_apply_none_response(self, resource):
        """apply returns None when response is invalid."""
        resource._post.return_value = None

        result = resource.apply("run-456")

        assert result is None


class TestJudgeOptimizationsErrorHandling:
    """Test error handling in JudgeOptimizations resource."""

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
    def resource(self, mock_client):
        """JudgeOptimizations resource instance."""
        return JudgeOptimizations(mock_client)

    def test_estimate_handles_bad_request(self, resource):
        """estimate propagates bad request errors."""
        from layerlens._exceptions import BadRequestError

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        resource._post.side_effect = BadRequestError(
            "Minimum 10 annotations required", response=mock_response, body=None
        )

        with pytest.raises(BadRequestError):
            resource.estimate(judge_id="judge-789")

    def test_create_handles_bad_request(self, resource):
        """create propagates bad request errors."""
        from layerlens._exceptions import BadRequestError

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        resource._post.side_effect = BadRequestError("Invalid budget value", response=mock_response, body=None)

        with pytest.raises(BadRequestError):
            resource.create(judge_id="judge-789", budget="invalid")

    def test_get_handles_not_found(self, resource):
        """get propagates not found errors."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        resource._get.side_effect = NotFoundError("Optimization run not found", response=mock_response, body=None)

        with pytest.raises(NotFoundError):
            resource.get("nonexistent-run")

    def test_apply_handles_conflict(self, resource):
        """apply propagates conflict errors (already applied)."""
        from layerlens._exceptions import ConflictError

        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.headers = {}

        resource._post.side_effect = ConflictError("Optimization already applied", response=mock_response, body=None)

        with pytest.raises(ConflictError):
            resource.apply("run-456")

    def test_apply_handles_auth_error(self, resource):
        """apply propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        resource._post.side_effect = AuthenticationError("Unauthorized", response=mock_response, body=None)

        with pytest.raises(AuthenticationError):
            resource.apply("run-456")


class TestJudgeOptimizationsURLConstruction:
    """Test URL construction in JudgeOptimizations resource."""

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
    def resource(self, mock_client):
        """JudgeOptimizations resource instance."""
        return JudgeOptimizations(mock_client)

    def test_base_url_construction(self, resource):
        """Base URL uses correct organization and project IDs."""
        assert resource._base_url() == "/organizations/custom-org/projects/custom-proj/judge-optimizations"

    def test_get_url_includes_run_id(self, resource):
        """Get URL includes run ID."""
        resource._get.return_value = None

        resource.get("run-abc")

        call_args = resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judge-optimizations/run-abc"

    def test_apply_url_includes_run_id(self, resource):
        """Apply URL includes run ID and /apply suffix."""
        resource._post.return_value = None

        resource.apply("run-xyz")

        call_args = resource._post.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judge-optimizations/run-xyz/apply"

    def test_estimate_url(self, resource):
        """Estimate URL includes /estimate suffix."""
        resource._post.return_value = None

        resource.estimate(judge_id="judge-123")

        call_args = resource._post.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/judge-optimizations/estimate"
