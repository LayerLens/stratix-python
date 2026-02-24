from unittest.mock import Mock

import pytest

from layerlens.models import (
    TraceEvaluation,
    CostEstimateResponse,
    TraceEvaluationStatus,
    TraceEvaluationsResponse,
    TraceEvaluationResultsResponse,
)
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.trace_evaluations.trace_evaluations import TraceEvaluations


class TestTraceEvaluations:
    """Test TraceEvaluations resource API methods."""

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
    def trace_evals_resource(self, mock_client):
        """TraceEvaluations resource instance."""
        return TraceEvaluations(mock_client)

    @pytest.fixture
    def sample_trace_eval_data(self):
        """Sample trace evaluation data."""
        return {
            "id": "te-123",
            "trace_id": "trace-456",
            "judge_id": "judge-789",
            "status": "success",
            "judge_snapshot": {
                "name": "Quality Judge",
                "version": 1,
                "evaluationGoal": "Evaluate output quality",
                "modelId": "model-1",
                "modelName": "GPT-4",
                "modelCompany": "OpenAI",
            },
            "created_at": "2024-01-01T00:00:00Z",
            "started_at": "2024-01-01T00:00:01Z",
            "finished_at": "2024-01-01T00:00:05Z",
        }

    @pytest.fixture
    def sample_result_data(self):
        """Sample trace evaluation result data."""
        return {
            "id": "result-123",
            "trace_evaluation_id": "te-123",
            "trace_id": "trace-456",
            "judge_id": "judge-789",
            "score": 0.85,
            "passed": True,
            "reasoning": "The output meets quality standards",
            "steps": [
                {"step": 1, "reasoning": "Checked correctness"},
                {"step": 2, "reasoning": "Checked style"},
            ],
            "model": "claude-sonnet-4-20250514",
            "turns": 3,
            "latency_ms": 2500,
            "prompt_tokens": 1500,
            "completion_tokens": 300,
            "total_cost": 0.0045,
            "created_at": "2024-01-01T00:00:05Z",
        }

    def test_trace_evaluations_initialization(self, mock_client):
        """TraceEvaluations resource initializes correctly."""
        te = TraceEvaluations(mock_client)

        assert te._client is mock_client
        assert te._get is mock_client.get_cast
        assert te._post is mock_client.post_cast

    def test_create_trace_evaluation_success(self, trace_evals_resource, sample_trace_eval_data):
        """create method returns TraceEvaluation on success."""
        trace_evals_resource._post.return_value = sample_trace_eval_data

        result = trace_evals_resource.create(trace_id="trace-456", judge_id="judge-789")

        assert isinstance(result, TraceEvaluation)
        assert result.id == "te-123"
        assert result.trace_id == "trace-456"
        assert result.judge_id == "judge-789"
        assert result.status == TraceEvaluationStatus.SUCCESS

    def test_create_trace_evaluation_request_parameters(self, trace_evals_resource, sample_trace_eval_data):
        """create method makes correct API request."""
        trace_evals_resource._post.return_value = sample_trace_eval_data

        trace_evals_resource.create(trace_id="trace-456", judge_id="judge-789")

        trace_evals_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/trace-evaluations",
            body={"trace_id": "trace-456", "judge_id": "judge-789"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_create_trace_evaluation_none_response(self, trace_evals_resource):
        """create method returns None when response is invalid."""
        trace_evals_resource._post.return_value = None

        result = trace_evals_resource.create(trace_id="trace-456", judge_id="judge-789")

        assert result is None

    def test_create_trace_evaluation_with_judge_snapshot(self, trace_evals_resource, sample_trace_eval_data):
        """create method handles judge snapshot data."""
        trace_evals_resource._post.return_value = sample_trace_eval_data

        result = trace_evals_resource.create(trace_id="trace-456", judge_id="judge-789")

        assert result.judge_snapshot is not None
        assert result.judge_snapshot.name == "Quality Judge"
        assert result.judge_snapshot.version == 1
        assert result.judge_snapshot.evaluation_goal == "Evaluate output quality"

    def test_get_trace_evaluation_success(self, trace_evals_resource, sample_trace_eval_data):
        """get method returns TraceEvaluation on success."""
        trace_evals_resource._get.return_value = TraceEvaluation(**sample_trace_eval_data)

        result = trace_evals_resource.get("te-123")

        assert isinstance(result, TraceEvaluation)
        assert result.id == "te-123"

    def test_get_trace_evaluation_request_parameters(self, trace_evals_resource, sample_trace_eval_data):
        """get method makes correct API request."""
        trace_evals_resource._get.return_value = TraceEvaluation(**sample_trace_eval_data)

        trace_evals_resource.get("te-123")

        trace_evals_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/trace-evaluations/te-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=TraceEvaluation,
        )

    def test_get_trace_evaluation_none_response(self, trace_evals_resource):
        """get method returns None when response is invalid."""
        trace_evals_resource._get.return_value = None

        result = trace_evals_resource.get("te-123")

        assert result is None

    def test_get_many_trace_evaluations_success(self, trace_evals_resource, sample_trace_eval_data):
        """get_many returns TraceEvaluationsResponse on success."""
        trace_evals_resource._get.return_value = {
            "trace_evaluations": [sample_trace_eval_data],
            "count": 1,
            "total": 1,
        }

        result = trace_evals_resource.get_many()

        assert isinstance(result, TraceEvaluationsResponse)
        assert len(result.trace_evaluations) == 1
        assert result.trace_evaluations[0].id == "te-123"
        assert result.count == 1
        assert result.total == 1

    def test_get_many_with_filters(self, trace_evals_resource, sample_trace_eval_data):
        """get_many passes filter parameters correctly."""
        trace_evals_resource._get.return_value = {
            "trace_evaluations": [sample_trace_eval_data],
            "count": 1,
            "total": 1,
        }

        trace_evals_resource.get_many(
            page=2,
            page_size=10,
            judge_id="judge-789",
            trace_id="trace-456",
            outcome="pass",
            time_range="7d",
            search="quality",
            sort_by="created_at",
            sort_order="desc",
        )

        call_args = trace_evals_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "2"
        assert params["page_size"] == "10"
        assert params["judgeId"] == "judge-789"
        assert params["traceId"] == "trace-456"
        assert params["outcome"] == "pass"
        assert params["timeRange"] == "7d"
        assert params["search"] == "quality"
        assert params["sortBy"] == "created_at"
        assert params["sortOrder"] == "desc"

    def test_get_many_default_pagination(self, trace_evals_resource):
        """get_many uses default pagination."""
        trace_evals_resource._get.return_value = {
            "trace_evaluations": [],
            "count": 0,
            "total": 0,
        }

        trace_evals_resource.get_many()

        call_args = trace_evals_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["page_size"] == "20"

    def test_get_many_none_response(self, trace_evals_resource):
        """get_many returns None when response is invalid."""
        trace_evals_resource._get.return_value = None

        result = trace_evals_resource.get_many()

        assert result is None

    def test_get_results_success(self, trace_evals_resource, sample_result_data):
        """get_results returns TraceEvaluationResultsResponse on success."""
        trace_evals_resource._get.return_value = {
            "results": [sample_result_data],
        }

        result = trace_evals_resource.get_results("te-123")

        assert isinstance(result, TraceEvaluationResultsResponse)
        assert len(result.results) == 1
        res = result.results[0]
        assert res.id == "result-123"
        assert res.score == 0.85
        assert res.passed is True
        assert res.reasoning == "The output meets quality standards"
        assert len(res.steps) == 2
        assert res.model == "claude-sonnet-4-20250514"
        assert res.latency_ms == 2500

    def test_get_results_request_parameters(self, trace_evals_resource, sample_result_data):
        """get_results makes correct API request."""
        trace_evals_resource._get.return_value = {"results": [sample_result_data]}

        trace_evals_resource.get_results("te-123")

        trace_evals_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/trace-evaluations/te-123/results",
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_results_empty(self, trace_evals_resource):
        """get_results returns empty results list."""
        trace_evals_resource._get.return_value = {"results": []}

        result = trace_evals_resource.get_results("te-123")

        assert isinstance(result, TraceEvaluationResultsResponse)
        assert len(result.results) == 0

    def test_get_results_none_response(self, trace_evals_resource):
        """get_results returns None when response is invalid."""
        trace_evals_resource._get.return_value = None

        result = trace_evals_resource.get_results("te-123")

        assert result is None

    def test_estimate_cost_success(self, trace_evals_resource):
        """estimate_cost returns CostEstimateResponse on success."""
        trace_evals_resource._post.return_value = CostEstimateResponse(
            estimated_cost=0.0045,
            input_tokens=1500,
            output_tokens=300,
            model="claude-sonnet-4-20250514",
            trace_count=5,
        )

        result = trace_evals_resource.estimate_cost(
            trace_ids=["t1", "t2", "t3", "t4", "t5"],
            judge_id="judge-789",
        )

        assert isinstance(result, CostEstimateResponse)
        assert result.estimated_cost == 0.0045
        assert result.trace_count == 5

    def test_estimate_cost_request_parameters(self, trace_evals_resource):
        """estimate_cost makes correct API request."""
        trace_evals_resource._post.return_value = CostEstimateResponse(
            estimated_cost=0.01, input_tokens=100, output_tokens=50, model="test", trace_count=2
        )

        trace_evals_resource.estimate_cost(
            trace_ids=["t1", "t2"],
            judge_id="judge-789",
        )

        trace_evals_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/trace-evaluations/estimate",
            body={"trace_ids": ["t1", "t2"], "judge_id": "judge-789"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=CostEstimateResponse,
        )

    def test_estimate_cost_none_response(self, trace_evals_resource):
        """estimate_cost returns None when response is invalid."""
        trace_evals_resource._post.return_value = None

        result = trace_evals_resource.estimate_cost(trace_ids=["t1"], judge_id="judge-789")

        assert result is None


class TestTraceEvaluationsErrorHandling:
    """Test error handling in TraceEvaluations resource."""

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
    def trace_evals_resource(self, mock_client):
        """TraceEvaluations resource instance."""
        return TraceEvaluations(mock_client)

    def test_create_handles_not_found(self, trace_evals_resource):
        """create method propagates not found errors."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        trace_evals_resource._post.side_effect = NotFoundError(
            "Trace or judge not found", response=mock_response, body=None
        )

        with pytest.raises(NotFoundError):
            trace_evals_resource.create(trace_id="bad-trace", judge_id="bad-judge")

    def test_get_handles_auth_error(self, trace_evals_resource):
        """get method propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        trace_evals_resource._get.side_effect = AuthenticationError("Unauthorized", response=mock_response, body=None)

        with pytest.raises(AuthenticationError):
            trace_evals_resource.get("te-123")

    def test_estimate_cost_handles_server_error(self, trace_evals_resource):
        """estimate_cost method propagates server errors."""
        from layerlens._exceptions import InternalServerError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        trace_evals_resource._post.side_effect = InternalServerError(
            "Internal server error", response=mock_response, body=None
        )

        with pytest.raises(InternalServerError):
            trace_evals_resource.estimate_cost(trace_ids=["t1"], judge_id="judge-789")


class TestTraceEvaluationsURLConstruction:
    """Test URL construction in TraceEvaluations resource."""

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
    def trace_evals_resource(self, mock_client):
        """TraceEvaluations resource instance."""
        return TraceEvaluations(mock_client)

    def test_base_url_construction(self, trace_evals_resource):
        """Base URL uses correct organization and project IDs."""
        assert trace_evals_resource._base_url() == "/organizations/custom-org/projects/custom-proj/trace-evaluations"

    def test_get_url_includes_evaluation_id(self, trace_evals_resource):
        """Get URL includes evaluation ID."""
        trace_evals_resource._get.return_value = None

        trace_evals_resource.get("te-abc")

        call_args = trace_evals_resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/trace-evaluations/te-abc"

    def test_results_url_includes_evaluation_id(self, trace_evals_resource):
        """Results URL includes evaluation ID."""
        trace_evals_resource._get.return_value = None

        trace_evals_resource.get_results("te-abc")

        call_args = trace_evals_resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/trace-evaluations/te-abc/results"

    def test_estimate_url(self, trace_evals_resource):
        """Estimate URL is constructed correctly."""
        trace_evals_resource._post.return_value = None

        trace_evals_resource.estimate_cost(trace_ids=["t1"], judge_id="j1")

        call_args = trace_evals_resource._post.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/trace-evaluations/estimate"
