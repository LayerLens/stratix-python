from unittest.mock import Mock

import httpx
import pytest

from layerlens.models import (
    Evaluation,
    ErrorAnalysis,
    AnalysisSummary,
    EvaluationMetric,
    EvaluationStatus,
    EvaluationDataset,
    EvaluationSummary,
    EvaluationTaskType,
    PerformanceDetails,
    EvaluationsResponse,
    CreateEvaluationsResponse,
)
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.evaluations.evaluations import Evaluations
from layerlens.resources.public_evaluations.public_evaluations import (
    PublicEvaluationsResource,
)


class TestEvaluations:
    """Test Evaluations resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def mock_benchmark(self):
        """Mock benchmark."""
        benchmark = Mock()
        benchmark.id = "benchmark-789"
        benchmark.key = "mmlu"
        benchmark.name = "MMLU"
        return benchmark

    @pytest.fixture
    def mock_model(self):
        """Mock model."""
        model = Mock()
        model.id = "model-123"
        model.key = "gpt-4"
        model.name = "GPT-4"
        return model

    @pytest.fixture
    def evaluations_resource(self, mock_client):
        """Evaluations resource instance."""
        return Evaluations(mock_client)

    @pytest.fixture
    def sample_evaluation_data(self):
        """Sample evaluation data for testing."""
        return {
            "id": "eval-123",
            "status": "success",
            "status_description": "Evaluation completed successfully",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "dataset_id": "benchmark-789",
            "average_duration": 2500,
            "accuracy": 0.89,
        }

    @pytest.fixture
    def mock_evaluations_response(self, sample_evaluation_data):
        """Mock CreateEvaluationsResponse response."""
        evaluation = Evaluation(**sample_evaluation_data)
        return CreateEvaluationsResponse(data=[evaluation])

    def test_evaluations_initialization(self, mock_client):
        """Evaluations resource initializes correctly."""
        evaluations = Evaluations(mock_client)

        assert evaluations._client is mock_client
        assert evaluations._get is mock_client.get_cast
        assert evaluations._post is mock_client.post_cast

    def test_create_evaluation_success(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method returns first evaluation on success."""
        evaluations_resource._post.return_value = mock_evaluations_response

        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        assert isinstance(result, Evaluation)
        assert result.id == "eval-123"
        assert result.model_id == "model-456"
        assert result.benchmark_id == "benchmark-789"

    def test_create_evaluation_request_parameters(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method makes correct API request."""
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        evaluations_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/evaluations",
            body=[
                {
                    "model_id": "model-123",
                    "dataset_id": "benchmark-789",
                    "is_custom_model": False,
                    "is_custom_dataset": False,
                }
            ],
            timeout=DEFAULT_TIMEOUT,
            cast_to=CreateEvaluationsResponse,
        )

    def test_create_evaluation_with_custom_timeout(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method accepts custom timeout."""
        evaluations_resource._post.return_value = mock_evaluations_response
        custom_timeout = 30.0

        evaluations_resource.create(
            model=mock_model,
            benchmark=mock_benchmark,
            timeout=custom_timeout,
        )

        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_create_evaluation_with_httpx_timeout(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method accepts httpx.Timeout object."""
        evaluations_resource._post.return_value = mock_evaluations_response
        custom_timeout = httpx.Timeout(30.0)

        evaluations_resource.create(
            model=mock_model,
            benchmark=mock_benchmark,
            timeout=custom_timeout,
        )

        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    def test_create_evaluation_empty_response(self, mock_model, mock_benchmark, evaluations_resource):
        """create method returns None when no evaluations in response."""
        empty_response = CreateEvaluationsResponse(data=[])
        evaluations_resource._post.return_value = empty_response

        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        assert result is None

    def test_create_evaluation_none_response(self, mock_model, mock_benchmark, evaluations_resource):
        """create method returns None when response is None."""
        evaluations_resource._post.return_value = None

        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        assert result is None

    def test_create_evaluation_invalid_response_type(self, mock_model, mock_benchmark, evaluations_resource):
        """create method handles non-CreateEvaluationsResponse response gracefully."""
        evaluations_resource._post.return_value = "invalid-response"

        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        assert result is None

    def test_create_evaluation_multiple_evaluations_returns_first(
        self, mock_model, mock_benchmark, evaluations_resource, sample_evaluation_data
    ):
        """create method returns first evaluation when multiple exist."""
        eval1 = Evaluation(**sample_evaluation_data)
        eval2_data = sample_evaluation_data.copy()
        eval2_data["id"] = "eval-456"
        eval2 = Evaluation(**eval2_data)

        response = CreateEvaluationsResponse(data=[eval1, eval2])
        evaluations_resource._post.return_value = response

        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        assert result.id == "eval-123"  # First evaluation
        assert result is not eval2

    def test_create_evaluation_url_construction(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method constructs URL correctly with org and project IDs."""
        evaluations_resource._client.organization_id = "custom-org"
        evaluations_resource._client.project_id = "custom-project"
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        expected_url = "/organizations/custom-org/projects/custom-project/evaluations"
        call_args = evaluations_resource._post.call_args
        assert call_args[0][0] == expected_url

    def test_create_evaluation_request_body_structure(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method sends correct request body structure."""
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        call_args = evaluations_resource._post.call_args
        body = call_args.kwargs["body"]

        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["model_id"] == mock_model.id
        assert body[0]["dataset_id"] == mock_benchmark.id
        assert body[0]["is_custom_model"] is False
        assert body[0]["is_custom_dataset"] is False

    def test_create_evaluation_cast_to_parameter(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method specifies correct cast_to parameter."""
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["cast_to"] is CreateEvaluationsResponse

    def test_create_evaluation_timeout_default(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method uses DEFAULT_TIMEOUT when no timeout specified."""
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_create_evaluation_with_none_timeout(
        self,
        mock_model,
        mock_benchmark,
        evaluations_resource,
        mock_evaluations_response,
    ):
        """create method accepts None timeout."""
        evaluations_resource._post.return_value = mock_evaluations_response

        evaluations_resource.create(model=mock_model, benchmark=mock_benchmark, timeout=None)

        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_all_returns_evaluations(self, evaluations_resource, mock_client, sample_evaluation_data):
        """get_all returns list of evaluations when response is valid."""
        evaluation = Evaluation(**sample_evaluation_data)
        response = {"evaluations": [evaluation], "total_count": 1}
        evaluations_resource._get.return_value = response

        result = evaluations_resource.get_many()

        assert isinstance(result, EvaluationsResponse)
        assert result.evaluations[0].id == "eval-123"
        evaluations_resource._get.assert_called_once_with(
            "/evaluations",
            params={
                "organizationID": mock_client.organization_id,
                "projectID": mock_client.project_id,
                "page": "1",
                "pageSize": "100",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )
        assert result.evaluations[0]._client is mock_client

    def test_get_all_returns_none_on_invalid_response(self, evaluations_resource):
        """get_all returns None when response is invalid type."""
        evaluations_resource._get.return_value = "not-a-response"

        result = evaluations_resource.get_many()

        assert result is None


class TestEvaluationsErrorHandling:
    """Test error handling in Evaluations resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def evaluations_resource(self, mock_client):
        """Evaluations resource instance."""
        return Evaluations(mock_client)

    def test_create_evaluation_handles_api_error(self, evaluations_resource):
        """create method propagates API errors."""
        from layerlens._exceptions import APIStatusError

        mock_model = Mock()
        mock_model.id = "invalid-model"

        mock_benchmark = Mock()
        mock_benchmark.id = "invalid-benchmark"

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        api_error = APIStatusError("Bad Request", response=mock_response, body=None)
        evaluations_resource._post.side_effect = api_error

        with pytest.raises(APIStatusError):
            evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

    def test_create_evaluation_handles_connection_error(self, evaluations_resource):
        """create method propagates connection errors."""
        from layerlens._exceptions import APIConnectionError

        mock_model = Mock()
        mock_model.id = "invalid-model"

        mock_benchmark = Mock()
        mock_benchmark.id = "invalid-benchmark"

        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        evaluations_resource._post.side_effect = connection_error

        with pytest.raises(APIConnectionError):
            evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

    def test_create_evaluation_handles_timeout_error(self, evaluations_resource):
        """create method propagates timeout errors."""
        from layerlens._exceptions import APITimeoutError

        mock_model = Mock()
        mock_model.id = "invalid-model"

        mock_benchmark = Mock()
        mock_benchmark.id = "invalid-benchmark"

        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        evaluations_resource._post.side_effect = timeout_error

        with pytest.raises(APITimeoutError):
            evaluations_resource.create(model=mock_model, benchmark=mock_benchmark, timeout=1.0)


class TestEvaluationsResourceIntegration:
    """Integration-style tests for Evaluations resource."""

    def test_create_evaluation_end_to_end_flow(self):
        """Test complete evaluation creation flow."""
        # Mock the full chain: client -> resource -> API call -> response
        mock_client = Mock()
        mock_client.organization_id = "test-org"
        mock_client.project_id = "test-project"

        mock_benchmark = Mock()
        mock_benchmark.id = "benchmark-789"
        mock_benchmark.key = "mmlu"
        mock_benchmark.name = "MMLU"

        mock_model = Mock()
        mock_model.id = "model-123"
        mock_model.key = "gpt-4"
        mock_model.name = "GPT-4"

        # Create sample evaluation data
        evaluation_data = {
            "id": "eval-integration-test",
            "status": "in-progress",
            "status_description": "Evaluation submitted",
            "submitted_at": 1640995200,
            "finished_at": 0,
            "model_id": mock_model.id,
            "dataset_id": mock_benchmark.id,
            "average_duration": 0,
            "accuracy": 0.0,
        }

        evaluation = Evaluation(**evaluation_data)
        response = CreateEvaluationsResponse(data=[evaluation])
        mock_client.post_cast.return_value = response

        # Test the resource
        evaluations_resource = Evaluations(mock_client)
        result = evaluations_resource.create(model=mock_model, benchmark=mock_benchmark)

        # Verify the complete flow
        assert result is not None
        assert result.id == "eval-integration-test"
        assert result.model_id == mock_model.id
        assert result.benchmark_id == mock_benchmark.id
        assert result.status == EvaluationStatus.IN_PROGRESS

        # Verify the API call was made correctly
        mock_client.post_cast.assert_called_once()
        call_args = mock_client.post_cast.call_args
        assert "/organizations/test-org/projects/test-project/evaluations" in call_args[0][0]
        assert call_args.kwargs["body"][0]["model_id"] == mock_model.id
        assert call_args.kwargs["body"][0]["dataset_id"] == mock_benchmark.id


class TestEvaluationModelFields:
    """Test Evaluation model parses all backend fields."""

    @pytest.fixture
    def full_evaluation_data(self):
        return {
            "id": "eval-full",
            "status": "success",
            "status_description": "Evaluation completed successfully",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "model_name": "GPT-4",
            "model_key": "gpt-4",
            "model_company": "OpenAI",
            "dataset_id": "benchmark-789",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "accuracy": 0.89,
            "readability_score": 0.75,
            "toxicity_score": 0.02,
            "ethics_score": 0.95,
            "failed_prompt_count": 3,
            "queue_id": 42,
            "summary": {
                "name": "GPT-4 on MMLU",
                "goal": "Evaluate general knowledge",
                "metrics": [
                    {"name": "accuracy", "description": "Correctness of responses"},
                    {"name": "toxicity", "description": "Harmful content detection"},
                ],
                "task_types": [
                    {"name": "multiple_choice", "description": "Select correct answer"},
                ],
                "dataset": {
                    "total_size": 15908,
                    "training_size": 0,
                    "test_size": 15908,
                    "characteristics": ["multi-domain", "multiple-choice"],
                },
                "model": {
                    "model_name": "GPT-4",
                    "performance": {"overall": 0.89},
                },
                "performance_details": {
                    "strengths": ["Strong reasoning", "Good factual recall"],
                    "challenges": ["Abstract math", "Ambiguous questions"],
                },
                "error_analysis": {
                    "common_failure_modes": ["Off-by-one errors", "Misinterpreting negation"],
                    "example": "Q: Which is NOT true? A: Selected a true statement.",
                },
                "analysis_summary": {
                    "key_takeaways": [
                        "Strong overall performance at 89%",
                        "Struggles with negation-based questions",
                    ],
                },
            },
        }

    def test_parse_all_fields(self, full_evaluation_data):
        """Evaluation model parses all backend fields correctly."""
        evaluation = Evaluation(**full_evaluation_data)

        assert evaluation.id == "eval-full"
        assert evaluation.status == EvaluationStatus.SUCCESS
        assert evaluation.status_description == "Evaluation completed successfully"
        assert evaluation.model_id == "model-456"
        assert evaluation.model_name == "GPT-4"
        assert evaluation.model_key == "gpt-4"
        assert evaluation.model_company == "OpenAI"
        assert evaluation.benchmark_id == "benchmark-789"
        assert evaluation.benchmark_name == "MMLU"
        assert evaluation.accuracy == 0.89
        assert evaluation.readability_score == 0.75
        assert evaluation.toxicity_score == 0.02
        assert evaluation.ethics_score == 0.95
        assert evaluation.failed_prompt_count == 3
        assert evaluation.queue_id == 42

    def test_parse_summary(self, full_evaluation_data):
        """Evaluation model parses nested summary correctly."""
        evaluation = Evaluation(**full_evaluation_data)

        assert evaluation.summary is not None
        summary = evaluation.summary
        assert isinstance(summary, EvaluationSummary)
        assert summary.name == "GPT-4 on MMLU"
        assert summary.goal == "Evaluate general knowledge"

    def test_parse_summary_metrics(self, full_evaluation_data):
        """Summary metrics are parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        metrics = evaluation.summary.metrics

        assert len(metrics) == 2
        assert isinstance(metrics[0], EvaluationMetric)
        assert metrics[0].name == "accuracy"
        assert metrics[1].name == "toxicity"

    def test_parse_summary_task_types(self, full_evaluation_data):
        """Summary task types are parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        task_types = evaluation.summary.task_types

        assert len(task_types) == 1
        assert isinstance(task_types[0], EvaluationTaskType)
        assert task_types[0].name == "multiple_choice"

    def test_parse_summary_dataset(self, full_evaluation_data):
        """Summary dataset info is parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        dataset = evaluation.summary.dataset

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.total_size == 15908
        assert dataset.test_size == 15908
        assert "multi-domain" in dataset.characteristics

    def test_parse_summary_performance_details(self, full_evaluation_data):
        """Summary performance details are parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        perf = evaluation.summary.performance_details

        assert isinstance(perf, PerformanceDetails)
        assert len(perf.strengths) == 2
        assert "Strong reasoning" in perf.strengths
        assert len(perf.challenges) == 2

    def test_parse_summary_error_analysis(self, full_evaluation_data):
        """Summary error analysis is parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        errors = evaluation.summary.error_analysis

        assert isinstance(errors, ErrorAnalysis)
        assert len(errors.common_failure_modes) == 2
        assert "Off-by-one errors" in errors.common_failure_modes
        assert "NOT true" in errors.example

    def test_parse_summary_analysis_summary(self, full_evaluation_data):
        """Summary analysis summary is parsed correctly."""
        evaluation = Evaluation(**full_evaluation_data)
        analysis = evaluation.summary.analysis_summary

        assert isinstance(analysis, AnalysisSummary)
        assert len(analysis.key_takeaways) == 2

    def test_missing_optional_fields_default(self):
        """Evaluation model uses defaults for missing optional fields."""
        minimal_data = {
            "id": "eval-min",
            "status": "pending",
            "submitted_at": 1640995200,
            "finished_at": 0,
            "model_id": "m1",
            "dataset_id": "b1",
            "average_duration": 0,
            "accuracy": 0.0,
        }
        evaluation = Evaluation(**minimal_data)

        assert evaluation.status_description == ""
        assert evaluation.model_name == ""
        assert evaluation.model_key == ""
        assert evaluation.model_company == ""
        assert evaluation.benchmark_name == ""
        assert evaluation.readability_score == 0.0
        assert evaluation.toxicity_score == 0.0
        assert evaluation.ethics_score == 0.0
        assert evaluation.failed_prompt_count == 0
        assert evaluation.queue_id == 0
        assert evaluation.summary is None

    def test_null_summary_field(self):
        """Evaluation model handles null summary."""
        data = {
            "id": "eval-no-summary",
            "status": "in-progress",
            "submitted_at": 1640995200,
            "finished_at": 0,
            "model_id": "m1",
            "dataset_id": "b1",
            "average_duration": 0,
            "accuracy": 0.0,
            "summary": None,
        }
        evaluation = Evaluation(**data)
        assert evaluation.summary is None

    def test_get_by_id_returns_full_evaluation(self):
        """get_by_id returns Evaluation with all fields populated."""
        mock_client = Mock()
        mock_client.organization_id = "org-123"
        mock_client.project_id = "proj-456"
        mock_client.get_cast = Mock()

        full_eval = Evaluation(
            id="eval-123",
            status=EvaluationStatus.SUCCESS,
            status_description="Done",
            submitted_at=1640995200,
            finished_at=1640995800,
            model_id="m1",
            model_name="GPT-4",
            model_key="gpt-4",
            model_company="OpenAI",
            dataset_id="b1",
            dataset_name="MMLU",
            average_duration=2500,
            accuracy=0.89,
            readability_score=0.75,
            toxicity_score=0.02,
            ethics_score=0.95,
            failed_prompt_count=3,
            queue_id=42,
            summary=EvaluationSummary(
                name="Test",
                goal="Evaluate",
                metrics=[EvaluationMetric(name="accuracy", description="Correctness")],
            ),
        )
        mock_client.get_cast.return_value = full_eval

        evaluations = Evaluations(mock_client)
        result = evaluations.get_by_id("eval-123")

        assert result is not None
        assert result.model_name == "GPT-4"
        assert result.benchmark_name == "MMLU"
        assert result.readability_score == 0.75
        assert result.summary is not None
        assert result.summary.goal == "Evaluate"


class TestPublicEvaluationsResource:
    """Test PublicEvaluationsResource for the public client."""

    @pytest.fixture
    def mock_public_client(self):
        """Mock PublicClient."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def public_evaluations(self, mock_public_client):
        """PublicEvaluationsResource instance."""
        return PublicEvaluationsResource(mock_public_client)

    @pytest.fixture
    def sample_evaluation_data(self):
        return {
            "id": "eval-pub-123",
            "status": "success",
            "status_description": "Done",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "model_name": "GPT-4",
            "dataset_id": "benchmark-789",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "accuracy": 0.89,
            "summary": {
                "name": "GPT-4 on MMLU",
                "goal": "Evaluate general knowledge",
                "metrics": [{"name": "accuracy", "description": "Correctness"}],
            },
        }

    def test_get_by_id_success(self, public_evaluations, sample_evaluation_data):
        """get_by_id returns Evaluation on success."""
        evaluation = Evaluation(**sample_evaluation_data)
        public_evaluations._get.return_value = evaluation

        result = public_evaluations.get_by_id("eval-pub-123")

        assert isinstance(result, Evaluation)
        assert result.id == "eval-pub-123"
        assert result.model_name == "GPT-4"
        assert result.summary is not None
        assert result.summary.name == "GPT-4 on MMLU"

    def test_get_by_id_correct_url(self, public_evaluations, sample_evaluation_data):
        """get_by_id calls correct endpoint."""
        evaluation = Evaluation(**sample_evaluation_data)
        public_evaluations._get.return_value = evaluation

        public_evaluations.get_by_id("eval-pub-123")

        public_evaluations._get.assert_called_once_with(
            "/evaluations/eval-pub-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=Evaluation,
        )

    def test_get_by_id_returns_none_on_invalid(self, public_evaluations):
        """get_by_id returns None when response is not Evaluation."""
        public_evaluations._get.return_value = None

        result = public_evaluations.get_by_id("nonexistent")

        assert result is None

    def test_get_by_id_no_client_attached(self, public_evaluations, sample_evaluation_data):
        """get_by_id does not attach client (public client has no org/project)."""
        evaluation = Evaluation(**sample_evaluation_data)
        public_evaluations._get.return_value = evaluation

        result = public_evaluations.get_by_id("eval-pub-123")

        assert result._client is None

    def test_get_many_success(self, public_evaluations, sample_evaluation_data):
        """get_many returns EvaluationsResponse with evaluations."""
        resp = {
            "evaluations": [sample_evaluation_data],
            "total_count": 1,
        }
        public_evaluations._get.return_value = resp

        result = public_evaluations.get_many()

        assert isinstance(result, EvaluationsResponse)
        assert len(result.evaluations) == 1
        assert result.evaluations[0].id == "eval-pub-123"

    def test_get_many_with_filters(self, public_evaluations, sample_evaluation_data):
        """get_many passes filter parameters correctly."""
        resp = {"evaluations": [sample_evaluation_data], "total_count": 1}
        public_evaluations._get.return_value = resp

        public_evaluations.get_many(
            page=2,
            page_size=50,
            sort_by="accuracy",
            order="desc",
            model_ids=["m1", "m2"],
            benchmark_ids=["b1"],
            status=EvaluationStatus.SUCCESS,
        )

        call_args = public_evaluations._get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["page"] == "2"
        assert params["pageSize"] == "50"
        assert params["sortBy"] == "accuracy"
        assert params["order"] == "desc"
        assert params["models"] == "m1,m2"
        assert params["datasets"] == "b1"
        assert params["status"] == "success"
        assert "organizationID" not in params
        assert "projectID" not in params

    def test_get_many_pagination(self, public_evaluations, sample_evaluation_data):
        """get_many computes pagination correctly."""
        resp = {"evaluations": [sample_evaluation_data] * 3, "total_count": 25}
        public_evaluations._get.return_value = resp

        result = public_evaluations.get_many(
            page=1,
            page_size=10,
        )

        assert result.pagination.page == 1
        assert result.pagination.page_size == 10
        assert result.pagination.total_count == 25
        assert result.pagination.total_pages == 3  # ceil(25/10)

    def test_get_many_returns_none_on_invalid(self, public_evaluations):
        """get_many returns None when response is invalid."""
        public_evaluations._get.return_value = "not-a-dict"

        result = public_evaluations.get_many()

        assert result is None

    def test_get_many_empty_results(self, public_evaluations):
        """get_many handles empty evaluations list."""
        resp = {"evaluations": [], "total_count": 0}
        public_evaluations._get.return_value = resp

        result = public_evaluations.get_many()

        assert isinstance(result, EvaluationsResponse)
        assert len(result.evaluations) == 0
        assert result.pagination.total_count == 0
