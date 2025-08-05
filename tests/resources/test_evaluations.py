from unittest.mock import Mock

import httpx
import pytest

from atlas._models import Evaluation, Evaluations as EvaluationsData
from atlas._constants import DEFAULT_TIMEOUT
from atlas.resources.evaluations.evaluations import Evaluations


class TestEvaluations:
    """Test Evaluations resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def evaluations_resource(self, mock_client):
        """Evaluations resource instance."""
        return Evaluations(mock_client)

    @pytest.fixture
    def sample_evaluation_data(self):
        """Sample evaluation data for testing."""
        return {
            "id": "eval-123",
            "status": "completed",
            "status_description": "Evaluation completed successfully",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "model_name": "GPT-4",
            "model_key": "gpt-4",
            "model_company": "OpenAI",
            "dataset_id": "dataset-789",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "readability_score": 0.85,
            "toxicity_score": 0.02,
            "ethics_score": 0.92,
            "accuracy": 0.89,
        }

    @pytest.fixture
    def mock_evaluations_response(self, sample_evaluation_data):
        """Mock EvaluationsData response."""
        evaluation = Evaluation(**sample_evaluation_data)
        return EvaluationsData(data=[evaluation])

    def test_evaluations_initialization(self, mock_client):
        """Evaluations resource initializes correctly."""
        evaluations = Evaluations(mock_client)
        
        assert evaluations._client is mock_client
        assert evaluations._get is mock_client.get_cast
        assert evaluations._post is mock_client.post_cast

    def test_create_evaluation_success(self, evaluations_resource, mock_evaluations_response):
        """create method returns first evaluation on success."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        result = evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        assert isinstance(result, Evaluation)
        assert result.id == "eval-123"
        assert result.model_name == "GPT-4"
        assert result.dataset_name == "MMLU"

    def test_create_evaluation_request_parameters(self, evaluations_resource, mock_evaluations_response):
        """create method makes correct API request."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        evaluations_resource._post.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/evaluations",
            body=[{
                "model_id": "gpt-4",
                "dataset_id": "mmlu",
                "is_custom_model": False,
                "is_custom_dataset": False,
            }],
            timeout=DEFAULT_TIMEOUT,
            cast_to=EvaluationsData,
        )

    def test_create_evaluation_with_custom_timeout(self, evaluations_resource, mock_evaluations_response):
        """create method accepts custom timeout."""
        evaluations_resource._post.return_value = mock_evaluations_response
        custom_timeout = 30.0
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu", timeout=custom_timeout)
        
        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_create_evaluation_with_httpx_timeout(self, evaluations_resource, mock_evaluations_response):
        """create method accepts httpx.Timeout object."""
        evaluations_resource._post.return_value = mock_evaluations_response
        custom_timeout = httpx.Timeout(30.0)
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu", timeout=custom_timeout)
        
        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    def test_create_evaluation_empty_response(self, evaluations_resource):
        """create method returns None when no evaluations in response."""
        empty_response = EvaluationsData(data=[])
        evaluations_resource._post.return_value = empty_response
        
        result = evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        assert result is None

    def test_create_evaluation_none_response(self, evaluations_resource):
        """create method returns None when response is None."""
        evaluations_resource._post.return_value = None
        
        result = evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        assert result is None

    def test_create_evaluation_invalid_response_type(self, evaluations_resource):
        """create method handles non-EvaluationsData response gracefully."""
        evaluations_resource._post.return_value = "invalid-response"
        
        result = evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        assert result is None

    def test_create_evaluation_multiple_evaluations_returns_first(self, evaluations_resource, sample_evaluation_data):
        """create method returns first evaluation when multiple exist."""
        eval1 = Evaluation(**sample_evaluation_data)
        eval2_data = sample_evaluation_data.copy()
        eval2_data["id"] = "eval-456"
        eval2 = Evaluation(**eval2_data)
        
        response = EvaluationsData(data=[eval1, eval2])
        evaluations_resource._post.return_value = response
        
        result = evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        assert result.id == "eval-123"  # First evaluation
        assert result is not eval2

    def test_create_evaluation_url_construction(self, evaluations_resource, mock_evaluations_response):
        """create method constructs URL correctly with org and project IDs."""
        evaluations_resource._client.organization_id = "custom-org"
        evaluations_resource._client.project_id = "custom-project"
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="test-model", benchmark="test-benchmark")
        
        expected_url = "/organizations/custom-org/projects/custom-project/evaluations"
        call_args = evaluations_resource._post.call_args
        assert call_args[0][0] == expected_url

    def test_create_evaluation_request_body_structure(self, evaluations_resource, mock_evaluations_response):
        """create method sends correct request body structure."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="custom-model", benchmark="custom-benchmark")
        
        call_args = evaluations_resource._post.call_args
        body = call_args.kwargs["body"]
        
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["model_id"] == "custom-model"
        assert body[0]["dataset_id"] == "custom-benchmark"
        assert body[0]["is_custom_model"] is False
        assert body[0]["is_custom_dataset"] is False

    @pytest.mark.parametrize("model_name,benchmark_name", [
        ("gpt-3.5-turbo", "hellaswag"),
        ("claude-3-opus", "arc-challenge"),
        ("llama-2-70b", "truthfulqa"),
        ("custom-model-123", "custom-benchmark-456"),
    ])
    def test_create_evaluation_with_different_parameters(self, evaluations_resource, mock_evaluations_response, model_name, benchmark_name):
        """create method works with various model and benchmark combinations."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        result = evaluations_resource.create(model=model_name, benchmark=benchmark_name)
        
        assert isinstance(result, Evaluation)
        call_args = evaluations_resource._post.call_args
        body = call_args.kwargs["body"][0]
        assert body["model_id"] == model_name
        assert body["dataset_id"] == benchmark_name

    def test_create_evaluation_cast_to_parameter(self, evaluations_resource, mock_evaluations_response):
        """create method specifies correct cast_to parameter."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["cast_to"] is EvaluationsData

    def test_create_evaluation_timeout_default(self, evaluations_resource, mock_evaluations_response):
        """create method uses DEFAULT_TIMEOUT when no timeout specified."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu")
        
        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_create_evaluation_with_none_timeout(self, evaluations_resource, mock_evaluations_response):
        """create method accepts None timeout."""
        evaluations_resource._post.return_value = mock_evaluations_response
        
        evaluations_resource.create(model="gpt-4", benchmark="mmlu", timeout=None)
        
        call_args = evaluations_resource._post.call_args
        assert call_args.kwargs["timeout"] is None


class TestEvaluationsErrorHandling:
    """Test error handling in Evaluations resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
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
        from atlas._exceptions import APIStatusError
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}
        
        api_error = APIStatusError("Bad Request", response=mock_response, body=None)
        evaluations_resource._post.side_effect = api_error
        
        with pytest.raises(APIStatusError):
            evaluations_resource.create(model="invalid-model", benchmark="invalid-benchmark")

    def test_create_evaluation_handles_connection_error(self, evaluations_resource):
        """create method propagates connection errors."""
        from atlas._exceptions import APIConnectionError
        
        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        evaluations_resource._post.side_effect = connection_error
        
        with pytest.raises(APIConnectionError):
            evaluations_resource.create(model="gpt-4", benchmark="mmlu")

    def test_create_evaluation_handles_timeout_error(self, evaluations_resource):
        """create method propagates timeout errors."""
        from atlas._exceptions import APITimeoutError
        
        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        evaluations_resource._post.side_effect = timeout_error
        
        with pytest.raises(APITimeoutError):
            evaluations_resource.create(model="gpt-4", benchmark="mmlu", timeout=1.0)


class TestEvaluationsResourceIntegration:
    """Integration-style tests for Evaluations resource."""

    def test_create_evaluation_end_to_end_flow(self):
        """Test complete evaluation creation flow."""
        # Mock the full chain: client -> resource -> API call -> response
        mock_client = Mock()
        mock_client.organization_id = "test-org"
        mock_client.project_id = "test-project"
        
        # Create sample evaluation data
        evaluation_data = {
            "id": "eval-integration-test",
            "status": "submitted",
            "status_description": "Evaluation submitted",
            "submitted_at": 1640995200,
            "finished_at": 0,
            "model_id": "integration-model",
            "model_name": "Integration Test Model",
            "model_key": "integration-model",
            "model_company": "TestCorp",
            "dataset_id": "integration-dataset",
            "dataset_name": "Integration Test Dataset",
            "average_duration": 0,
            "readability_score": 0.0,
            "toxicity_score": 0.0,
            "ethics_score": 0.0,
            "accuracy": 0.0,
        }
        
        evaluation = Evaluation(**evaluation_data)
        response = EvaluationsData(data=[evaluation])
        mock_client.post_cast.return_value = response
        
        # Test the resource
        evaluations_resource = Evaluations(mock_client)
        result = evaluations_resource.create(
            model="integration-model",
            benchmark="integration-dataset"
        )
        
        # Verify the complete flow
        assert result is not None
        assert result.id == "eval-integration-test"
        assert result.model_id == "integration-model"
        assert result.dataset_id == "integration-dataset"
        assert result.status == "submitted"
        
        # Verify the API call was made correctly
        mock_client.post_cast.assert_called_once()
        call_args = mock_client.post_cast.call_args
        assert "/organizations/test-org/projects/test-project/evaluations" in call_args[0][0]
        assert call_args.kwargs["body"][0]["model_id"] == "integration-model"
        assert call_args.kwargs["body"][0]["dataset_id"] == "integration-dataset"