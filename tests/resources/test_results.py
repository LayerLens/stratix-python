from datetime import timedelta
from unittest.mock import Mock

import httpx
import pytest

from atlas._models import Result, Results as ResultsData
from atlas._constants import DEFAULT_TIMEOUT
from atlas.resources.results.results import Results


class TestResults:
    """Test Results resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def results_resource(self, mock_client):
        """Results resource instance."""
        return Results(mock_client)

    @pytest.fixture
    def sample_result_data(self):
        """Sample result data for testing."""
        return {
            "subset": "mathematics",
            "prompt": "What is the derivative of x^2?",
            "result": "2x",
            "truth": "2x",
            "duration": timedelta(seconds=2.5),
            "score": 1.0,
            "metrics": {
                "accuracy": 1.0,
                "confidence": 0.95,
                "reasoning_quality": 0.9
            }
        }

    @pytest.fixture
    def mock_results_response(self, sample_result_data):
        """Mock ResultsData response."""
        result = Result(**sample_result_data)
        return ResultsData(results=[result])

    def test_results_initialization(self, mock_client):
        """Results resource initializes correctly."""
        results = Results(mock_client)
        
        assert results._client is mock_client
        assert results._get is mock_client.get_cast

    def test_get_results_success(self, results_resource, mock_results_response):
        """get method returns results successfully."""
        results_resource._get.return_value = mock_results_response
        
        result = results_resource.get(evaluation_id="eval-123")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Result)
        assert result[0].subset == "mathematics"
        assert result[0].prompt == "What is the derivative of x^2?"
        assert result[0].result == "2x"
        assert result[0].score == 1.0

    def test_get_results_request_parameters(self, results_resource, mock_results_response):
        """get method makes correct API request."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="eval-456")
        
        results_resource._get.assert_called_once_with(
            "/results",
            params={"evaluation_id": "eval-456"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=ResultsData,
        )

    def test_get_results_with_custom_timeout(self, results_resource, mock_results_response):
        """get method accepts custom timeout."""
        results_resource._get.return_value = mock_results_response
        custom_timeout = 120.0
        
        results_resource.get(evaluation_id="eval-123", timeout=custom_timeout)
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_get_results_with_httpx_timeout(self, results_resource, mock_results_response):
        """get method accepts httpx.Timeout object."""
        results_resource._get.return_value = mock_results_response
        custom_timeout = httpx.Timeout(120.0)
        
        results_resource.get(evaluation_id="eval-123", timeout=custom_timeout)
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    def test_get_results_none_response(self, results_resource):
        """get method returns None when response is None."""
        results_resource._get.return_value = None
        
        result = results_resource.get(evaluation_id="eval-123")
        
        assert result is None

    def test_get_results_invalid_response_type(self, results_resource):
        """get method handles non-ResultsData response gracefully."""
        results_resource._get.return_value = "invalid-response"
        
        result = results_resource.get(evaluation_id="eval-123")
        
        assert result is None

    def test_get_results_empty_response(self, results_resource):
        """get method returns empty list when no results in response."""
        empty_response = ResultsData(results=[])
        results_resource._get.return_value = empty_response
        
        result = results_resource.get(evaluation_id="eval-123")
        
        assert result == []
        assert isinstance(result, list)

    def test_get_results_multiple_items(self, results_resource, sample_result_data):
        """get method returns multiple results correctly."""
        result1 = Result(**sample_result_data)
        
        # Create second result with different data
        result2_data = sample_result_data.copy()
        result2_data["subset"] = "science"
        result2_data["prompt"] = "What is photosynthesis?"
        result2_data["result"] = "Process of converting light to energy"
        result2_data["truth"] = "Process of converting light to energy"
        result2_data["score"] = 0.95
        result2_data["duration"] = timedelta(seconds=3.2)
        result2 = Result(**result2_data)
        
        response = ResultsData(results=[result1, result2])
        results_resource._get.return_value = response
        
        result = results_resource.get(evaluation_id="eval-123")
        
        assert len(result) == 2
        assert result[0].subset == "mathematics"
        assert result[1].subset == "science"
        assert result[0].score == 1.0
        assert result[1].score == 0.95

    def test_get_results_url_construction(self, results_resource, mock_results_response):
        """get method uses correct URL endpoint."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="eval-123")
        
        call_args = results_resource._get.call_args
        assert call_args[0][0] == "/results"

    def test_get_results_evaluation_id_parameter(self, results_resource, mock_results_response):
        """get method correctly passes evaluation_id parameter."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="test-eval-789")
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["params"]["evaluation_id"] == "test-eval-789"

    def test_get_results_cast_to_parameter(self, results_resource, mock_results_response):
        """get method specifies correct cast_to parameter."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="eval-123")
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["cast_to"] is ResultsData

    def test_get_results_timeout_default(self, results_resource, mock_results_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="eval-123")
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_results_with_none_timeout(self, results_resource, mock_results_response):
        """get method accepts None timeout."""
        results_resource._get.return_value = mock_results_response
        
        results_resource.get(evaluation_id="eval-123", timeout=None)
        
        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_results_preserves_result_attributes(self, results_resource, mock_results_response):
        """get method preserves all result attributes correctly."""
        results_resource._get.return_value = mock_results_response
        
        result = results_resource.get(evaluation_id="eval-123")
        result_item = result[0]
        
        assert isinstance(result_item.duration, timedelta)
        assert result_item.duration.total_seconds() == 2.5
        assert isinstance(result_item.metrics, dict)
        assert result_item.metrics["accuracy"] == 1.0
        assert result_item.metrics["confidence"] == 0.95
        assert result_item.metrics["reasoning_quality"] == 0.9

    @pytest.mark.parametrize("evaluation_id", [
        "eval-123",
        "evaluation-456-abc",
        "test_eval_789",
        "long-evaluation-id-with-many-characters-123456789",
    ])
    def test_get_results_with_different_evaluation_ids(self, results_resource, mock_results_response, evaluation_id):
        """get method works with various evaluation ID formats."""
        results_resource._get.return_value = mock_results_response
        
        result = results_resource.get(evaluation_id=evaluation_id)
        
        assert isinstance(result, list)
        call_args = results_resource._get.call_args
        assert call_args.kwargs["params"]["evaluation_id"] == evaluation_id


class TestResultsErrorHandling:
    """Test error handling in Results resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def results_resource(self, mock_client):
        """Results resource instance."""
        return Results(mock_client)

    def test_get_results_handles_not_found_error(self, results_resource):
        """get method propagates not found errors."""
        from atlas._exceptions import NotFoundError
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        
        not_found_error = NotFoundError("Evaluation not found", response=mock_response, body=None)
        results_resource._get.side_effect = not_found_error
        
        with pytest.raises(NotFoundError):
            results_resource.get(evaluation_id="nonexistent-eval")

    def test_get_results_handles_auth_error(self, results_resource):
        """get method propagates authentication errors."""
        from atlas._exceptions import AuthenticationError
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}
        
        auth_error = AuthenticationError("Unauthorized", response=mock_response, body=None)
        results_resource._get.side_effect = auth_error
        
        with pytest.raises(AuthenticationError):
            results_resource.get(evaluation_id="eval-123")

    def test_get_results_handles_permission_error(self, results_resource):
        """get method propagates permission errors."""
        from atlas._exceptions import PermissionDeniedError
        
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}
        
        permission_error = PermissionDeniedError("Access denied", response=mock_response, body=None)
        results_resource._get.side_effect = permission_error
        
        with pytest.raises(PermissionDeniedError):
            results_resource.get(evaluation_id="restricted-eval")

    def test_get_results_handles_server_error(self, results_resource):
        """get method propagates server errors."""
        from atlas._exceptions import InternalServerError
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        
        server_error = InternalServerError("Internal server error", response=mock_response, body=None)
        results_resource._get.side_effect = server_error
        
        with pytest.raises(InternalServerError):
            results_resource.get(evaluation_id="eval-123")

    def test_get_results_handles_connection_error(self, results_resource):
        """get method propagates connection errors."""
        from atlas._exceptions import APIConnectionError
        
        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        results_resource._get.side_effect = connection_error
        
        with pytest.raises(APIConnectionError):
            results_resource.get(evaluation_id="eval-123")

    def test_get_results_handles_timeout_error(self, results_resource):
        """get method propagates timeout errors."""
        from atlas._exceptions import APITimeoutError
        
        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        results_resource._get.side_effect = timeout_error
        
        with pytest.raises(APITimeoutError):
            results_resource.get(evaluation_id="eval-123", timeout=1.0)


class TestResultsDataHandling:
    """Test data handling specifics in Results resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def results_resource(self, mock_client):
        """Results resource instance."""
        return Results(mock_client)

    def test_get_results_handles_complex_metrics(self, results_resource):
        """get method handles complex metrics structures."""
        complex_result_data = {
            "subset": "reasoning",
            "prompt": "Complex reasoning question",
            "result": "Complex answer",
            "truth": "Expected answer",
            "duration": timedelta(seconds=5.75),
            "score": 0.87,
            "metrics": {
                "accuracy": 0.87,
                "precision": 0.92,
                "recall": 0.83,
                "f1_score": 0.875,
                "perplexity": 12.34,
                "bleu_score": 0.78,
                "rouge_1": 0.85,
                "rouge_2": 0.72,
                "rouge_l": 0.80,
                "semantic_similarity": 0.91,
                "factual_correctness": 0.95,
                "reasoning_steps": 4.0
            }
        }
        
        complex_result = Result(**complex_result_data)
        response = ResultsData(results=[complex_result])
        results_resource._get.return_value = response
        
        result = results_resource.get(evaluation_id="eval-complex")
        
        assert len(result) == 1
        result_item = result[0]
        
        assert result_item.score == 0.87
        assert len(result_item.metrics) == 12
        assert result_item.metrics["f1_score"] == 0.875
        assert result_item.metrics["perplexity"] == 12.34
        assert result_item.metrics["reasoning_steps"] == 4.0

    def test_get_results_handles_different_durations(self, results_resource):
        """get method handles various duration formats."""
        durations_to_test = [
            timedelta(seconds=0.1),     # Very short
            timedelta(seconds=1.5),     # Normal
            timedelta(seconds=30.0),    # Long
            timedelta(minutes=2.5),     # Very long
            timedelta(hours=1),         # Extremely long
        ]
        
        results = []
        for i, duration in enumerate(durations_to_test):
            result_data = {
                "subset": f"test-{i}",
                "prompt": f"Test prompt {i}",
                "result": f"Test result {i}",
                "truth": f"Test truth {i}",
                "duration": duration,
                "score": 0.8 + i * 0.05,
                "metrics": {"accuracy": 0.8 + i * 0.05}
            }
            results.append(Result(**result_data))
        
        response = ResultsData(results=results)
        results_resource._get.return_value = response
        
        result = results_resource.get(evaluation_id="eval-durations")
        
        assert len(result) == 5
        assert result[0].duration == timedelta(seconds=0.1)
        assert result[1].duration == timedelta(seconds=1.5)
        assert result[2].duration == timedelta(seconds=30.0)
        assert result[3].duration == timedelta(minutes=2.5)
        assert result[4].duration == timedelta(hours=1)

    def test_get_results_handles_empty_metrics(self, results_resource):
        """get method handles results with empty metrics."""
        result_data = {
            "subset": "minimal",
            "prompt": "Minimal test",
            "result": "Minimal result",
            "truth": "Minimal truth",
            "duration": timedelta(seconds=1.0),
            "score": 0.5,
            "metrics": {}  # Empty metrics
        }
        
        minimal_result = Result(**result_data)
        response = ResultsData(results=[minimal_result])
        results_resource._get.return_value = response
        
        result = results_resource.get(evaluation_id="eval-minimal")
        
        assert len(result) == 1
        assert result[0].metrics == {}
        assert isinstance(result[0].metrics, dict)

    def test_get_results_return_type_consistency(self, results_resource):
        """get method returns consistent types."""
        # Test that the method returns either a list or None
        results_resource._get.return_value = None
        result = results_resource.get(evaluation_id="eval-123")
        assert result is None
        
        # Test that it returns a list when successful
        results_resource._get.return_value = ResultsData(results=[])
        result = results_resource.get(evaluation_id="eval-123")
        assert isinstance(result, list)