from datetime import timedelta
from unittest.mock import Mock

import httpx
import pytest

from atlas.models import Result, Pagination, ResultMetrics, ResultsResponse
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
            "metrics": {"accuracy": 1.0, "confidence": 0.95, "reasoning_quality": 0.9},
        }

    @pytest.fixture
    def mock_results_response(self, sample_result_data):
        """Mock raw API response with pagination."""
        return {
            "evaluation_id": "eval-123",
            "results": [sample_result_data],
            "metrics": {
                "total_count": 1,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }

    def test_results_initialization(self, mock_client):
        """Results resource initializes correctly."""
        results = Results(mock_client)

        assert results._client is mock_client
        assert results._get is mock_client.get_cast

    def test_get_results_success(self, results_resource, mock_results_response):
        """get method returns ResultsResponse successfully."""
        results_resource._get.return_value = mock_results_response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert result.evaluation_id == "eval-123"
        assert len(result.results) == 1
        assert isinstance(result.results[0], Result)
        assert result.results[0].subset == "mathematics"
        assert result.results[0].prompt == "What is the derivative of x^2?"
        assert result.results[0].result == "2x"
        assert result.results[0].score == 1.0
        assert isinstance(result.metrics, ResultMetrics)
        assert isinstance(result.pagination, Pagination)
        assert result.pagination.total_count == 1
        assert result.pagination.page_size == 100
        assert result.pagination.total_pages == 1

    def test_get_results_request_parameters(self, results_resource, mock_results_response):
        """get method makes correct API request."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="eval-456")

        results_resource._get.assert_called_once_with(
            "/results",
            params={"evaluation_id": "eval-456", "page": "1", "pageSize": "100"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_results_with_custom_timeout(self, results_resource, mock_results_response):
        """get method accepts custom timeout."""
        results_resource._get.return_value = mock_results_response
        custom_timeout = 120.0

        results_resource.get_by_id(evaluation_id="eval-123", timeout=custom_timeout)

        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_get_results_with_httpx_timeout(self, results_resource, mock_results_response):
        """get method accepts httpx.Timeout object."""
        results_resource._get.return_value = mock_results_response
        custom_timeout = httpx.Timeout(120.0)

        results_resource.get_by_id(evaluation_id="eval-123", timeout=custom_timeout)

        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    def test_get_results_none_response(self, results_resource):
        """get method returns None when response is None."""
        results_resource._get.return_value = None

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert result is None

    def test_get_results_invalid_response_type(self, results_resource):
        """get method handles non-ResultsResponse response gracefully."""
        results_resource._get.return_value = "invalid-response"

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert result is None

    def test_get_results_empty_response(self, results_resource):
        """get method returns ResultsResponse with empty results list when no results in response."""
        empty_response = {
            "evaluation_id": "eval-123",
            "results": [],
            "metrics": {
                "total_count": 0,
                "min_toxicity_score": None,
                "max_toxicity_score": None,
                "min_readability_score": None,
                "max_readability_score": None,
            },
        }
        results_resource._get.return_value = empty_response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert result.evaluation_id == "eval-123"
        assert result.results == []
        assert isinstance(result.results, list)
        assert result.pagination.total_count == 0

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

        response = {
            "evaluation_id": "eval-123",
            "results": [sample_result_data, result2_data],
            "metrics": {
                "total_count": 2,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert len(result.results) == 2
        assert result.results[0].subset == "mathematics"
        assert result.results[1].subset == "science"
        assert result.results[0].score == 1.0
        assert result.results[1].score == 0.95
        assert result.pagination.total_count == 2

    def test_get_results_url_construction(self, results_resource, mock_results_response):
        """get method uses correct URL endpoint."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="eval-123")

        call_args = results_resource._get.call_args
        assert call_args[0][0] == "/results"

    def test_get_results_evaluation_id_parameter(self, results_resource, mock_results_response):
        """get method correctly passes evaluation_id parameter."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="test-eval-789")

        call_args = results_resource._get.call_args
        assert call_args.kwargs["params"]["evaluation_id"] == "test-eval-789"

    def test_get_results_cast_to_parameter(self, results_resource, mock_results_response):
        """get method specifies correct cast_to parameter."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="eval-123")

        call_args = results_resource._get.call_args
        assert call_args.kwargs["cast_to"] is dict

    def test_get_results_timeout_default(self, results_resource, mock_results_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="eval-123")

        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_results_with_none_timeout(self, results_resource, mock_results_response):
        """get method accepts None timeout."""
        results_resource._get.return_value = mock_results_response

        results_resource.get_by_id(evaluation_id="eval-123", timeout=None)

        call_args = results_resource._get.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_results_preserves_result_attributes(self, results_resource, mock_results_response):
        """get method preserves all result attributes correctly."""
        results_resource._get.return_value = mock_results_response

        result = results_resource.get_by_id(evaluation_id="eval-123")
        result_item = result.results[0]

        assert isinstance(result_item.duration, timedelta)
        assert result_item.duration.total_seconds() == 2.5
        assert isinstance(result_item.metrics, dict)
        assert result_item.metrics["accuracy"] == 1.0
        assert result_item.metrics["confidence"] == 0.95
        assert result_item.metrics["reasoning_quality"] == 0.9

    @pytest.mark.parametrize(
        "evaluation_id",
        [
            "eval-123",
            "evaluation-456-abc",
            "test_eval_789",
            "long-evaluation-id-with-many-characters-123456789",
        ],
    )
    def test_get_results_with_different_evaluation_ids(self, results_resource, mock_results_response, evaluation_id):
        """get method works with various evaluation ID formats."""
        results_resource._get.return_value = mock_results_response

        result = results_resource.get_by_id(evaluation_id=evaluation_id)

        assert isinstance(result, ResultsResponse)
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
            results_resource.get_by_id(evaluation_id="nonexistent-eval")

    def test_get_results_handles_auth_error(self, results_resource):
        """get method propagates authentication errors."""
        from atlas._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        auth_error = AuthenticationError("Unauthorized", response=mock_response, body=None)
        results_resource._get.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_permission_error(self, results_resource):
        """get method propagates permission errors."""
        from atlas._exceptions import PermissionDeniedError

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        permission_error = PermissionDeniedError("Access denied", response=mock_response, body=None)
        results_resource._get.side_effect = permission_error

        with pytest.raises(PermissionDeniedError):
            results_resource.get_by_id(evaluation_id="restricted-eval")

    def test_get_results_handles_server_error(self, results_resource):
        """get method propagates server errors."""
        from atlas._exceptions import InternalServerError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        server_error = InternalServerError("Internal server error", response=mock_response, body=None)
        results_resource._get.side_effect = server_error

        with pytest.raises(InternalServerError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_connection_error(self, results_resource):
        """get method propagates connection errors."""
        from atlas._exceptions import APIConnectionError

        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        results_resource._get.side_effect = connection_error

        with pytest.raises(APIConnectionError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_timeout_error(self, results_resource):
        """get method propagates timeout errors."""
        from atlas._exceptions import APITimeoutError

        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        results_resource._get.side_effect = timeout_error

        with pytest.raises(APITimeoutError):
            results_resource.get_by_id(evaluation_id="eval-123", timeout=1.0)


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
                "reasoning_steps": 4.0,
            },
        }

        response = {
            "evaluation_id": "eval-complex",
            "results": [complex_result_data],
            "metrics": {
                "total_count": 1,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        result = results_resource.get_by_id(evaluation_id="eval-complex")

        assert isinstance(result, ResultsResponse)
        assert len(result.results) == 1
        result_item = result.results[0]

        assert result_item.score == 0.87
        assert len(result_item.metrics) == 12
        assert result_item.metrics["f1_score"] == 0.875
        assert result_item.metrics["perplexity"] == 12.34
        assert result_item.metrics["reasoning_steps"] == 4.0

    def test_get_results_handles_different_durations(self, results_resource):
        """get method handles various duration formats."""
        durations_to_test = [
            timedelta(seconds=0.1),  # Very short
            timedelta(seconds=1.5),  # Normal
            timedelta(seconds=30.0),  # Long
            timedelta(minutes=2.5),  # Very long
            timedelta(hours=1),  # Extremely long
        ]

        results_data = []
        for i, duration in enumerate(durations_to_test):
            result_data = {
                "subset": f"test-{i}",
                "prompt": f"Test prompt {i}",
                "result": f"Test result {i}",
                "truth": f"Test truth {i}",
                "duration": duration,
                "score": 0.8 + i * 0.05,
                "metrics": {"accuracy": 0.8 + i * 0.05},
            }
            results_data.append(result_data)

        response = {
            "evaluation_id": "eval-durations",
            "results": results_data,
            "metrics": {
                "total_count": 5,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        result = results_resource.get_by_id(evaluation_id="eval-durations")

        assert isinstance(result, ResultsResponse)
        assert len(result.results) == 5
        assert result.results[0].duration == timedelta(seconds=0.1)
        assert result.results[1].duration == timedelta(seconds=1.5)
        assert result.results[2].duration == timedelta(seconds=30.0)
        assert result.results[3].duration == timedelta(minutes=2.5)
        assert result.results[4].duration == timedelta(hours=1)

    def test_get_results_handles_empty_metrics(self, results_resource):
        """get method handles results with empty metrics."""
        result_data = {
            "subset": "minimal",
            "prompt": "Minimal test",
            "result": "Minimal result",
            "truth": "Minimal truth",
            "duration": timedelta(seconds=1.0),
            "score": 0.5,
            "metrics": {},  # Empty metrics
        }

        response = {
            "evaluation_id": "eval-minimal",
            "results": [result_data],
            "metrics": {
                "total_count": 1,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        result = results_resource.get_by_id(evaluation_id="eval-minimal")

        assert isinstance(result, ResultsResponse)
        assert len(result.results) == 1
        assert result.results[0].metrics == {}
        assert isinstance(result.results[0].metrics, dict)

    def test_get_results_return_type_consistency(self, results_resource):
        """get method returns consistent types."""
        # Test that the method returns either a ResultsResponse object or None
        results_resource._get.return_value = None
        result = results_resource.get_by_id(evaluation_id="eval-123")
        assert result is None

        # Test that it returns a ResultsResponse object when successful
        empty_response = {
            "evaluation_id": "eval-123",
            "results": [],
            "metrics": {
                "total_count": 0,
                "min_toxicity_score": None,
                "max_toxicity_score": None,
                "min_readability_score": None,
                "max_readability_score": None,
            },
        }
        results_resource._get.return_value = empty_response
        result = results_resource.get_by_id(evaluation_id="eval-123")
        assert isinstance(result, ResultsResponse)


class TestResultsPagination:
    """Test pagination functionality in Results resource."""

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
            "metrics": {"accuracy": 1.0, "confidence": 0.95},
        }

    def test_get_results_with_pagination_parameters(self, results_resource, sample_result_data):
        """get method accepts pagination parameters."""
        mock_response = {
            "evaluation_id": "eval-paginated",
            "results": [sample_result_data],
            "metrics": {
                "total_count": 250,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.5,
                "min_readability_score": 0.3,
                "max_readability_score": 0.95,
            },
        }
        results_resource._get.return_value = mock_response

        result_data = results_resource.get_by_id(
            evaluation_id="eval-paginated",
            page=2,
            page_size=50,
        )

        # Verify the call was made with correct parameters
        results_resource._get.assert_called_once_with(
            "/results",
            params={
                "evaluation_id": "eval-paginated",
                "page": "2",
                "pageSize": "50",
            },
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

        # Verify the response structure
        assert isinstance(result_data, ResultsResponse)
        assert result_data.evaluation_id == "eval-paginated"
        assert result_data.pagination.total_count == 250
        assert result_data.pagination.page_size == 50
        assert result_data.pagination.total_pages == 5  # ceil(250 / 50) = 5

    def test_get_results_pagination_parameter_conversion(self, results_resource, sample_result_data):
        """get method converts pagination parameters to strings."""
        mock_response = {
            "evaluation_id": "eval-123",
            "results": [sample_result_data],
            "metrics": {
                "total_count": 100,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = mock_response

        results_resource.get_by_id(evaluation_id="eval-123", page=3, page_size=25)

        call_args = results_resource._get.call_args
        params = call_args.kwargs["params"]

        # Verify parameters are converted to strings
        assert params["page"] == "3"
        assert params["pageSize"] == "25"
        assert isinstance(params["page"], str)
        assert isinstance(params["pageSize"], str)

    def test_get_results_default_page_parameter(self, results_resource, sample_result_data):
        """get method defaults to page 1 when no page is specified."""
        mock_response = {
            "evaluation_id": "eval-123",
            "results": [sample_result_data],
            "metrics": {
                "total_count": 100,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = mock_response

        results_resource.get_by_id(evaluation_id="eval-123")

        call_args = results_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["pageSize"] == "100"  # pageSize is now always included with default value

    def test_get_results_pagination_metadata_calculation(self, results_resource, sample_result_data):
        """get method correctly calculates pagination metadata."""
        # Mock API response without pagination
        api_response = {
            "evaluation_id": "eval-math",
            "results": [sample_result_data],
            "metrics": {
                "total_count": 487,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.15,
                "min_readability_score": 0.75,
                "max_readability_score": 0.98,
            },
        }
        results_resource._get.return_value = api_response

        result = results_resource.get_by_id(evaluation_id="eval-math", page=3, page_size=50)

        # Should have calculated pagination correctly
        assert isinstance(result, ResultsResponse)
        assert result.pagination.total_count == 487
        assert result.pagination.page_size == 50
        assert result.pagination.total_pages == 10  # ceil(487 / 50) = 10

    @pytest.mark.parametrize(
        "total_count,page_size,expected_pages",
        [
            (100, 50, 2),
            (99, 50, 2),
            (101, 50, 3),
            (1000, 100, 10),
            (999, 100, 10),
            (1001, 100, 11),
            (1, 100, 1),
            (0, 100, 0),
            (250, 25, 10),
            (251, 25, 11),
        ],
    )
    def test_pagination_total_pages_calculation(
        self,
        results_resource,
        sample_result_data,
        total_count,
        page_size,
        expected_pages,
    ):
        """get method correctly calculates total_pages for various scenarios."""
        api_response = {
            "evaluation_id": "eval-calc",
            "results": [sample_result_data] if total_count > 0 else [],
            "metrics": {
                "total_count": total_count,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = api_response

        result = results_resource.get_by_id(evaluation_id="eval-calc", page_size=page_size)

        assert result.pagination.total_count == total_count
        assert result.pagination.page_size == page_size
        assert result.pagination.total_pages == expected_pages


class TestResultsPaginationErrorHandling:
    """Test error handling and edge cases for pagination."""

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

    def test_get_results_invalid_api_response(self, results_resource):
        """get method handles invalid API response structure."""
        # Response missing metrics
        invalid_response = {
            "evaluation_id": "eval-123",
            "results": [],
            # Missing metrics
        }
        results_resource._get.return_value = invalid_response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        # Should return None when response structure is invalid
        assert result is None

    def test_get_results_with_zero_total_count_in_metrics(self, results_resource):
        """get method handles zero total_count in metrics."""
        invalid_response = {
            "evaluation_id": "eval-123",
            "results": [],
            "metrics": {
                "total_count": 0,  # Now included, testing graceful handling
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": None,
                "max_readability_score": None,
            },
        }
        results_resource._get.return_value = invalid_response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        # Should handle zero total_count gracefully
        assert isinstance(result, ResultsResponse)
        assert result.pagination.total_count == 0
        assert result.pagination.total_pages == 0

    def test_get_results_non_dict_response(self, results_resource):
        """get method handles non-dict API response."""
        results_resource._get.return_value = "invalid-string-response"

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert result is None

    def test_get_results_pydantic_validation_error(self, results_resource):
        """get method handles Pydantic validation errors."""
        # Response with invalid data types
        invalid_response = {
            "evaluation_id": "eval-123",
            "results": "not-a-list",  # Should be a list
            "metrics": {
                "total_count": 100,
            },
        }
        results_resource._get.return_value = invalid_response

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert result is None

    def test_get_results_extreme_pagination_values(self, results_resource):
        """get method handles extreme pagination values."""
        extreme_response = {
            "evaluation_id": "eval-extreme",
            "results": [],
            "metrics": {
                "total_count": 999999,  # Very large number
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = extreme_response

        result = results_resource.get_by_id(evaluation_id="eval-extreme", page_size=1)

        assert isinstance(result, ResultsResponse)
        assert result.pagination.total_count == 999999
        assert result.pagination.page_size == 1
        assert result.pagination.total_pages == 999999  # ceil(999999 / 1)

    def test_get_results_zero_page_size_edge_case(self, results_resource):
        """get method handles zero page_size (should use default)."""
        response = {
            "evaluation_id": "eval-123",
            "results": [],
            "metrics": {
                "total_count": 100,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        # Pass 0 as page_size
        result = results_resource.get_by_id(evaluation_id="eval-123", page_size=0)

        assert isinstance(result, ResultsResponse)
        # page_size of 0 should be corrected to minimum value of 1
        assert result.pagination.page_size == 1

    def test_get_results_negative_page_values(self, results_resource):
        """get method handles negative page values."""
        response = {
            "evaluation_id": "eval-123",
            "results": [],
            "metrics": {
                "total_count": 100,
                "min_toxicity_score": 0.0,
                "max_toxicity_score": 0.1,
                "min_readability_score": 0.8,
                "max_readability_score": 0.9,
            },
        }
        results_resource._get.return_value = response

        # Test with negative page and page_size
        result = results_resource.get_by_id(evaluation_id="eval-123", page=-1, page_size=-50)

        # Should still make the API call and process response
        call_args = results_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "-1"
        assert params["pageSize"] == "1"  # negative page_size should be corrected to 1

        assert isinstance(result, ResultsResponse)
        assert result.pagination.page_size == 1  # negative page_size should be corrected to 1
        # total_pages calculation with corrected page_size
        assert result.pagination.total_pages == 100  # math.ceil(100/1) = 100
