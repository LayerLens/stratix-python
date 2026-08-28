"""Results resource tests.

Every fixture here crosses the wire: bodies are handed to ``httpx.Response(...,
json=...)``, which serializes them, and the resource parses the bytes back. That
is deliberate. The previous version of this file fed a live ``timedelta`` object
straight into ``Result(**data)`` and asserted ``duration.total_seconds() == 2.5``
— an assertion true of the fixture and false of every production response, which
is why a 10^9x unit error on ``duration`` survived (LAY-3765). ``duration`` is an
int64 nanosecond count on the wire; write it that way.
"""

from datetime import timedelta
from unittest.mock import Mock

import httpx
import pytest

from layerlens._wire import json_object
from layerlens.models import Result, Pagination, ResultMetrics, ResultsResponse
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens._exceptions import StratixError, APIResponseValidationError
from layerlens.resources.results.results import Results

REQUEST = httpx.Request("GET", "https://api.test.invalid/api/v1/results")


def wire(body, status_code: int = 200) -> httpx.Response:
    """A real HTTP response carrying `body` as JSON bytes."""
    return httpx.Response(status_code, json=body, request=REQUEST)


def raw(content: bytes, status_code: int = 200) -> httpx.Response:
    """A real HTTP response carrying arbitrary bytes."""
    return httpx.Response(
        status_code,
        content=content,
        headers={"Content-Type": "application/json"},
        request=REQUEST,
    )


class TestResults:
    """Test Results resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def results_resource(self, mock_client):
        """Results resource instance."""
        return Results(mock_client)

    @pytest.fixture
    def sample_result_data(self):
        """Sample result row, shaped as models.LLMResult marshals it."""
        return {
            "subset": "mathematics",
            "prompt": "What is the derivative of x^2?",
            "result": "2x",
            "truth": "2x",
            "duration": 2_500_000_000,  # 2.5s as int64 nanoseconds
            "score": 1.0,
            "metrics": {"accuracy": 1.0, "confidence": 0.95, "reasoning_quality": 0.9},
            "input_tokens": 512,
            "output_tokens": 128,
        }

    @pytest.fixture
    def mock_results_response(self, sample_result_data):
        """Raw API response body."""
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
        results_resource._get.return_value = wire(mock_results_response)

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
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="eval-456")

        results_resource._get.assert_called_once_with(
            "/results",
            params={"evaluation_id": "eval-456", "page": "1", "page_size": "100"},
            timeout=DEFAULT_TIMEOUT,
        )

    def test_get_results_requests_the_raw_response(self, results_resource, mock_results_response):
        """No cast_to is passed: the resource needs the response itself.

        Pagination has to be derived from the body before the envelope can be
        validated, and holding the httpx.Response is what lets a schema failure
        raise APIResponseValidationError with the payload attached instead of being
        swallowed into None.
        """
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="eval-123")

        assert "cast_to" not in results_resource._get.call_args.kwargs

    def test_get_results_with_custom_timeout(self, results_resource, mock_results_response):
        """get method accepts custom timeout."""
        results_resource._get.return_value = wire(mock_results_response)
        custom_timeout = 120.0

        results_resource.get_by_id(evaluation_id="eval-123", timeout=custom_timeout)

        assert results_resource._get.call_args.kwargs["timeout"] == custom_timeout

    def test_get_results_with_httpx_timeout(self, results_resource, mock_results_response):
        """get method accepts httpx.Timeout object."""
        results_resource._get.return_value = wire(mock_results_response)
        custom_timeout = httpx.Timeout(120.0)

        results_resource.get_by_id(evaluation_id="eval-123", timeout=custom_timeout)

        assert results_resource._get.call_args.kwargs["timeout"] is custom_timeout

    def test_get_results_empty_response(self, results_resource):
        """An evaluation with no rows yields an empty results list, not an error."""
        results_resource._get.return_value = wire(
            {
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
        )

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert result.evaluation_id == "eval-123"
        assert result.results == []
        assert result.pagination.total_count == 0

    def test_get_results_null_results_reads_as_empty(self, results_resource):
        """The deployed API sends `"results": null` for a page with no rows.

        `Results []LLMResult` has no omitempty and the SQL repositories leave the
        slice nil when nothing matches, so nil marshals to null. Read it as "no
        rows" rather than rejecting the page.
        """
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": None, "metrics": {"total_count": 0}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert result.results == []

    def test_get_results_multiple_items(self, results_resource, sample_result_data):
        """get method returns multiple results correctly."""
        second = dict(
            sample_result_data,
            subset="science",
            prompt="What is photosynthesis?",
            result="Process of converting light to energy",
            truth="Process of converting light to energy",
            score=0.95,
            duration=3_200_000_000,
        )

        results_resource._get.return_value = wire(
            {
                "evaluation_id": "eval-123",
                "results": [sample_result_data, second],
                "metrics": {"total_count": 2},
            }
        )

        result = results_resource.get_by_id(evaluation_id="eval-123")

        assert isinstance(result, ResultsResponse)
        assert len(result.results) == 2
        assert result.results[0].subset == "mathematics"
        assert result.results[1].subset == "science"
        assert result.results[0].score == 1.0
        assert result.results[1].score == 0.95
        assert result.results[1].duration == timedelta(seconds=3.2)
        assert result.pagination.total_count == 2

    def test_get_results_url_construction(self, results_resource, mock_results_response):
        """get method uses correct URL endpoint."""
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="eval-123")

        assert results_resource._get.call_args[0][0] == "/results"

    def test_get_results_evaluation_id_parameter(self, results_resource, mock_results_response):
        """get method correctly passes evaluation_id parameter."""
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="test-eval-789")

        assert results_resource._get.call_args.kwargs["params"]["evaluation_id"] == "test-eval-789"

    def test_get_results_timeout_default(self, results_resource, mock_results_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="eval-123")

        assert results_resource._get.call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_results_with_none_timeout(self, results_resource, mock_results_response):
        """get method accepts None timeout."""
        results_resource._get.return_value = wire(mock_results_response)

        results_resource.get_by_id(evaluation_id="eval-123", timeout=None)

        assert results_resource._get.call_args.kwargs["timeout"] is None

    def test_get_results_preserves_result_attributes(self, results_resource, mock_results_response):
        """get method preserves all result attributes correctly."""
        results_resource._get.return_value = wire(mock_results_response)

        result_item = results_resource.get_by_id(evaluation_id="eval-123").results[0]

        assert isinstance(result_item.duration, timedelta)
        # 2_500_000_000 nanoseconds on the wire is 2.5 seconds, not 2.5 billion.
        assert result_item.duration.total_seconds() == 2.5
        assert isinstance(result_item.metrics, dict)
        assert result_item.metrics["accuracy"] == 1.0
        assert result_item.metrics["confidence"] == 0.95
        assert result_item.metrics["reasoning_quality"] == 0.9
        assert result_item.input_tokens == 512
        assert result_item.output_tokens == 128

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
        results_resource._get.return_value = wire(mock_results_response)

        result = results_resource.get_by_id(evaluation_id=evaluation_id)

        assert isinstance(result, ResultsResponse)
        assert results_resource._get.call_args.kwargs["params"]["evaluation_id"] == evaluation_id


class TestResultsErrorHandling:
    """Test error handling in Results resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
        client = Mock()
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def results_resource(self, mock_client):
        """Results resource instance."""
        return Results(mock_client)

    def test_get_results_handles_not_found_error(self, results_resource):
        """get method propagates not found errors."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        results_resource._get.side_effect = NotFoundError("Evaluation not found", response=mock_response, body=None)

        with pytest.raises(NotFoundError):
            results_resource.get_by_id(evaluation_id="nonexistent-eval")

    def test_get_results_handles_auth_error(self, results_resource):
        """get method propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        results_resource._get.side_effect = AuthenticationError("Unauthorized", response=mock_response, body=None)

        with pytest.raises(AuthenticationError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_permission_error(self, results_resource):
        """get method propagates permission errors."""
        from layerlens._exceptions import PermissionDeniedError

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        results_resource._get.side_effect = PermissionDeniedError("Access denied", response=mock_response, body=None)

        with pytest.raises(PermissionDeniedError):
            results_resource.get_by_id(evaluation_id="restricted-eval")

    def test_get_results_handles_server_error(self, results_resource):
        """get method propagates server errors."""
        from layerlens._exceptions import InternalServerError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        results_resource._get.side_effect = InternalServerError(
            "Internal server error", response=mock_response, body=None
        )

        with pytest.raises(InternalServerError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_connection_error(self, results_resource):
        """get method propagates connection errors."""
        from layerlens._exceptions import APIConnectionError

        results_resource._get.side_effect = APIConnectionError(request=Mock())

        with pytest.raises(APIConnectionError):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_get_results_handles_timeout_error(self, results_resource):
        """get method propagates timeout errors."""
        from layerlens._exceptions import APITimeoutError

        results_resource._get.side_effect = APITimeoutError(Mock())

        with pytest.raises(APITimeoutError):
            results_resource.get_by_id(evaluation_id="eval-123", timeout=1.0)


class TestResultsSchemaFailures:
    """A 2xx body that does not match the schema must raise, never return None.

    All four cases below used to hit `except Exception: return None`, and
    `get_all_by_id` then read that None as end-of-pages — turning a schema failure
    into a silently truncated list (LAY-2772).
    """

    @pytest.fixture
    def results_resource(self):
        client = Mock()
        client.get_cast = Mock()
        return Results(client)

    def test_missing_metrics_raises(self, results_resource):
        """metrics is required by ResultsResponse; its absence is a contract break."""
        results_resource._get.return_value = wire({"evaluation_id": "eval-123", "results": []})

        with pytest.raises(APIResponseValidationError, match="metrics"):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_results_not_a_list_raises(self, results_resource):
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": "not-a-list", "metrics": {"total_count": 100}}
        )

        with pytest.raises(APIResponseValidationError, match="results"):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_body_that_is_not_an_object_raises(self, results_resource):
        """A JSON scalar or array where an object is documented."""
        results_resource._get.return_value = wire("invalid-response")

        with pytest.raises(APIResponseValidationError, match="not a JSON object"):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_body_that_is_not_json_raises(self, results_resource):
        """A 2xx whose body is not JSON at all — e.g. an HTML error page from a proxy."""
        results_resource._get.return_value = raw(b"<html>502 Bad Gateway</html>")

        with pytest.raises(APIResponseValidationError, match="not valid JSON"):
            results_resource.get_by_id(evaluation_id="eval-123")

    def test_raised_error_carries_the_offending_body(self, results_resource):
        """So a customer can report the drift rather than describe it."""
        body = {"evaluation_id": "eval-123", "results": "not-a-list", "metrics": {"total_count": 1}}
        results_resource._get.return_value = wire(body)

        with pytest.raises(APIResponseValidationError) as caught:
            results_resource.get_by_id(evaluation_id="eval-123")

        # The payload that actually failed validation: the server's body plus the
        # client-derived `pagination`. Carrying exactly what was validated is what
        # makes the error reproducible.
        assert caught.value.body is not None
        assert caught.value.body["results"] == "not-a-list"
        assert caught.value.body["evaluation_id"] == body["evaluation_id"]
        assert caught.value.body["pagination"]["page"] == 1
        assert isinstance(caught.value, StratixError), "must be catchable as a layerlens error"

    def test_json_object_helper_rejects_a_json_array(self):
        """Direct unit cover for the decode helper's own contract."""
        with pytest.raises(APIResponseValidationError, match="list, not a JSON object"):
            json_object(wire([1, 2, 3]), endpoint="/results")


class TestResultsDataHandling:
    """Test data handling specifics in Results resource."""

    @pytest.fixture
    def results_resource(self):
        client = Mock()
        client.get_cast = Mock()
        return Results(client)

    def test_get_results_handles_complex_metrics(self, results_resource):
        """get method handles many built-in metric keys."""
        complex_result_data = {
            "subset": "reasoning",
            "prompt": "Complex reasoning question",
            "result": "Complex answer",
            "truth": "Expected answer",
            "duration": 5_750_000_000,
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

        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-complex", "results": [complex_result_data], "metrics": {"total_count": 1}}
        )

        result_item = results_resource.get_by_id(evaluation_id="eval-complex").results[0]

        assert result_item.score == 0.87
        assert len(result_item.metrics) == 12
        assert result_item.metrics["f1_score"] == 0.875
        assert result_item.metrics["perplexity"] == 12.34
        assert result_item.metrics["reasoning_steps"] == 4.0
        assert result_item.duration == timedelta(seconds=5.75)

    def test_get_results_handles_different_durations(self, results_resource):
        """Durations across magnitudes, written as the wire writes them."""
        nanoseconds_to_expected = [
            (100_000_000, timedelta(seconds=0.1)),
            (1_500_000_000, timedelta(seconds=1.5)),
            (30_000_000_000, timedelta(seconds=30.0)),
            (150_000_000_000, timedelta(minutes=2.5)),
            (3_600_000_000_000, timedelta(hours=1)),
        ]

        results_data = [
            {
                "subset": f"test-{i}",
                "prompt": f"Test prompt {i}",
                "result": f"Test result {i}",
                "truth": f"Test truth {i}",
                "duration": nanoseconds,
                "score": 0.8 + i * 0.05,
                "metrics": {"accuracy": 0.8 + i * 0.05},
            }
            for i, (nanoseconds, _) in enumerate(nanoseconds_to_expected)
        ]

        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-durations", "results": results_data, "metrics": {"total_count": 5}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-durations")

        assert len(result.results) == 5
        for parsed, (_, expected) in zip(result.results, nanoseconds_to_expected):
            assert parsed.duration == expected

    def test_get_results_handles_empty_metrics(self, results_resource):
        """get method handles results with empty metrics."""
        results_resource._get.return_value = wire(
            {
                "evaluation_id": "eval-minimal",
                "results": [
                    {
                        "subset": "minimal",
                        "prompt": "Minimal test",
                        "result": "Minimal result",
                        "truth": "Minimal truth",
                        "duration": 1_000_000_000,
                        "score": 0.5,
                        "metrics": {},
                    }
                ],
                "metrics": {"total_count": 1},
            }
        )

        result = results_resource.get_by_id(evaluation_id="eval-minimal")

        assert len(result.results) == 1
        assert result.results[0].metrics == {}


class TestResultsPagination:
    """Test pagination functionality in Results resource."""

    @pytest.fixture
    def results_resource(self):
        client = Mock()
        client.get_cast = Mock()
        return Results(client)

    @pytest.fixture
    def sample_result_data(self):
        return {
            "subset": "mathematics",
            "prompt": "What is the derivative of x^2?",
            "result": "2x",
            "truth": "2x",
            "duration": 2_500_000_000,
            "score": 1.0,
            "metrics": {"accuracy": 1.0, "confidence": 0.95},
        }

    def test_get_results_with_pagination_parameters(self, results_resource, sample_result_data):
        """get method accepts pagination parameters."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-paginated", "results": [sample_result_data], "metrics": {"total_count": 250}}
        )

        result_data = results_resource.get_by_id(evaluation_id="eval-paginated", page=2, page_size=50)

        results_resource._get.assert_called_once_with(
            "/results",
            params={"evaluation_id": "eval-paginated", "page": "2", "page_size": "50"},
            timeout=DEFAULT_TIMEOUT,
        )

        assert isinstance(result_data, ResultsResponse)
        assert result_data.evaluation_id == "eval-paginated"
        assert result_data.pagination.total_count == 250
        assert result_data.pagination.page_size == 50
        assert result_data.pagination.total_pages == 5  # ceil(250 / 50)

    def test_get_results_pagination_parameter_conversion(self, results_resource, sample_result_data):
        """get method converts pagination parameters to strings."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": [sample_result_data], "metrics": {"total_count": 100}}
        )

        results_resource.get_by_id(evaluation_id="eval-123", page=3, page_size=25)

        params = results_resource._get.call_args.kwargs["params"]
        assert params["page"] == "3"
        assert params["page_size"] == "25"
        assert isinstance(params["page"], str)
        assert isinstance(params["page_size"], str)

    def test_get_results_default_page_parameter(self, results_resource, sample_result_data):
        """get method defaults to page 1 when no page is specified."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": [sample_result_data], "metrics": {"total_count": 100}}
        )

        results_resource.get_by_id(evaluation_id="eval-123")

        params = results_resource._get.call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["page_size"] == "100"

    def test_get_results_pagination_metadata_calculation(self, results_resource, sample_result_data):
        """get method correctly calculates pagination metadata."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-math", "results": [sample_result_data], "metrics": {"total_count": 487}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-math", page=3, page_size=50)

        assert result.pagination.total_count == 487
        assert result.pagination.page_size == 50
        assert result.pagination.total_pages == 10  # ceil(487 / 50)

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
        results_resource._get.return_value = wire(
            {
                "evaluation_id": "eval-calc",
                "results": [sample_result_data] if total_count > 0 else [],
                "metrics": {"total_count": total_count},
            }
        )

        result = results_resource.get_by_id(evaluation_id="eval-calc", page_size=page_size)

        assert result.pagination.total_count == total_count
        assert result.pagination.page_size == page_size
        assert result.pagination.total_pages == expected_pages

    def test_get_results_extreme_pagination_values(self, results_resource):
        """get method handles extreme pagination values."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-extreme", "results": [], "metrics": {"total_count": 999999}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-extreme", page_size=1)

        assert result.pagination.total_count == 999999
        assert result.pagination.page_size == 1
        assert result.pagination.total_pages == 999999

    def test_get_results_zero_page_size_edge_case(self, results_resource):
        """page_size of 0 is corrected to the minimum of 1."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": [], "metrics": {"total_count": 100}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-123", page_size=0)

        assert result.pagination.page_size == 1

    def test_get_results_negative_page_values(self, results_resource):
        """get method handles negative page values."""
        results_resource._get.return_value = wire(
            {"evaluation_id": "eval-123", "results": [], "metrics": {"total_count": 100}}
        )

        result = results_resource.get_by_id(evaluation_id="eval-123", page=-1, page_size=-50)

        params = results_resource._get.call_args.kwargs["params"]
        assert params["page"] == "-1"
        assert params["page_size"] == "1"  # negative page_size corrected to 1

        assert result.pagination.page_size == 1
        assert result.pagination.total_pages == 100  # ceil(100/1)
