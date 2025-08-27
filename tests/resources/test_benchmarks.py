from unittest.mock import Mock, call

import httpx
import pytest

from layerlens.models import (
    Benchmark,
    CustomBenchmark,
    PublicBenchmark,
    BenchmarksResponse,
)
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.benchmarks.benchmarks import Benchmarks


class TestBenchmarks:
    """Test Benchmarks resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        """Benchmarks resource instance."""
        return Benchmarks(mock_client)

    @pytest.fixture
    def sample_benchmark_data(self):
        """Sample benchmark data for testing."""
        return {
            "id": "benchmark-123",
            "key": "mmlu",
            "name": "MMLU",
            "full_description": "Massive Multitask Language Understanding",
            "language": "english",
            "categories": ["reasoning", "knowledge"],
            "subsets": ["math", "science", "history"],
            "prompt_count": 15908,
            "deprecated": False,
        }

    @pytest.fixture
    def sample_custom_benchmark_data(self):
        """Sample custom benchmark data for testing."""
        return {
            "id": "custom-benchmark-456",
            "key": "my-benchmark",
            "name": "My Custom Benchmark",
            "description": "Custom benchmark description",
            "system_prompt": "You are a helpful assistant",
            "subsets": ["subset1", "subset2"],
            "prompt_count": 100,
            "version_count": 1,
            "regex_pattern": r"Answer: (.+)",
            "llm_judge_model_id": "gpt-4",
            "custom_instructions": "Rate responses on scale 1-10",
            "scoring_metric": "accuracy",
            "metrics": ["accuracy", "precision"],
            "files": ["data.jsonl"],
            "disabled": False,
        }

    @pytest.fixture
    def mock_benchmarks_response(self, sample_benchmark_data, sample_custom_benchmark_data):
        """Mock BenchmarksResponse response with public benchmarks."""
        public_benchmark = Benchmark(**sample_benchmark_data)
        custom_benchmark = CustomBenchmark(**sample_custom_benchmark_data)

        return BenchmarksResponse(data=BenchmarksResponse.Data(datasets=[public_benchmark, custom_benchmark]))

    def test_benchmarks_initialization(self, mock_client):
        """Benchmarks resource initializes correctly."""
        benchmarks = Benchmarks(mock_client)

        assert benchmarks._client is mock_client
        assert benchmarks._get is mock_client.get_cast

    def test_get_benchmarks_success(self, benchmarks_resource, mock_benchmarks_response):
        """get method returns benchmarks successfully."""
        benchmarks_resource._get.side_effect = lambda *_, **kwargs: (
            mock_benchmarks_response
            if kwargs.get("params", {}).get("type") == "public"
            else BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=[]))
        )

        result = benchmarks_resource.get()

        assert isinstance(result, list)
        assert len(result) == 2

        assert isinstance(result[0], Benchmark)
        assert result[0].name == "MMLU"
        assert result[0].key == "mmlu"

        assert isinstance(result[1], Benchmark)
        assert result[1].name == "My Custom Benchmark"
        assert result[1].key == "my-benchmark"

    def test_get_benchmarks_request_parameters(self, benchmarks_resource, mock_benchmarks_response):
        """get method makes correct API request for benchmarks."""
        benchmarks_resource._get.return_value = mock_benchmarks_response

        benchmarks_resource.get()

        expected_calls = [
            call(
                "/organizations/org-123/projects/proj-456/benchmarks",
                params={"type": "custom"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=BenchmarksResponse,
            ),
            call(
                "/organizations/org-123/projects/proj-456/benchmarks",
                params={"type": "public"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=BenchmarksResponse,
            ),
        ]

        benchmarks_resource._get.assert_has_calls(expected_calls)

    def test_get_benchmarks_with_custom_timeout(self, benchmarks_resource, mock_benchmarks_response):
        """get method accepts custom timeout."""
        benchmarks_resource._get.return_value = mock_benchmarks_response
        custom_timeout = 45.0

        benchmarks_resource.get(timeout=custom_timeout)

        call_args = benchmarks_resource._get.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_get_benchmarks_with_httpx_timeout(self, benchmarks_resource, mock_benchmarks_response):
        """get method accepts httpx.Timeout object."""
        benchmarks_resource._get.return_value = mock_benchmarks_response
        custom_timeout = httpx.Timeout(45.0)

        benchmarks_resource.get(timeout=custom_timeout)

        call_args = benchmarks_resource._get.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    @pytest.mark.parametrize(
        "mock_return, expected",
        [
            (None, []),
            ("invalid-response", []),
            (BenchmarksResponse(data=BenchmarksResponse.Data(datasets=[])), []),
        ],
        ids=["none_response", "invalid_type", "empty_response"],
    )
    def test_get_benchmarks_various_responses(self, benchmarks_resource, mock_return, expected):
        benchmarks_resource._get.return_value = mock_return

        result = benchmarks_resource.get()

        assert result == expected
        if expected == []:
            assert isinstance(result, list)

    def test_get_benchmarks_multiple_items(
        self, benchmarks_resource, sample_benchmark_data, sample_custom_benchmark_data
    ):
        """get method returns multiple benchmarks correctly."""
        _ = sample_custom_benchmark_data  # Fixture used for side effects
        benchmark = Benchmark(**sample_benchmark_data)

        # Create second benchmark with different data
        benchmark2_data = sample_benchmark_data.copy()
        benchmark2_data["id"] = "benchmark-456"
        benchmark2_data["key"] = "hellaswag"
        benchmark2_data["name"] = "HellaSwag"
        benchmark2 = Benchmark(**benchmark2_data)

        response = BenchmarksResponse(data=BenchmarksResponse.Data(datasets=[benchmark, benchmark2]))
        benchmarks_resource._get.side_effect = lambda *_, **kwargs: (
            response
            if kwargs.get("params", {}).get("type") == "public"
            else BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=[]))
        )

        result = benchmarks_resource.get()

        assert len(result) == 2
        assert result[0].key == "mmlu"
        assert result[1].key == "hellaswag"

    def test_get_benchmarks_url_construction(self, benchmarks_resource, mock_benchmarks_response):
        """get method constructs URL correctly with org and project IDs."""
        benchmarks_resource._client.organization_id = "custom-org"
        benchmarks_resource._client.project_id = "custom-project"
        benchmarks_resource._get.return_value = mock_benchmarks_response

        benchmarks_resource.get()

        expected_url = "/organizations/custom-org/projects/custom-project/benchmarks"
        call_args = benchmarks_resource._get.call_args
        assert call_args[0][0] == expected_url

    def test_get_benchmarks_cast_to_parameter(self, benchmarks_resource, mock_benchmarks_response):
        """get method specifies correct cast_to parameter."""
        benchmarks_resource._get.return_value = mock_benchmarks_response

        benchmarks_resource.get()

        call_args = benchmarks_resource._get.call_args
        assert call_args.kwargs["cast_to"] is BenchmarksResponse

    def test_get_benchmarks_timeout_default(self, benchmarks_resource, mock_benchmarks_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        benchmarks_resource._get.return_value = mock_benchmarks_response

        benchmarks_resource.get()

        call_args = benchmarks_resource._get.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_benchmarks_with_none_timeout(self, benchmarks_resource, mock_benchmarks_response):
        """get method accepts None timeout."""
        benchmarks_resource._get.return_value = mock_benchmarks_response

        benchmarks_resource.get(timeout=None)

        call_args = benchmarks_resource._get.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_by_id_custom_benchmark(self, benchmarks_resource, sample_custom_benchmark_data):
        """get_by_id returns CustomBenchmark when organization_id is present."""
        sample_custom_benchmark_data["organization_id"] = "org-123"  # Required for type detection
        benchmarks_resource._get.return_value = {"data": sample_custom_benchmark_data}

        result = benchmarks_resource.get_by_id("custom-benchmark-456")

        assert isinstance(result, CustomBenchmark)
        assert result.id == "custom-benchmark-456"
        assert result.name == "My Custom Benchmark"

    def test_get_by_id_public_benchmark(self, benchmarks_resource, sample_benchmark_data):
        """get_by_id returns PublicBenchmark when organization_id is missing."""
        benchmarks_resource._get.return_value = {"data": sample_benchmark_data}

        result = benchmarks_resource.get_by_id("benchmark-123")

        assert isinstance(result, PublicBenchmark)
        assert result.id == "benchmark-123"
        assert result.name == "MMLU"

    def test_get_by_id_invalid_response(self, benchmarks_resource):
        """get_by_id returns None for invalid responses."""
        benchmarks_resource._get.return_value = "not-a-dict"

        result = benchmarks_resource.get_by_id("invalid")

        assert result is None

    def test_get_by_key_custom_benchmark(self, benchmarks_resource, sample_custom_benchmark_data):
        """get_by_key returns CustomBenchmark when key matches and organization_id is present."""
        sample_custom_benchmark_data["organization_id"] = "org-123"
        custom_benchmark = CustomBenchmark(**sample_custom_benchmark_data)
        benchmarks_resource.get = Mock(return_value=[custom_benchmark])

        result = benchmarks_resource.get_by_key(key="my-benchmark")

        assert isinstance(result, CustomBenchmark)
        assert result.key == "my-benchmark"
        assert result.name == "My Custom Benchmark"

    def test_get_by_key_public_benchmark(self, benchmarks_resource, sample_benchmark_data):
        """get_by_key returns PublicBenchmark when key matches and organization_id is missing."""
        public_benchmark = PublicBenchmark(**sample_benchmark_data)
        benchmarks_resource.get = Mock(return_value=[public_benchmark])

        result = benchmarks_resource.get_by_key(key="mmlu")

        assert isinstance(result, PublicBenchmark)
        assert result.key == "mmlu"
        assert result.name == "MMLU"

    def test_get_by_key_no_match(self, benchmarks_resource, sample_benchmark_data):
        """get_by_key returns None if no benchmark has the exact key."""
        public_benchmark = PublicBenchmark(**sample_benchmark_data)
        benchmarks_resource.get = Mock(return_value=[public_benchmark])

        result = benchmarks_resource.get_by_key(key="nonexistent-key")

        assert result is None

    def test_get_by_key_invalid_response(self, benchmarks_resource):
        """get_by_key returns None when get() returns None or invalid type."""
        benchmarks_resource.get = Mock(return_value=None)

        result = benchmarks_resource.get_by_key(key="some-key")

        assert result is None


class TestBenchmarksErrorHandling:
    """Test error handling in Benchmarks resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        """Benchmarks resource instance."""
        return Benchmarks(mock_client)

    def test_get_benchmarks_handles_api_error(self, benchmarks_resource):
        """get method propagates API errors."""
        from layerlens._exceptions import APIStatusError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        api_error = APIStatusError("Not Found", response=mock_response, body=None)
        benchmarks_resource._get.side_effect = api_error

        with pytest.raises(APIStatusError):
            benchmarks_resource.get()

    def test_get_benchmarks_handles_auth_error(self, benchmarks_resource):
        """get method propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        auth_error = AuthenticationError("Unauthorized", response=mock_response, body=None)
        benchmarks_resource._get.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            benchmarks_resource.get()

    def test_get_benchmarks_handles_connection_error(self, benchmarks_resource):
        """get method propagates connection errors."""
        from layerlens._exceptions import APIConnectionError

        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        benchmarks_resource._get.side_effect = connection_error

        with pytest.raises(APIConnectionError):
            benchmarks_resource.get()

    def test_get_benchmarks_handles_timeout_error(self, benchmarks_resource):
        """get method propagates timeout errors."""
        from layerlens._exceptions import APITimeoutError

        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        benchmarks_resource._get.side_effect = timeout_error

        with pytest.raises(APITimeoutError):
            benchmarks_resource.get(timeout=1.0)


class TestBenchmarksTyping:
    """Test type handling in Benchmarks resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        """Benchmarks resource instance."""
        return Benchmarks(mock_client)

    def test_get_benchmarks_return_type_consistency(self, benchmarks_resource):
        """get method returns consistent types."""
        # Test that the method returns either a list or None
        benchmarks_resource._get.return_value = None
        result = benchmarks_resource.get()
        assert result == []

        # Test that it returns a list when successful
        benchmarks_resource._get.return_value = BenchmarksResponse(data=BenchmarksResponse.Data(datasets=[]))
        result = benchmarks_resource.get()
        assert result == []

    def test_get_benchmarks_mixed_benchmark_types(self, benchmarks_resource):
        """get method can handle mixed benchmark types in response."""
        # Create mixed response with both Benchmark and CustomBenchmark
        public_data = {
            "id": "public-123",
            "key": "mmlu",
            "name": "MMLU",
            "full_description": "Public benchmark",
            "language": "english",
            "categories": ["reasoning"],
            "subsets": ["math"],
            "prompt_count": 1000,
            "deprecated": False,
        }

        custom_data = {
            "id": "custom-456",
            "key": "my-bench",
            "name": "My Benchmark",
            "description": "Custom benchmark",
            "system_prompt": None,
            "subsets": ["custom"],
            "prompt_count": 50,
            "version_count": 1,
            "regex_pattern": None,
            "llm_judge_model_id": "gpt-4",
            "custom_instructions": "Custom instructions",
            "scoring_metric": None,
            "metrics": ["accuracy"],
            "files": ["test.jsonl"],
            "disabled": False,
        }

        public_benchmark = PublicBenchmark(**public_data)
        custom_benchmark = CustomBenchmark(**custom_data)

        benchmarks_resource._get.side_effect = lambda *_, **kwargs: (
            BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=[public_benchmark]))
            if kwargs.get("params", {}).get("type") == "public"
            else BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=[custom_benchmark]))
        )

        result = benchmarks_resource.get()  # Type doesn't matter for this test

        assert len(result) == 2
        assert isinstance(result[0], CustomBenchmark)
        assert isinstance(result[1], PublicBenchmark)
        assert result[0].key == "my-bench"
        assert result[1].key == "mmlu"
