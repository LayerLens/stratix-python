from unittest.mock import Mock, call, patch

import httpx
import pytest

from layerlens.models import (
    Benchmark,
    CustomBenchmark,
    PublicBenchmark,
    BenchmarksResponse,
    CreateBenchmarkResponse,
)
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.benchmarks.benchmarks import Benchmarks


class TestBenchmarks:
    """Test Benchmarks resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Stratix client."""
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
        """Mock Stratix client."""
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
        """Mock Stratix client."""
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


class TestBenchmarksAdd:
    """Test Benchmarks.add() method."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.patch_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    def test_add_single_benchmark(self, benchmarks_resource):
        """add() merges new ID with current benchmarks and PATCHes."""
        existing = PublicBenchmark(id="b1", key="b1", name="B1")
        benchmarks_resource.get = Mock(return_value=[existing])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        result = benchmarks_resource.add("b2")

        assert result is True
        benchmarks_resource._patch.assert_called_once_with(
            "/organizations/org-123/projects/proj-456",
            body={"datasets": ["b1", "b2"]},
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_add_multiple_benchmarks(self, benchmarks_resource):
        """add() handles multiple benchmark IDs."""
        benchmarks_resource.get = Mock(return_value=[])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        result = benchmarks_resource.add("b1", "b2", "b3")

        assert result is True
        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b1", "b2", "b3"]}

    def test_add_deduplicates(self, benchmarks_resource):
        """add() deduplicates IDs already in the project."""
        existing = PublicBenchmark(id="b1", key="b1", name="B1")
        benchmarks_resource.get = Mock(return_value=[existing])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        benchmarks_resource.add("b1", "b2")

        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b1", "b2"]}

    def test_add_returns_false_on_failure(self, benchmarks_resource):
        """add() returns False when PATCH fails."""
        benchmarks_resource.get = Mock(return_value=[])
        benchmarks_resource._patch.return_value = "error"

        result = benchmarks_resource.add("b1")

        assert result is False

    def test_add_with_none_get_response(self, benchmarks_resource):
        """add() handles None from get() gracefully."""
        benchmarks_resource.get = Mock(return_value=None)
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        result = benchmarks_resource.add("b1")

        assert result is True
        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b1"]}

    def test_add_uses_datasets_field(self, benchmarks_resource):
        """add() sends 'datasets' (not 'benchmarks') in the PATCH body."""
        benchmarks_resource.get = Mock(return_value=[])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        benchmarks_resource.add("b1")

        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert "datasets" in call_body
        assert "benchmarks" not in call_body


class TestBenchmarksRemove:
    """Test Benchmarks.remove() method."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.patch_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    def test_remove_single_benchmark(self, benchmarks_resource):
        """remove() removes specified ID and PATCHes remaining."""
        b1 = PublicBenchmark(id="b1", key="b1", name="B1")
        b2 = PublicBenchmark(id="b2", key="b2", name="B2")
        benchmarks_resource.get = Mock(return_value=[b1, b2])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        result = benchmarks_resource.remove("b1")

        assert result is True
        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b2"]}

    def test_remove_multiple_benchmarks(self, benchmarks_resource):
        """remove() handles removing multiple IDs."""
        b1 = PublicBenchmark(id="b1", key="b1", name="B1")
        b2 = PublicBenchmark(id="b2", key="b2", name="B2")
        b3 = PublicBenchmark(id="b3", key="b3", name="B3")
        benchmarks_resource.get = Mock(return_value=[b1, b2, b3])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        benchmarks_resource.remove("b1", "b3")

        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b2"]}

    def test_remove_nonexistent_id(self, benchmarks_resource):
        """remove() ignores IDs that aren't in the project."""
        b1 = PublicBenchmark(id="b1", key="b1", name="B1")
        benchmarks_resource.get = Mock(return_value=[b1])
        benchmarks_resource._patch.return_value = {"id": "proj-456"}

        benchmarks_resource.remove("nonexistent")

        call_body = benchmarks_resource._patch.call_args.kwargs["body"]
        assert call_body == {"datasets": ["b1"]}

    def test_remove_returns_false_on_failure(self, benchmarks_resource):
        """remove() returns False when PATCH fails."""
        benchmarks_resource.get = Mock(return_value=[])
        benchmarks_resource._patch.return_value = None

        result = benchmarks_resource.remove("b1")

        assert result is False


class TestBenchmarksCreateCustom:
    """Test Benchmarks.create_custom() method."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    @pytest.fixture
    def tmp_jsonl(self, tmp_path):
        """Create a temporary JSONL file."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"input": "What is 2+2?", "truth": "4"}\n')
        return str(f)

    def test_create_custom_success_with_envelope(self, benchmarks_resource, tmp_jsonl):
        """create_custom() unwraps envelope and returns CreateBenchmarkResponse."""
        # Mock _upload_file to skip actual upload
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {
                "benchmark_id": "bench-123",
                "organization_id": "org-123",
                "project_id": "proj-456",
            },
        }

        result = benchmarks_resource.create_custom(
            name="Test Benchmark",
            description="Test description",
            file_path=tmp_jsonl,
        )

        assert isinstance(result, CreateBenchmarkResponse)
        assert result.benchmark_id == "bench-123"
        assert result.organization_id == "org-123"

    def test_create_custom_success_without_envelope(self, benchmarks_resource, tmp_jsonl):
        """create_custom() works when response has no envelope."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "benchmark_id": "bench-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
        }

        result = benchmarks_resource.create_custom(
            name="Test",
            description="Test",
            file_path=tmp_jsonl,
        )

        assert isinstance(result, CreateBenchmarkResponse)
        assert result.benchmark_id == "bench-123"

    def test_create_custom_sends_correct_body(self, benchmarks_resource, tmp_jsonl):
        """create_custom() sends all fields in the request body."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_custom(
            name="My Bench",
            description="A benchmark",
            file_path=tmp_jsonl,
            additional_metrics=["toxicity", "readability"],
            custom_scorer_ids=["scorer-1"],
            input_type="messages",
        )

        call_kwargs = benchmarks_resource._post.call_args.kwargs
        assert call_kwargs["body"] == {
            "name": "My Bench",
            "description": "A benchmark",
            "file": "test.jsonl",
            "additional_metrics": ["toxicity", "readability"],
            "custom_scorers": ["scorer-1"],
            "input_type": "messages",
        }

    def test_create_custom_omits_optional_fields(self, benchmarks_resource, tmp_jsonl):
        """create_custom() does not include optional fields when not provided."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_custom(
            name="Bench",
            description="Desc",
            file_path=tmp_jsonl,
        )

        call_body = benchmarks_resource._post.call_args.kwargs["body"]
        assert "additional_metrics" not in call_body
        assert "custom_scorers" not in call_body
        assert "input_type" not in call_body

    def test_create_custom_correct_url(self, benchmarks_resource, tmp_jsonl):
        """create_custom() posts to the correct endpoint."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_custom(
            name="B",
            description="D",
            file_path=tmp_jsonl,
        )

        call_args = benchmarks_resource._post.call_args
        assert call_args[0][0] == "/organizations/org-123/projects/proj-456/custom-benchmarks"

    def test_create_custom_returns_none_on_failure(self, benchmarks_resource, tmp_jsonl):
        """create_custom() returns None when response is unexpected."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = "not-a-dict"

        result = benchmarks_resource.create_custom(
            name="B",
            description="D",
            file_path=tmp_jsonl,
        )

        assert result is None

    def test_create_custom_calls_upload_file(self, benchmarks_resource, tmp_jsonl):
        """create_custom() calls _upload_file with correct args."""
        benchmarks_resource._upload_file = Mock(return_value="test.jsonl")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_custom(
            name="My Bench",
            description="Desc",
            file_path=tmp_jsonl,
        )

        benchmarks_resource._upload_file.assert_called_once_with(tmp_jsonl, "My Bench", DEFAULT_TIMEOUT)


class TestBenchmarksCreateSmart:
    """Test Benchmarks.create_smart() method."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    def test_create_smart_success_with_envelope(self, benchmarks_resource):
        """create_smart() unwraps envelope and returns CreateBenchmarkResponse."""
        benchmarks_resource._upload_file = Mock(return_value="doc.txt")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {
                "benchmark_id": "smart-123",
                "organization_id": "org-123",
                "project_id": "proj-456",
            },
        }

        result = benchmarks_resource.create_smart(
            name="Smart Bench",
            description="Smart benchmark",
            system_prompt="Generate QA pairs",
            file_paths=["/tmp/doc.txt"],
        )

        assert isinstance(result, CreateBenchmarkResponse)
        assert result.benchmark_id == "smart-123"

    def test_create_smart_sends_correct_body(self, benchmarks_resource):
        """create_smart() sends all fields in the request body."""
        benchmarks_resource._upload_file = Mock(side_effect=["doc1.txt", "doc2.pdf"])
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_smart(
            name="Smart",
            description="Desc",
            system_prompt="Generate pairs",
            file_paths=["/tmp/doc1.txt", "/tmp/doc2.pdf"],
            metrics=["hallucination"],
        )

        call_kwargs = benchmarks_resource._post.call_args.kwargs
        assert call_kwargs["body"] == {
            "name": "Smart",
            "description": "Desc",
            "system_prompt": "Generate pairs",
            "files": ["doc1.txt", "doc2.pdf"],
            "metrics": ["hallucination"],
        }

    def test_create_smart_uploads_all_files(self, benchmarks_resource):
        """create_smart() calls _upload_file for each file path."""
        benchmarks_resource._upload_file = Mock(side_effect=["a.txt", "b.pdf", "c.csv"])
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_smart(
            name="S",
            description="D",
            system_prompt="P",
            file_paths=["/tmp/a.txt", "/tmp/b.pdf", "/tmp/c.csv"],
        )

        assert benchmarks_resource._upload_file.call_count == 3

    def test_create_smart_correct_url(self, benchmarks_resource):
        """create_smart() posts to the correct endpoint."""
        benchmarks_resource._upload_file = Mock(return_value="doc.txt")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_smart(
            name="S",
            description="D",
            system_prompt="P",
            file_paths=["/tmp/doc.txt"],
        )

        call_args = benchmarks_resource._post.call_args
        assert call_args[0][0] == "/organizations/org-123/projects/proj-456/smart-benchmarks"

    def test_create_smart_omits_metrics_when_none(self, benchmarks_resource):
        """create_smart() does not include metrics when not provided."""
        benchmarks_resource._upload_file = Mock(return_value="doc.txt")
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"benchmark_id": "x", "organization_id": "o", "project_id": "p"},
        }

        benchmarks_resource.create_smart(
            name="S",
            description="D",
            system_prompt="P",
            file_paths=["/tmp/doc.txt"],
        )

        call_body = benchmarks_resource._post.call_args.kwargs["body"]
        assert "metrics" not in call_body

    def test_create_smart_returns_none_on_failure(self, benchmarks_resource):
        """create_smart() returns None when response is unexpected."""
        benchmarks_resource._upload_file = Mock(return_value="doc.txt")
        benchmarks_resource._post.return_value = None

        result = benchmarks_resource.create_smart(
            name="S",
            description="D",
            system_prompt="P",
            file_paths=["/tmp/doc.txt"],
        )

        assert result is None


class TestBenchmarksUploadFile:
    """Test Benchmarks._upload_file() method."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    @pytest.fixture
    def tmp_jsonl(self, tmp_path):
        """Create a temporary JSONL file."""
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "test", "truth": "answer"}\n')
        return str(f)

    @patch("layerlens.resources.benchmarks.benchmarks.httpx.put")
    def test_upload_file_success_with_envelope(self, mock_put, benchmarks_resource, tmp_jsonl):
        """_upload_file() unwraps envelope and uploads to presigned URL."""
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"url": "https://s3.example.com/upload?signed=1"},
        }
        mock_put.return_value = Mock(status_code=200, raise_for_status=Mock())

        result = benchmarks_resource._upload_file(tmp_jsonl, "my-bench", DEFAULT_TIMEOUT)

        assert result == "data.jsonl"
        mock_put.assert_called_once()
        assert mock_put.call_args.args[0] == "https://s3.example.com/upload?signed=1"

    @patch("layerlens.resources.benchmarks.benchmarks.httpx.put")
    def test_upload_file_success_without_envelope(self, mock_put, benchmarks_resource, tmp_jsonl):
        """_upload_file() works when response has no envelope."""
        benchmarks_resource._post.return_value = {
            "url": "https://s3.example.com/upload?signed=1",
        }
        mock_put.return_value = Mock(status_code=200, raise_for_status=Mock())

        result = benchmarks_resource._upload_file(tmp_jsonl, "my-bench", DEFAULT_TIMEOUT)

        assert result == "data.jsonl"

    def test_upload_file_raises_on_missing_url(self, benchmarks_resource, tmp_jsonl):
        """_upload_file() raises ValueError when URL is missing."""
        benchmarks_resource._post.return_value = {"status": "success", "data": {"no_url": True}}

        with pytest.raises(ValueError, match="Failed to get upload URL"):
            benchmarks_resource._upload_file(tmp_jsonl, "my-bench", DEFAULT_TIMEOUT)

    def test_upload_file_raises_on_invalid_response(self, benchmarks_resource, tmp_jsonl):
        """_upload_file() raises ValueError when response is not a dict."""
        benchmarks_resource._post.return_value = "not-a-dict"

        with pytest.raises(ValueError, match="Failed to get upload URL"):
            benchmarks_resource._upload_file(tmp_jsonl, "my-bench", DEFAULT_TIMEOUT)

    def test_upload_file_raises_on_oversized_file(self, benchmarks_resource, tmp_path):
        """_upload_file() raises ValueError when file exceeds size limit."""
        big_file = tmp_path / "big.jsonl"
        # Create a file that appears to be larger than MAX_UPLOAD_SIZE
        big_file.write_text("x")

        with patch("os.path.getsize", return_value=51 * 1024 * 1024):
            with pytest.raises(ValueError, match="exceeds maximum"):
                benchmarks_resource._upload_file(str(big_file), "my-bench", DEFAULT_TIMEOUT)

    @patch("layerlens.resources.benchmarks.benchmarks.httpx.put")
    def test_upload_file_sends_correct_upload_request(self, mock_put, benchmarks_resource, tmp_jsonl):
        """_upload_file() sends correct metadata to upload endpoint."""
        benchmarks_resource._post.return_value = {
            "status": "success",
            "data": {"url": "https://s3.example.com/upload"},
        }
        mock_put.return_value = Mock(status_code=200, raise_for_status=Mock())

        benchmarks_resource._upload_file(tmp_jsonl, "my-bench", DEFAULT_TIMEOUT)

        post_kwargs = benchmarks_resource._post.call_args.kwargs
        body = post_kwargs["body"]
        assert body["key"] == "my-bench"
        assert body["filename"] == "data.jsonl"
        assert "type" in body
        assert "size" in body


class TestBenchmarksClientSideFiltering:
    """Test client-side filtering for benchmarks (fixes API not filtering custom objects)."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def benchmarks_resource(self, mock_client):
        return Benchmarks(mock_client)

    @pytest.fixture
    def public_reasoning(self):
        return PublicBenchmark(
            id="pub-1",
            key="mmlu",
            name="MMLU",
            language="english",
            categories=["reasoning", "knowledge"],
        )

    @pytest.fixture
    def public_coding(self):
        return PublicBenchmark(
            id="pub-2",
            key="humaneval",
            name="HumanEval",
            language="english",
            categories=["coding"],
        )

    @pytest.fixture
    def public_french(self):
        return PublicBenchmark(
            id="pub-3",
            key="french-bench",
            name="French Bench",
            language="french",
            categories=["reasoning"],
        )

    @pytest.fixture
    def custom_bench(self):
        return CustomBenchmark(
            id="custom-1",
            key="my-bench",
            name="My Custom Benchmark",
        )

    def _mock_responses(self, resource, custom_list, public_list):
        """Helper to set up mock API responses returning custom and public benchmarks."""
        custom_resp = BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=custom_list))
        public_resp = BenchmarksResponse(data=BenchmarksResponse.Data(benchmarks=public_list))
        resource._get.side_effect = lambda *_, **kwargs: (
            custom_resp if kwargs.get("params", {}).get("type") == "custom" else public_resp
        )

    def test_filter_by_categories_excludes_custom(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
        public_coding,
    ):
        """Filtering by categories excludes custom benchmarks (they have no categories)."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning, public_coding])

        result = benchmarks_resource.get(categories=["reasoning"])

        assert len(result) == 1
        assert result[0].key == "mmlu"
        assert isinstance(result[0], PublicBenchmark)

    def test_filter_by_categories_no_match_returns_empty(
        self,
        benchmarks_resource,
        custom_bench,
        public_coding,
    ):
        """Filtering by a category that no benchmark matches returns empty list."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_coding])

        result = benchmarks_resource.get(categories=["math"])

        assert result == []

    def test_filter_by_languages_excludes_custom(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
        public_french,
    ):
        """Filtering by language excludes custom benchmarks (they have no language)."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning, public_french])

        result = benchmarks_resource.get(languages=["french"])

        assert len(result) == 1
        assert result[0].key == "french-bench"

    def test_filter_by_languages_no_match_returns_empty(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
    ):
        """Filtering by a language that no benchmark matches returns empty list."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning])

        result = benchmarks_resource.get(languages=["spanish"])

        assert result == []

    def test_filter_by_key_sends_param_to_api(
        self,
        benchmarks_resource,
        public_reasoning,
    ):
        """Filtering by key sends the key param to the API."""
        self._mock_responses(benchmarks_resource, [], [public_reasoning])

        benchmarks_resource.get(key="mmlu")

        # Verify key param was sent in API calls
        for c in benchmarks_resource._get.call_args_list:
            assert c.kwargs["params"]["key"] == "mmlu"

    def test_combined_filters_categories_and_languages(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
        public_french,
    ):
        """Multiple filters are applied together (AND logic)."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning, public_french])

        result = benchmarks_resource.get(categories=["reasoning"], languages=["french"])

        assert len(result) == 1
        assert result[0].key == "french-bench"

    def test_no_filters_returns_all(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
        public_coding,
    ):
        """When no filters are applied, all benchmarks are returned."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning, public_coding])

        result = benchmarks_resource.get()

        assert len(result) == 3

    def test_filter_case_insensitive(
        self,
        benchmarks_resource,
        custom_bench,
        public_reasoning,
    ):
        """Filters are case-insensitive."""
        self._mock_responses(benchmarks_resource, [custom_bench], [public_reasoning])

        result = benchmarks_resource.get(categories=["REASONING"])

        assert len(result) == 1
        assert result[0].key == "mmlu"
