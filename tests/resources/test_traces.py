import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from layerlens.models import Trace, TracesResponse, CreateTracesResponse
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.traces.traces import Traces


class TestTraces:
    """Test Traces resource API methods."""

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
    def traces_resource(self, mock_client):
        """Traces resource instance."""
        return Traces(mock_client)

    @pytest.fixture
    def sample_trace_data(self):
        """Sample trace data for testing."""
        return {
            "id": "trace-123",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "created_at": "2024-01-01T00:00:00Z",
            "filename": "test-traces.json",
            "data": {"input": "Hello", "output": "Hi there"},
            "input": "Hello",
        }

    @pytest.fixture
    def sample_trace_with_evals_data(self, sample_trace_data):
        """Sample trace with evaluations data."""
        return {
            **sample_trace_data,
            "evaluations_count": 2,
            "last_evaluations": [
                {
                    "judge_id": "judge-1",
                    "judge_name": "Quality Judge",
                    "judge_version": 1,
                    "created_at": "2024-01-02T00:00:00Z",
                    "passed": True,
                },
            ],
        }

    def test_traces_initialization(self, mock_client):
        """Traces resource initializes correctly."""
        traces = Traces(mock_client)

        assert traces._client is mock_client
        assert traces._get is mock_client.get_cast
        assert traces._post is mock_client.post_cast
        assert traces._delete is mock_client.delete_cast

    def test_get_trace_success(self, traces_resource, sample_trace_data):
        """get method returns Trace on success."""
        traces_resource._get.return_value = sample_trace_data

        result = traces_resource.get("trace-123")

        assert isinstance(result, Trace)
        assert result.id == "trace-123"
        assert result.filename == "test-traces.json"

    def test_get_trace_request_parameters(self, traces_resource, sample_trace_data):
        """get method makes correct API request."""
        traces_resource._get.return_value = sample_trace_data

        traces_resource.get("trace-123")

        traces_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/traces/trace-123",
            timeout=DEFAULT_TIMEOUT,
            cast_to=dict,
        )

    def test_get_trace_none_response(self, traces_resource):
        """get method returns None when response is invalid."""
        traces_resource._get.return_value = None

        result = traces_resource.get("trace-123")

        assert result is None

    def test_get_many_traces_success(self, traces_resource, sample_trace_with_evals_data):
        """get_many returns TracesResponse on success."""
        traces_resource._get.return_value = {
            "traces": [sample_trace_with_evals_data],
            "count": 1,
            "total_count": 1,
        }

        result = traces_resource.get_many()

        assert isinstance(result, TracesResponse)
        assert len(result.traces) == 1
        assert result.traces[0].id == "trace-123"
        assert result.traces[0].evaluations_count == 2
        assert result.count == 1

    def test_get_many_traces_with_filters(self, traces_resource, sample_trace_with_evals_data):
        """get_many passes filter parameters correctly."""
        traces_resource._get.return_value = {
            "traces": [sample_trace_with_evals_data],
            "count": 1,
            "total_count": 1,
        }

        traces_resource.get_many(
            page=2,
            page_size=50,
            source="test-file.json",
            judge_id="judge-1",
            status="pass",
            time_range="7d",
            search="hello",
            sort_by="created_at",
            sort_order="desc",
        )

        call_args = traces_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "2"
        assert params["pageSize"] == "50"
        assert params["source"] == "test-file.json"
        assert params["judgeId"] == "judge-1"
        assert params["status"] == "pass"
        assert params["timeRange"] == "7d"
        assert params["search"] == "hello"
        assert params["sortBy"] == "created_at"
        assert params["sortOrder"] == "desc"

    def test_get_many_traces_default_pagination(self, traces_resource):
        """get_many uses default pagination."""
        traces_resource._get.return_value = {
            "traces": [],
            "count": 0,
            "total_count": 0,
        }

        traces_resource.get_many()

        call_args = traces_resource._get.call_args
        params = call_args.kwargs["params"]
        assert params["page"] == "1"
        assert params["pageSize"] == "100"

    def test_get_many_traces_none_response(self, traces_resource):
        """get_many returns None when response is invalid."""
        traces_resource._get.return_value = None

        result = traces_resource.get_many()

        assert result is None

    def test_delete_trace_success(self, traces_resource):
        """delete method returns True on success."""
        traces_resource._delete.return_value = Mock()

        result = traces_resource.delete("trace-123")

        assert result is True

    def test_delete_trace_request_parameters(self, traces_resource):
        """delete method makes correct API request."""
        traces_resource._delete.return_value = Mock()

        traces_resource.delete("trace-123")

        traces_resource._delete.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/traces/trace-123",
            timeout=DEFAULT_TIMEOUT,
        )

    def test_delete_trace_none_response(self, traces_resource):
        """delete method returns False when response is None."""
        traces_resource._delete.return_value = None

        result = traces_resource.delete("trace-123")

        assert result is False

    def test_get_sources_success(self, traces_resource):
        """get_sources returns list of source filenames."""
        traces_resource._get.return_value = {
            "sources": ["file1.json", "file2.jsonl", "imported-traces.json"],
        }

        result = traces_resource.get_sources()

        assert result == ["file1.json", "file2.jsonl", "imported-traces.json"]

    def test_get_sources_empty(self, traces_resource):
        """get_sources returns empty list when no sources."""
        traces_resource._get.return_value = {"sources": []}

        result = traces_resource.get_sources()

        assert result == []

    def test_get_sources_invalid_response(self, traces_resource):
        """get_sources returns empty list on invalid response."""
        traces_resource._get.return_value = None

        result = traces_resource.get_sources()

        assert result == []


class TestTracesUpload:
    """Test Traces upload functionality."""

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
    def traces_resource(self, mock_client):
        """Traces resource instance."""
        return Traces(mock_client)

    def test_upload_json_file(self, traces_resource):
        """upload method handles JSON files correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('[{"input": "hello", "output": "world"}]')
            tmp_path = f.name

        try:
            # Mock step 1: presigned URL (now returns dict)
            traces_resource._post.side_effect = [
                {"url": "https://s3.example.com/upload?signed=true"},
                {"trace_ids": ["trace-new-1"]},
            ]

            # Mock step 2: S3 upload
            with patch("layerlens.resources.traces.traces.httpx.put") as mock_put:
                mock_put.return_value = Mock(status_code=200, raise_for_status=Mock())

                result = traces_resource.upload(tmp_path)

            assert isinstance(result, CreateTracesResponse)
            assert result.trace_ids == ["trace-new-1"]

            # Verify step 1 call
            first_call = traces_resource._post.call_args_list[0]
            assert first_call[0][0].endswith("/traces/upload")
            body = first_call.kwargs["body"]
            assert body["filename"] == os.path.basename(tmp_path)
            assert body["type"] == "application/json"

            # Verify step 2 call
            mock_put.assert_called_once()
            put_args = mock_put.call_args
            assert put_args[0][0] == "https://s3.example.com/upload?signed=true"

            # Verify step 3 call
            third_call = traces_resource._post.call_args_list[1]
            assert third_call[0][0].endswith("/traces")
            assert third_call.kwargs["body"]["filename"] == os.path.basename(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_upload_jsonl_file(self, traces_resource):
        """upload method handles JSONL files correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write('{"input": "hello"}\n{"input": "world"}\n')
            tmp_path = f.name

        try:
            traces_resource._post.side_effect = [
                {"url": "https://s3.example.com/upload"},
                {"trace_ids": ["trace-1", "trace-2"]},
            ]

            with patch("layerlens.resources.traces.traces.httpx.put") as mock_put:
                mock_put.return_value = Mock(status_code=200, raise_for_status=Mock())

                result = traces_resource.upload(tmp_path)

            assert isinstance(result, CreateTracesResponse)
            assert len(result.trace_ids) == 2

            first_call = traces_resource._post.call_args_list[0]
            assert first_call.kwargs["body"]["type"] == "application/jsonl"
        finally:
            os.unlink(tmp_path)

    def test_upload_file_too_large(self, traces_resource):
        """upload method raises ValueError for files exceeding 50 MB."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            tmp_path = f.name

        try:
            with patch("layerlens.resources.traces.traces.os.path.getsize", return_value=51 * 1024 * 1024):
                with pytest.raises(ValueError, match="exceeds maximum"):
                    traces_resource.upload(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_upload_presigned_url_failure(self, traces_resource):
        """upload method returns None when presigned URL request fails."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("[]")
            tmp_path = f.name

        try:
            traces_resource._post.return_value = None

            result = traces_resource.upload(tmp_path)

            assert result is None
        finally:
            os.unlink(tmp_path)


class TestTracesErrorHandling:
    """Test error handling in Traces resource."""

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
    def traces_resource(self, mock_client):
        """Traces resource instance."""
        return Traces(mock_client)

    def test_get_trace_handles_not_found(self, traces_resource):
        """get method propagates not found errors."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        traces_resource._get.side_effect = NotFoundError("Trace not found", response=mock_response, body=None)

        with pytest.raises(NotFoundError):
            traces_resource.get("nonexistent-trace")

    def test_delete_trace_handles_auth_error(self, traces_resource):
        """delete method propagates authentication errors."""
        from layerlens._exceptions import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        traces_resource._delete.side_effect = AuthenticationError("Unauthorized", response=mock_response, body=None)

        with pytest.raises(AuthenticationError):
            traces_resource.delete("trace-123")


class TestTracesURLConstruction:
    """Test URL construction in Traces resource."""

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
    def traces_resource(self, mock_client):
        """Traces resource instance."""
        return Traces(mock_client)

    def test_base_url_construction(self, traces_resource):
        """Base URL uses correct organization and project IDs."""
        assert traces_resource._base_url() == "/organizations/custom-org/projects/custom-proj/traces"

    def test_get_url_includes_trace_id(self, traces_resource):
        """Get URL includes trace ID."""
        traces_resource._get.return_value = None

        traces_resource.get("trace-abc")

        call_args = traces_resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/traces/trace-abc"

    def test_delete_url_includes_trace_id(self, traces_resource):
        """Delete URL includes trace ID."""
        traces_resource._delete.return_value = None

        traces_resource.delete("trace-xyz")

        call_args = traces_resource._delete.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/traces/trace-xyz"

    def test_sources_url(self, traces_resource):
        """Sources URL is constructed correctly."""
        traces_resource._get.return_value = {"sources": []}

        traces_resource.get_sources()

        call_args = traces_resource._get.call_args
        assert call_args[0][0] == "/organizations/custom-org/projects/custom-proj/traces/sources"
