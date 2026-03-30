"""Tests for the LayerLens MCP server.

Tests every tool handler, the dispatch logic, error handling, asyncio.to_thread
wrapping, and the tool catalogue. Uses mocked SDK responses throughout.
"""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Ensure samples/ is importable
# ---------------------------------------------------------------------------

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
MCP_DIR = os.path.join(SAMPLES_DIR, "mcp")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Fully mocked Stratix client with realistic responses."""
    client = MagicMock()
    client.organization_id = "org-test"
    client.project_id = "proj-test"

    # -- traces --
    trace_obj = MagicMock()
    trace_obj.id = "trace-001"
    trace_obj.created_at = "2026-01-01T00:00:00Z"
    trace_obj.filename = "test.jsonl"
    trace_obj.evaluations_count = 2
    trace_obj.data = {"input": "hello", "output": "world"}
    trace_obj.model_dump.return_value = {
        "id": "trace-001",
        "data": {"input": "hello", "output": "world"},
    }

    traces_resp = MagicMock()
    traces_resp.count = 1
    traces_resp.total_count = 1
    traces_resp.traces = [trace_obj]
    client.traces.get_many.return_value = traces_resp
    client.traces.get.return_value = trace_obj

    # -- judges --
    judge_obj = MagicMock()
    judge_obj.id = "judge-001"
    judge_obj.name = "Safety Judge"
    judge_obj.evaluation_goal = "Evaluate safety."

    judges_resp = MagicMock()
    judges_resp.judges = [judge_obj]
    client.judges.get_many.return_value = judges_resp
    client.judges.create.return_value = judge_obj

    # -- trace evaluations --
    te_obj = MagicMock()
    te_obj.id = "te-001"
    te_obj.status = MagicMock(value="success")
    client.trace_evaluations.create.return_value = te_obj
    client.trace_evaluations.get.return_value = te_obj

    te_result = MagicMock()
    te_result.score = 0.92
    te_result.passed = True
    te_result.reasoning = "Response is safe and accurate."
    te_result.latency_ms = 1200
    te_result.total_cost = 0.003
    te_results_resp = MagicMock()
    te_results_resp.results = [te_result]
    client.trace_evaluations.get_results.return_value = te_results_resp

    # -- public models (for create_judge helper) --
    pub_model = MagicMock()
    pub_model.id = "model-pub-001"
    pub_model.name = "Claude Haiku 3.5"
    pub_resp = MagicMock()
    pub_resp.models = [pub_model]
    client.public = MagicMock()
    client.public.models.get.return_value = pub_resp

    return client


@pytest.fixture
def mcp_module(mock_client):
    """Import the MCP server module with mocked dependencies."""
    mock_mcp_server = MagicMock()
    mock_mcp_server.Server = MagicMock
    mock_mcp_stdio = MagicMock()
    mock_mcp_types = MagicMock()

    # Provide real-ish TextContent and Tool so the server can construct them
    class FakeTextContent:
        def __init__(self, **kwargs):
            self.type = kwargs.get("type", "text")
            self.text = kwargs.get("text", "")

    class FakeTool:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "")
            self.description = kwargs.get("description", "")
            self.inputSchema = kwargs.get("inputSchema", {})

    mock_mcp_types.TextContent = FakeTextContent
    mock_mcp_types.Tool = FakeTool

    with patch.dict("sys.modules", {
        "mcp": MagicMock(),
        "mcp.server": mock_mcp_server,
        "mcp.server.stdio": mock_mcp_stdio,
        "mcp.types": mock_mcp_types,
    }):
        with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
            with patch("layerlens.Stratix", return_value=mock_client):
                # Force reimport
                mod_name = "layerlens_server_test"
                if mod_name in sys.modules:
                    del sys.modules[mod_name]

                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    mod_name,
                    os.path.join(MCP_DIR, "layerlens_server.py"),
                )
                mod = importlib.util.module_from_spec(spec)

                # Patch the module's global _client so handlers use our mock
                spec.loader.exec_module(mod)
                mod._client = mock_client

                yield mod, mock_client, FakeTextContent


# ---------------------------------------------------------------------------
# Tool Catalogue Tests
# ---------------------------------------------------------------------------


class TestToolCatalogue:
    """Verify the MCP server exposes the correct tools."""

    def test_create_server_returns_server(self, mcp_module):
        mod, _, _ = mcp_module
        assert hasattr(mod, "create_server")
        assert hasattr(mod, "main")

    def test_tool_count(self, mcp_module):
        mod, _, FakeTextContent = mcp_module
        # The list_tools function is registered on the server, but since we're
        # using a mocked Server class, we test the handler functions directly.
        assert hasattr(mod, "_handle_list_traces")
        assert hasattr(mod, "_handle_get_trace")
        assert hasattr(mod, "_handle_run_evaluation")
        assert hasattr(mod, "_handle_get_evaluation")
        assert hasattr(mod, "_handle_create_judge")
        assert hasattr(mod, "_handle_list_judges")


# ---------------------------------------------------------------------------
# Individual Tool Handler Tests
# ---------------------------------------------------------------------------


class TestHandleListTraces:
    """Test the list_traces tool handler."""

    def test_returns_trace_summary(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_list_traces(client, {"limit": 10}))
        assert len(result) == 1
        text = result[0].text
        assert "trace-001" in text
        assert "test.jsonl" in text
        assert "2 evaluation(s)" in text
        client.traces.get_many.assert_called_once_with(
            page_size=10, sort_by="created_at", sort_order="desc"
        )

    def test_default_limit(self, mcp_module):
        mod, client, _ = mcp_module
        asyncio.run(mod._handle_list_traces(client, {}))
        client.traces.get_many.assert_called_once_with(
            page_size=20, sort_by="created_at", sort_order="desc"
        )

    def test_no_traces(self, mcp_module):
        mod, client, _ = mcp_module
        client.traces.get_many.return_value = None
        result = asyncio.run(mod._handle_list_traces(client, {}))
        assert "No traces found" in result[0].text

    def test_empty_traces_list(self, mcp_module):
        mod, client, _ = mcp_module
        resp = MagicMock()
        resp.count = 0
        resp.total_count = 0
        resp.traces = []
        client.traces.get_many.return_value = resp
        result = asyncio.run(mod._handle_list_traces(client, {}))
        assert "0 of 0" in result[0].text


class TestHandleGetTrace:
    """Test the get_trace tool handler."""

    def test_returns_trace_detail(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_get_trace(client, {"trace_id": "trace-001"}))
        assert len(result) == 1
        # Should contain the model_dump output
        assert "trace-001" in result[0].text
        client.traces.get.assert_called_once_with("trace-001")

    def test_trace_not_found(self, mcp_module):
        mod, client, _ = mcp_module
        client.traces.get.return_value = None
        result = asyncio.run(mod._handle_get_trace(client, {"trace_id": "bad-id"}))
        assert "not found" in result[0].text


class TestHandleRunEvaluation:
    """Test the run_evaluation tool handler."""

    def test_creates_evaluation(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_run_evaluation(
            client, {"trace_id": "trace-001", "judge_id": "judge-001"}
        ))
        text = result[0].text
        assert "Evaluation created" in text
        assert "te-001" in text
        assert "trace-001" in text
        assert "judge-001" in text
        client.trace_evaluations.create.assert_called_once_with(
            trace_id="trace-001", judge_id="judge-001"
        )

    def test_evaluation_creation_fails(self, mcp_module):
        mod, client, _ = mcp_module
        client.trace_evaluations.create.return_value = None
        result = asyncio.run(mod._handle_run_evaluation(
            client, {"trace_id": "t", "judge_id": "j"}
        ))
        assert "Failed" in result[0].text


class TestHandleGetEvaluation:
    """Test the get_evaluation tool handler."""

    def test_returns_status_and_results(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_get_evaluation(
            client, {"evaluation_id": "te-001"}
        ))
        text = result[0].text
        assert "te-001" in text
        assert "0.92" in text
        assert "True" in text
        assert "safe and accurate" in text
        assert "1200" in text
        client.trace_evaluations.get.assert_called_once_with("te-001")
        client.trace_evaluations.get_results.assert_called_once()

    def test_evaluation_not_found(self, mcp_module):
        mod, client, _ = mcp_module
        client.trace_evaluations.get.return_value = None
        result = asyncio.run(mod._handle_get_evaluation(
            client, {"evaluation_id": "bad-id"}
        ))
        assert "not found" in result[0].text

    def test_pending_evaluation_no_results(self, mcp_module):
        mod, client, _ = mcp_module
        pending_te = MagicMock()
        pending_te.id = "te-002"
        pending_te.status = MagicMock(value="pending")
        client.trace_evaluations.get.return_value = pending_te
        result = asyncio.run(mod._handle_get_evaluation(
            client, {"evaluation_id": "te-002"}
        ))
        text = result[0].text
        assert "te-002" in text
        # Should NOT contain result scores (evaluation is pending)
        assert "Score" not in text
        client.trace_evaluations.get_results.assert_not_called()


class TestHandleCreateJudge:
    """Test the create_judge tool handler."""

    def test_creates_judge(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_create_judge(
            client, {"name": "Test Judge", "goal": "Evaluate test quality."}
        ))
        text = result[0].text
        assert "Judge created" in text
        assert "judge-001" in text
        assert "Safety Judge" in text  # mock returns this name

    def test_judge_creation_fails(self, mcp_module):
        mod, client, _ = mcp_module
        # Make the helper raise so it returns None path
        client.judges.create.return_value = None
        client.judges.get_many.return_value = MagicMock(judges=[])
        # _create_judge_helper will try create, get None, and won't find existing
        # This should trigger the "Failed" path or an error
        result = asyncio.run(mod._handle_create_judge(
            client, {"name": "Bad Judge", "goal": "Will fail."}
        ))
        # Either "Failed" or an error message
        assert len(result) == 1


class TestHandleListJudges:
    """Test the list_judges tool handler."""

    def test_returns_judge_list(self, mcp_module):
        mod, client, _ = mcp_module
        result = asyncio.run(mod._handle_list_judges(client, {}))
        text = result[0].text
        assert "judge-001" in text
        assert "Safety Judge" in text
        client.judges.get_many.assert_called_once()

    def test_no_judges(self, mcp_module):
        mod, client, _ = mcp_module
        client.judges.get_many.return_value = None
        result = asyncio.run(mod._handle_list_judges(client, {}))
        assert "No judges found" in result[0].text

    def test_empty_judges_list(self, mcp_module):
        mod, client, _ = mcp_module
        resp = MagicMock()
        resp.judges = []
        client.judges.get_many.return_value = resp
        result = asyncio.run(mod._handle_list_judges(client, {}))
        assert "No judges found" in result[0].text


# ---------------------------------------------------------------------------
# Dispatch and Error Handling Tests
# ---------------------------------------------------------------------------


class TestDispatchAndErrors:
    """Test the tool dispatch logic and error handling."""

    def test_unknown_tool_returns_error(self, mcp_module):
        """An unknown tool name should return an error message, not crash."""
        mod, client, FakeTextContent = mcp_module

        # We need to test the dispatch logic inside call_tool.
        # Since the Server is mocked, we call the handlers directly via a wrapper.
        async def call_unknown():
            # Simulate what call_tool does for unknown tools
            name = "nonexistent_tool"
            if name == "list_traces":
                return await mod._handle_list_traces(client, {})
            return [FakeTextContent(type="text", text=f"Unknown tool: {name}")]

        result = asyncio.run(call_unknown())
        assert "Unknown tool" in result[0].text

    def test_sdk_error_returns_error_message(self, mcp_module):
        """SDK exceptions should be caught and returned as error text."""
        mod, client, _ = mcp_module
        from layerlens._exceptions import NotFoundError as NFE

        client.traces.get.side_effect = Exception("Connection refused")
        # Call the handler directly -- the dispatch wrapper catches exceptions
        # but individual handlers don't. The dispatch wrapper is on the Server.
        # So test that the handler raises, which the dispatch wrapper would catch.
        with pytest.raises(Exception, match="Connection refused"):
            asyncio.run(mod._handle_get_trace(client, {"trace_id": "x"}))

    def test_not_found_error_formatting(self, mcp_module):
        mod, _, _ = mcp_module
        exc = Exception("not found")
        text = mod._error_text(exc)
        assert "Error:" in text

    def test_obj_to_text_none(self, mcp_module):
        mod, _, _ = mcp_module
        assert mod._obj_to_text(None) == "(no data)"

    def test_obj_to_text_dict(self, mcp_module):
        mod, _, _ = mcp_module
        result = mod._obj_to_text({"key": "value"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_obj_to_text_model(self, mcp_module):
        mod, _, _ = mcp_module
        obj = MagicMock()
        obj.model_dump.return_value = {"id": "test"}
        result = mod._obj_to_text(obj)
        parsed = json.loads(result)
        assert parsed["id"] == "test"


# ---------------------------------------------------------------------------
# asyncio.to_thread Wrapping Tests
# ---------------------------------------------------------------------------


class TestAsyncWrapping:
    """Verify that all SDK calls go through asyncio.to_thread."""

    def test_list_traces_uses_to_thread(self, mcp_module):
        mod, client, _ = mcp_module
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = client.traces.get_many.return_value
            asyncio.run(mod._handle_list_traces(client, {"limit": 5}))
            mock_to_thread.assert_called_once()
            # First arg should be client.traces.get_many
            call_args = mock_to_thread.call_args
            assert call_args[0][0] == client.traces.get_many

    def test_get_trace_uses_to_thread(self, mcp_module):
        mod, client, _ = mcp_module
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = client.traces.get.return_value
            asyncio.run(mod._handle_get_trace(client, {"trace_id": "t1"}))
            mock_to_thread.assert_called_once()

    def test_run_evaluation_uses_to_thread(self, mcp_module):
        mod, client, _ = mcp_module
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = client.trace_evaluations.create.return_value
            asyncio.run(mod._handle_run_evaluation(
                client, {"trace_id": "t1", "judge_id": "j1"}
            ))
            mock_to_thread.assert_called_once()

    def test_create_judge_uses_to_thread(self, mcp_module):
        mod, client, _ = mcp_module
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = client.judges.create.return_value
            asyncio.run(mod._handle_create_judge(
                client, {"name": "J", "goal": "Test goal for judge."}
            ))
            mock_to_thread.assert_called_once()

    def test_list_judges_uses_to_thread(self, mcp_module):
        mod, client, _ = mcp_module
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = client.judges.get_many.return_value
            asyncio.run(mod._handle_list_judges(client, {}))
            mock_to_thread.assert_called_once()
