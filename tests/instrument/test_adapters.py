from __future__ import annotations

import json
import sys
import types
import importlib
from uuid import uuid4
from unittest.mock import Mock

from .conftest import find_events, find_event


def _capture_framework_trace(mock_client):
    """Helper to capture uploaded trace from framework adapters (which manage their own collector)."""
    uploaded = {}

    def _capture(path):
        with open(path) as f:
            data = json.load(f)
        payload = data[0]
        uploaded["trace_id"] = payload.get("trace_id")
        uploaded["events"] = payload.get("events", [])
        uploaded["capture_config"] = payload.get("capture_config", {})
        uploaded["attestation"] = payload.get("attestation", {})

    mock_client.traces.upload.side_effect = _capture
    return uploaded


class TestLangChainAdapter:
    def _setup_langchain_mock(self):
        mock_lc_core = types.ModuleType("langchain_core")
        mock_lc_callbacks = types.ModuleType("langchain_core.callbacks")

        class FakeBaseCallbackHandler:
            def __init__(self):
                pass

        mock_lc_callbacks.BaseCallbackHandler = FakeBaseCallbackHandler
        mock_lc_core.callbacks = mock_lc_callbacks

        sys.modules["langchain_core"] = mock_lc_core
        sys.modules["langchain_core.callbacks"] = mock_lc_callbacks

    def _teardown_langchain_mock(self):
        for key in list(sys.modules.keys()):
            if key.startswith("langchain_core"):
                del sys.modules[key]

    def _get_handler(self, mock_client):
        from layerlens.instrument.adapters.frameworks import langchain as lc_mod

        importlib.reload(lc_mod)
        return lc_mod.LangChainCallbackHandler(mock_client)

    def test_emits_flat_events(self, mock_client):
        self._setup_langchain_mock()
        try:
            uploaded = _capture_framework_trace(mock_client)
            handler = self._get_handler(mock_client)

            chain_run_id = uuid4()
            llm_run_id = uuid4()

            handler.on_chain_start(
                {"name": "RunnableSequence", "id": ["RunnableSequence"]},
                {"question": "What is AI?"},
                run_id=chain_run_id,
            )
            handler.on_llm_start(
                {"name": "ChatOpenAI", "id": ["ChatOpenAI"]},
                ["What is AI?"],
                run_id=llm_run_id,
                parent_run_id=chain_run_id,
            )

            llm_response = Mock()
            llm_response.generations = [[Mock(text="AI is...")]]
            llm_response.llm_output = {"token_usage": {"total_tokens": 50}, "model_name": "gpt-4"}
            handler.on_llm_end(llm_response, run_id=llm_run_id)
            handler.on_chain_end({"output": "AI is..."}, run_id=chain_run_id)

            events = uploaded["events"]
            # Should have: agent.input, model.invoke (start), model.invoke (end), cost.record, agent.output
            agent_input = find_event(events, "agent.input")
            assert agent_input["payload"]["name"] == "RunnableSequence"
            assert agent_input["payload"]["input"] == {"question": "What is AI?"}

            model_invokes = find_events(events, "model.invoke")
            assert len(model_invokes) >= 1
            # The end event has model name and output
            end_invoke = [m for m in model_invokes if m["payload"].get("model") == "gpt-4"]
            assert len(end_invoke) == 1
            assert end_invoke[0]["payload"]["output_message"] == "AI is..."

            cost = find_event(events, "cost.record")
            assert cost["payload"]["total_tokens"] == 50

            agent_output = find_event(events, "agent.output")
            assert agent_output["payload"]["status"] == "ok"

            # Parent-child: LLM events should reference chain's span_id as parent
            chain_span_id = agent_input["span_id"]
            llm_start = [m for m in model_invokes if m["payload"].get("name") == "ChatOpenAI"][0]
            assert llm_start["parent_span_id"] == chain_span_id
        finally:
            self._teardown_langchain_mock()

    def test_tracks_tools_and_retrievers(self, mock_client):
        self._setup_langchain_mock()
        try:
            uploaded = _capture_framework_trace(mock_client)
            handler = self._get_handler(mock_client)

            chain_id = uuid4()
            tool_id = uuid4()
            retriever_id = uuid4()

            handler.on_chain_start({"name": "Agent"}, {"input": "test"}, run_id=chain_id)
            handler.on_tool_start({"name": "search"}, "query", run_id=tool_id, parent_run_id=chain_id)
            handler.on_tool_end("results", run_id=tool_id)
            handler.on_retriever_start({"name": "vectorstore"}, "query", run_id=retriever_id, parent_run_id=chain_id)

            docs = [Mock(page_content="doc1", metadata={"source": "a"})]
            handler.on_retriever_end(docs, run_id=retriever_id)
            handler.on_chain_end({"output": "done"}, run_id=chain_id)

            events = uploaded["events"]
            tool_calls = find_events(events, "tool.call")
            assert len(tool_calls) == 2  # tool + retriever both emit tool.call
            tool_results = find_events(events, "tool.result")
            assert len(tool_results) == 2
        finally:
            self._teardown_langchain_mock()

    def test_error_on_chain(self, mock_client):
        self._setup_langchain_mock()
        try:
            uploaded = _capture_framework_trace(mock_client)
            handler = self._get_handler(mock_client)

            chain_id = uuid4()
            handler.on_chain_start({"name": "FailChain"}, {"input": "x"}, run_id=chain_id)
            handler.on_chain_error(ValueError("broke"), run_id=chain_id)

            events = uploaded["events"]
            error = find_event(events, "agent.error")
            assert error["payload"]["error"] == "broke"
            assert error["payload"]["status"] == "error"
        finally:
            self._teardown_langchain_mock()

    def test_null_serialized_handled(self, mock_client):
        self._setup_langchain_mock()
        try:
            uploaded = _capture_framework_trace(mock_client)
            handler = self._get_handler(mock_client)

            run_id = uuid4()
            handler.on_chain_start(None, {"input": "x"}, run_id=run_id)
            handler.on_chain_end({"output": "done"}, run_id=run_id)

            events = uploaded["events"]
            agent_input = find_event(events, "agent.input")
            assert agent_input["payload"]["name"] == "unknown"
        finally:
            self._teardown_langchain_mock()
