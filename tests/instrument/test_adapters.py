from __future__ import annotations

import sys
import types
import importlib
from uuid import uuid4
from unittest.mock import Mock


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

    def _get_handler(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.frameworks import langchain as lc_mod

        importlib.reload(lc_mod)
        return lc_mod.LangChainCallbackHandler(mock_client)

    def test_builds_span_tree(self, mock_client, capture_trace):
        self._setup_langchain_mock()
        try:
            handler = self._get_handler(mock_client, capture_trace)

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

            root = capture_trace["trace"][0]
            assert root["name"] == "RunnableSequence"
            assert root["kind"] == "chain"
            assert len(root["children"]) == 1

            llm = root["children"][0]
            assert llm["name"] == "ChatOpenAI"
            assert llm["kind"] == "llm"
            assert llm["output"] == "AI is..."
            assert llm["metadata"]["model"] == "gpt-4"
            assert llm["metadata"]["usage"]["total_tokens"] == 50
        finally:
            self._teardown_langchain_mock()

    def test_tracks_tools_and_retrievers(self, mock_client, capture_trace):
        self._setup_langchain_mock()
        try:
            handler = self._get_handler(mock_client, capture_trace)

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

            root = capture_trace["trace"][0]
            assert root["name"] == "Agent"
            assert len(root["children"]) == 2
            assert root["children"][0]["kind"] == "tool"
            assert root["children"][1]["kind"] == "retriever"
        finally:
            self._teardown_langchain_mock()

    def test_error_on_chain(self, mock_client, capture_trace):
        self._setup_langchain_mock()
        try:
            handler = self._get_handler(mock_client, capture_trace)

            chain_id = uuid4()
            handler.on_chain_start({"name": "FailChain"}, {"input": "x"}, run_id=chain_id)
            handler.on_chain_error(ValueError("broke"), run_id=chain_id)

            root = capture_trace["trace"][0]
            assert root["status"] == "error"
            assert root["error"] == "broke"
        finally:
            self._teardown_langchain_mock()

    def test_null_serialized_handled(self, mock_client, capture_trace):
        self._setup_langchain_mock()
        try:
            handler = self._get_handler(mock_client, capture_trace)

            run_id = uuid4()
            handler.on_chain_start(None, {"input": "x"}, run_id=run_id)
            handler.on_chain_end({"output": "done"}, run_id=run_id)

            root = capture_trace["trace"][0]
            assert root["name"] == "unknown"
            assert root["status"] == "ok"
        finally:
            self._teardown_langchain_mock()
