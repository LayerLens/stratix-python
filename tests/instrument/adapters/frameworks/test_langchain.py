from __future__ import annotations

from uuid import uuid4
from unittest.mock import Mock

from langchain_core.callbacks import BaseCallbackHandler

from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

from .conftest import capture_framework_trace, find_event, find_events


# ---------------------------------------------------------------------------
# Sanity: real base class
# ---------------------------------------------------------------------------


class TestBaseClass:
    def test_inherits_langchain_base(self):
        assert issubclass(LangChainCallbackHandler, BaseCallbackHandler)

    def test_name(self):
        handler = LangChainCallbackHandler(Mock())
        assert handler.name == "langchain"


# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_chain_lifecycle(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "RunnableSequence", "id": ["RunnableSequence"]},
            {"question": "What is AI?"},
            run_id=chain_id,
        )
        handler.on_chain_end({"output": "AI is..."}, run_id=chain_id)

        events = uploaded["events"]
        agent_input = find_event(events, "agent.input")
        assert agent_input["payload"]["name"] == "RunnableSequence"
        assert agent_input["payload"]["input"] == {"question": "What is AI?"}

        agent_output = find_event(events, "agent.output")
        assert agent_output["payload"]["status"] == "ok"

    def test_llm_lifecycle(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            {"name": "Chain"}, {"input": "x"}, run_id=chain_id,
        )
        handler.on_llm_start(
            {"name": "ChatOpenAI", "id": ["ChatOpenAI"]},
            ["What is AI?"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        llm_response = Mock()
        llm_response.generations = [[Mock(text="AI is...")]]
        llm_response.llm_output = {
            "token_usage": {"total_tokens": 50},
            "model_name": "gpt-4",
        }
        handler.on_llm_end(llm_response, run_id=llm_id)
        handler.on_chain_end({"output": "AI is..."}, run_id=chain_id)

        events = uploaded["events"]

        model_invokes = find_events(events, "model.invoke")
        assert len(model_invokes) >= 1
        # Start event has name and messages
        start_invoke = [m for m in model_invokes if m["payload"].get("name") == "ChatOpenAI"]
        assert len(start_invoke) == 1
        # End event has model and output
        end_invoke = [m for m in model_invokes if m["payload"].get("model") == "gpt-4"]
        assert len(end_invoke) == 1
        assert end_invoke[0]["payload"]["output_message"] == "AI is..."

        cost = find_event(events, "cost.record")
        assert cost["payload"]["total_tokens"] == 50

    def test_chat_model_start(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        chat_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        msg = Mock()
        msg.type = "human"
        msg.content = "Hello"
        handler.on_chat_model_start(
            {"name": "ChatAnthropic"},
            [[msg]],
            run_id=chat_id,
            parent_run_id=chain_id,
        )
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        assert invoke["payload"]["name"] == "ChatAnthropic"
        assert invoke["payload"]["messages"] == [[{"type": "human", "content": "Hello"}]]


# ---------------------------------------------------------------------------
# Tool and retriever events
# ---------------------------------------------------------------------------


class TestToolsAndRetrievers:
    def test_tool_lifecycle(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_tool_start(
            {"name": "search"}, "query text",
            run_id=tool_id, parent_run_id=chain_id,
        )
        handler.on_tool_end("search results", run_id=tool_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["name"] == "search"
        assert tool_call["payload"]["input"] == "query text"

        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["output"] == "search results"

    def test_retriever_lifecycle(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_retriever_start(
            {"name": "vectorstore"}, "query",
            run_id=ret_id, parent_run_id=chain_id,
        )
        docs = [Mock(page_content="doc text", metadata={"source": "a.txt"})]
        handler.on_retriever_end(docs, run_id=ret_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["name"] == "vectorstore"

        tool_result = find_event(events, "tool.result")
        output = tool_result["payload"]["output"]
        assert output[0]["page_content"] == "doc text"
        assert output[0]["metadata"] == {"source": "a.txt"}

    def test_combined_tools_and_retrievers(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        tool_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_tool_start({"name": "search"}, "q", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_end("results", run_id=tool_id)
        handler.on_retriever_start({"name": "vs"}, "q", run_id=ret_id, parent_run_id=chain_id)
        handler.on_retriever_end([Mock(page_content="d", metadata={})], run_id=ret_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        assert len(find_events(events, "tool.call")) == 2
        assert len(find_events(events, "tool.result")) == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_chain_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start({"name": "FailChain"}, {"input": "x"}, run_id=chain_id)
        handler.on_chain_error(ValueError("broke"), run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "broke"
        assert error["payload"]["status"] == "error"

    def test_llm_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["prompt"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_error(RuntimeError("timeout"), run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "timeout"

    def test_tool_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_tool_start({"name": "search"}, "q", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_error(RuntimeError("404"), run_id=tool_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "404"

    def test_retriever_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_retriever_start({"name": "vs"}, "q", run_id=ret_id, parent_run_id=chain_id)
        handler.on_retriever_error(ConnectionError("down"), run_id=ret_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "down"


# ---------------------------------------------------------------------------
# Parent-child span relationships
# ---------------------------------------------------------------------------


class TestSpanRelationships:
    def test_llm_parent_is_chain(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start(
            {"name": "LLM"}, ["prompt"],
            run_id=llm_id, parent_run_id=chain_id,
        )
        llm_response = Mock()
        llm_response.generations = [[Mock(text="out")]]
        llm_response.llm_output = {}
        handler.on_llm_end(llm_response, run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        chain_input = find_event(events, "agent.input")
        llm_invoke = [e for e in find_events(events, "model.invoke") if e["payload"].get("name") == "LLM"][0]
        assert llm_invoke["parent_span_id"] == chain_input["span_id"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_null_serialized(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        run_id = uuid4()
        handler.on_chain_start(None, {"input": "x"}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "unknown"

    def test_empty_serialized_id(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        run_id = uuid4()
        handler.on_chain_start({"id": ["FallbackName"]}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "FallbackName"

    def test_llm_end_no_output(self, mock_client):
        """LLM response with no generations should not crash."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["p"], run_id=llm_id, parent_run_id=chain_id)

        empty_response = Mock()
        empty_response.generations = []
        empty_response.llm_output = None
        handler.on_llm_end(empty_response, run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        # Should complete without error — no model.invoke end event since no output/model


# ---------------------------------------------------------------------------
# adapter_info
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def test_info(self):
        handler = LangChainCallbackHandler(Mock())
        info = handler.adapter_info()
        assert info.name == "langchain"
        assert info.adapter_type == "framework"
