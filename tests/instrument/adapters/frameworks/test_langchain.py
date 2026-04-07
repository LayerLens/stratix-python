from __future__ import annotations

from uuid import uuid4
from unittest.mock import Mock

from langchain_core.callbacks import BaseCallbackHandler

from layerlens.instrument._capture_config import CaptureConfig
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

    def test_adapter_info(self):
        handler = LangChainCallbackHandler(Mock())
        info = handler.adapter_info()
        assert info.name == "langchain"
        assert info.adapter_type == "framework"
        assert info.connected is False


# ---------------------------------------------------------------------------
# Chain lifecycle
# ---------------------------------------------------------------------------


class TestChainLifecycle:
    def test_chain_emits_input_and_output(self, mock_client):
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
        assert agent_output["payload"]["output"] == {"output": "AI is..."}

    def test_chain_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start({"name": "FailChain"}, {"input": "x"}, run_id=chain_id)
        handler.on_chain_error(ValueError("broke"), run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "broke"
        assert error["payload"]["status"] == "error"


# ---------------------------------------------------------------------------
# LLM lifecycle — single merged model.invoke
# ---------------------------------------------------------------------------


def _make_llm_response(
    text: str = "AI is...",
    model_name: str = "gpt-4",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Mock:
    resp = Mock()
    resp.generations = [[Mock(text=text)]]
    resp.llm_output = {
        "model_name": model_name,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


class TestLLMLifecycle:
    def test_single_model_invoke_with_merged_data(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {"input": "x"}, run_id=chain_id)
        handler.on_llm_start(
            {"name": "ChatOpenAI"},
            ["What is AI?"],
            run_id=llm_id, parent_run_id=chain_id,
        )
        handler.on_llm_end(_make_llm_response(), run_id=llm_id)
        handler.on_chain_end({"output": "AI is..."}, run_id=chain_id)

        events = uploaded["events"]
        model_invokes = find_events(events, "model.invoke")
        # Single event, not two
        assert len(model_invokes) == 1

        invoke = model_invokes[0]
        assert invoke["payload"]["name"] == "ChatOpenAI"
        assert invoke["payload"]["model"] == "gpt-4"
        assert invoke["payload"]["messages"] == ["What is AI?"]
        assert invoke["payload"]["output_message"] == "AI is..."
        assert invoke["payload"]["latency_ms"] >= 0

    def test_normalized_token_fields(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["p"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_end(_make_llm_response(), run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        assert invoke["payload"]["tokens_prompt"] == 100
        assert invoke["payload"]["tokens_completion"] == 50
        assert invoke["payload"]["tokens_total"] == 150

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_prompt"] == 100
        assert cost["payload"]["tokens_completion"] == 50
        assert cost["payload"]["tokens_total"] == 150
        assert cost["payload"]["model"] == "gpt-4"

    def test_chat_model_start_serializes_messages(self, mock_client):
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
            run_id=chat_id, parent_run_id=chain_id,
        )
        handler.on_llm_end(
            _make_llm_response(text="Hi!", model_name="claude-3"),
            run_id=chat_id,
        )
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        assert invoke["payload"]["name"] == "ChatAnthropic"
        assert invoke["payload"]["messages"] == [[{"type": "human", "content": "Hello"}]]
        assert invoke["payload"]["output_message"] == "Hi!"

    def test_llm_error_emits_model_invoke_with_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["prompt"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_error(RuntimeError("timeout"), run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        assert invoke["payload"]["error"] == "timeout"
        assert invoke["payload"]["latency_ms"] >= 0

        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "timeout"


# ---------------------------------------------------------------------------
# CaptureConfig content gating
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_capture_content_false_strips_inputs_and_messages(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {"secret": "data"}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["secret prompt"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_end(_make_llm_response(text="secret reply"), run_id=llm_id)
        handler.on_chain_end({"output": "secret"}, run_id=chain_id)

        events = uploaded["events"]

        # Chain events should not contain content
        agent_input = find_event(events, "agent.input")
        assert "input" not in agent_input["payload"]
        agent_output = find_event(events, "agent.output")
        assert "output" not in agent_output["payload"]

        # Model invoke should not contain messages or output
        invoke = find_event(events, "model.invoke")
        assert "messages" not in invoke["payload"]
        assert "output_message" not in invoke["payload"]
        # But structural fields are still present
        assert invoke["payload"]["name"] == "LLM"

    def test_capture_content_false_strips_tool_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))

        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_tool_start({"name": "search"}, "secret query", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_end("secret results", run_id=tool_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert "input" not in tool_call["payload"]
        assert tool_call["payload"]["name"] == "search"

        tool_result = find_event(events, "tool.result")
        assert "output" not in tool_result["payload"]

    def test_capture_content_false_strips_retriever_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))

        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_retriever_start({"name": "vs"}, "secret query", run_id=ret_id, parent_run_id=chain_id)
        docs = [Mock(page_content="secret doc", metadata={"source": "a.txt"})]
        handler.on_retriever_end(docs, run_id=ret_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert "input" not in tool_call["payload"]

        tool_result = find_event(events, "tool.result")
        assert "output" not in tool_result["payload"]


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
# Agent action / finish callbacks
# ---------------------------------------------------------------------------


class TestAgentCallbacks:
    def test_agent_action_emits_input(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        agent_id = uuid4()

        handler.on_chain_start({"name": "AgentExecutor"}, {}, run_id=chain_id)

        action = Mock()
        action.tool = "search"
        action.tool_input = "what is AI"
        action.log = "Thought: I need to search"
        handler.on_agent_action(action, run_id=agent_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        inputs = [e for e in find_events(events, "agent.input") if e["payload"].get("tool") == "search"]
        assert len(inputs) == 1
        assert inputs[0]["payload"]["tool_input"] == "what is AI"
        assert inputs[0]["payload"]["log"] == "Thought: I need to search"

    def test_agent_finish_emits_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        agent_id = uuid4()

        handler.on_chain_start({"name": "AgentExecutor"}, {}, run_id=chain_id)

        finish = Mock()
        finish.return_values = {"output": "AI is artificial intelligence"}
        finish.log = "Final Answer: AI is artificial intelligence"
        handler.on_agent_finish(finish, run_id=agent_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        outputs = [e for e in find_events(events, "agent.output") if e["payload"].get("log")]
        assert len(outputs) == 1
        assert outputs[0]["payload"]["output"] == {"output": "AI is artificial intelligence"}

    def test_agent_action_respects_capture_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))

        chain_id = uuid4()
        agent_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        action = Mock()
        action.tool = "secret_tool"
        action.tool_input = "secret input"
        action.log = "secret reasoning"
        handler.on_agent_action(action, run_id=agent_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        inputs = [e for e in find_events(events, "agent.input") if e["payload"].get("tool") == "secret_tool"]
        assert len(inputs) == 1
        assert "tool_input" not in inputs[0]["payload"]
        assert "log" not in inputs[0]["payload"]


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
        handler.on_llm_end(_make_llm_response(), run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        chain_input = find_event(events, "agent.input")
        invoke = find_event(events, "model.invoke")
        assert invoke["parent_span_id"] == chain_input["span_id"]


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

        # Should emit model.invoke with name but no output_message
        invoke = find_event(uploaded["events"], "model.invoke")
        assert invoke["payload"]["name"] == "LLM"
        assert "output_message" not in invoke["payload"]

    def test_llm_end_without_start(self, mock_client):
        """on_llm_end without a preceding on_llm_start should not crash."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_end(_make_llm_response(), run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        # Should still emit model.invoke from the response data
        invoke = find_event(uploaded["events"], "model.invoke")
        assert invoke["payload"]["model"] == "gpt-4"
