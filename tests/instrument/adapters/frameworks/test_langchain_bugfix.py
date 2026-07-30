from __future__ import annotations

from uuid import uuid4

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain import (
    LangChainCallbackHandler,
    _honest_run_name,
)

from .conftest import find_event, capture_framework_trace

# ---------------------------------------------------------------------------
# BUG-7: agent.error emissions must carry error_type == type(error).__name__,
# matching all 11 sibling framework adapters. A dashboard that groups/filters
# failures by error_type sees LangChain errors as an untyped blob otherwise.
# The four emission sites are on_chain_error / on_llm_error / on_tool_error /
# on_retriever_error. Each test drives the real callback and asserts the exact
# exception class name lands on the emitted agent.error payload.
# ---------------------------------------------------------------------------


class TestErrorTypeOnAgentError:
    def test_on_chain_error_real_runnable_carries_error_type(self, mock_client):
        """A RunnableLambda that raises drives the real on_chain_error callback;
        the emitted agent.error must record the concrete exception class name."""
        from langchain_core.runnables import RunnableLambda

        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        def boom(_x):
            raise ValueError("kaboom")

        try:
            RunnableLambda(boom).invoke({"q": "hi"}, config={"callbacks": [handler]})
        except ValueError:
            pass

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "kaboom"
        assert error["payload"]["status"] == "error"
        assert error["payload"]["error_type"] == "ValueError"

    def test_on_llm_error_carries_error_type(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        chain_id = uuid4()
        llm_id = uuid4()
        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "LLM"}, ["p"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_error(TimeoutError("timeout"), run_id=llm_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "timeout"
        assert error["payload"]["error_type"] == "TimeoutError"

    def test_on_tool_error_carries_error_type(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        chain_id = uuid4()
        tool_id = uuid4()
        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_tool_start({"name": "search"}, "q", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_error(KeyError("404"), run_id=tool_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error_type"] == "KeyError"

    def test_on_retriever_error_carries_error_type(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        chain_id = uuid4()
        ret_id = uuid4()
        handler.on_chain_start({"name": "Agent"}, {}, run_id=chain_id)
        handler.on_retriever_start({"name": "vs"}, "q", run_id=ret_id, parent_run_id=chain_id)
        handler.on_retriever_error(ConnectionError("down"), run_id=ret_id, parent_run_id=chain_id)
        handler.on_chain_end({}, run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error_type"] == "ConnectionError"


# ---------------------------------------------------------------------------
# FABRICATION (CRITICAL honesty): a bare `prompt | model` chain with NO
# developer-declared run_name must render honestly blank. LangChain surfaces
# the prompt component's get_name() class default ("ChatPromptTemplate") as the
# on_chain_start `name` kwarg for the prompt sub-run — that is plumbing, not a
# producer-declared agent. Accepting it fabricates an agent node. Only a genuine
# .with_config(run_name=...) may surface.
# ---------------------------------------------------------------------------


class TestNoFabricatedComponentAgent:
    def test_component_class_defaults_are_not_agent_identities(self):
        # LCEL prompt/parser/model component class defaults are plumbing, never
        # a producer-declared agent identity.
        for name in [
            "ChatPromptTemplate",
            "PromptTemplate",
            "HumanMessagePromptTemplate",
            "SystemMessagePromptTemplate",
            "StrOutputParser",
            "JsonOutputParser",
        ]:
            assert _honest_run_name(name) is None, f"{name!r} was fabricated as an agent identity"

    def test_bare_prompt_model_chain_stays_honestly_blank(self, mock_client):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
        model = FakeListChatModel(responses=["ok"])
        chain = prompt | model  # NO run_name declared
        chain.invoke({"q": "hi"}, config={"callbacks": [handler]})

        assert uploaded["events"], "expected a captured trace"
        for e in uploaded["events"]:
            payload = e.get("payload") or {}
            assert "agent_name" not in payload, (
                f"fabricated agent_name in {e['event_type']}: {payload.get('agent_name')!r}"
            )

    def test_declared_run_name_on_prompt_model_chain_surfaces(self, mock_client):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
        model = FakeListChatModel(responses=["ok"])
        chain = (prompt | model).with_config(run_name="my-fraud-agent")
        chain.invoke({"q": "hi"}, config={"callbacks": [handler]})

        # A genuine developer-declared run_name is the ONE identity that surfaces.
        named = [e for e in uploaded["events"] if (e.get("payload") or {}).get("agent_name") == "my-fraud-agent"]
        assert named, "developer-declared run_name did not surface as agent_name"
        # And no component class default leaked through as an agent identity.
        for e in uploaded["events"]:
            an = (e.get("payload") or {}).get("agent_name")
            if an is not None:
                assert an == "my-fraud-agent", f"unexpected agent_name {an!r} in {e['event_type']}"
