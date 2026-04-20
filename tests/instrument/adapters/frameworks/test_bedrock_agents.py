"""Tests for the Bedrock Agents adapter using real boto3 clients.

Uses ``botocore.stub.Stubber`` to intercept API calls (no real HTTP
requests) while still exercising the real boto3 event-hook system.
Because the Stubber validates responses against the AWS service model
and doesn't allow extra keys like ``outputText`` or ``trace``, we
inject test data via a separate event hook registered *before* the
adapter's hooks, which mutates the ``parsed`` dict in-place.

This gives us:
- Real client creation (``boto3.client("bedrock-agent-runtime")``)
- Real event hook registration / unregistration
- Real hook dispatch (provide-client-params, after-call)
- Real adapter lifecycle (connect / disconnect)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import Stubber  # noqa: E402

import layerlens.instrument.adapters.frameworks.bedrock_agents as _mod  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import (  # noqa: E402
    BedrockAgentsAdapter,
    _collect_steps,
    _extract_completion,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal valid Stubber response (compliant with the service model)
# ---------------------------------------------------------------------------


def _stub_response() -> Dict[str, Any]:
    """Return a fresh minimal valid InvokeAgent response for the Stubber."""
    return {
        "completion": {},
        "contentType": "text/plain",
        "sessionId": "sess-1",
        "memoryId": "mem-1",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_boto_client():
    """Create a real ``bedrock-agent-runtime`` client (no credentials needed)."""
    return boto3.client("bedrock-agent-runtime", region_name="us-east-1")


def _make_injector(
    *,
    output_text: Optional[str] = None,
    output_nested: Optional[str] = None,
    trace_steps: Optional[List[Dict[str, Any]]] = None,
    nested_trace_steps: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
):
    """Return an ``after-call`` hook that injects test data into ``parsed``.

    Registered *before* the adapter's hook so that the adapter sees the
    injected keys when its own ``_after_invoke`` fires.
    """

    def _inject(**kwargs: Any) -> None:
        parsed = kwargs.get("parsed", {})
        if output_text is not None:
            parsed["outputText"] = output_text
        if output_nested is not None:
            parsed["output"] = {"text": output_nested}
        if trace_steps is not None:
            parsed["trace"] = {"steps": trace_steps}
        if nested_trace_steps is not None:
            parsed.setdefault("trace", {})["trace"] = {"orchestrationTrace": {"steps": nested_trace_steps}}
        if session_id is not None:
            parsed["sessionId"] = session_id

    return _inject


def _setup(
    mock_client: Any,
    *,
    config: Optional[CaptureConfig] = None,
    injector: Any = None,
) -> tuple:
    """Wire up a real boto3 client + Stubber + adapter.

    Returns ``(adapter, uploaded, boto_client, stubber)``.
    """
    uploaded = capture_framework_trace(mock_client)
    boto = _make_boto_client()

    # Register injector BEFORE connecting adapter so it fires first
    if injector is not None:
        boto.meta.events.register(
            "after-call.bedrock-agent-runtime.InvokeAgent",
            injector,
        )

    adapter = BedrockAgentsAdapter(mock_client, capture_config=config)
    adapter.connect(target=boto)

    stubber = Stubber(boto)
    stubber.activate()
    return adapter, uploaded, boto, stubber


def _call_invoke(
    boto_client: Any,
    stubber: Stubber,
    *,
    agent_id: str = "agent-1",
    alias_id: str = "alias-1",
    session_id: str = "sess-1",
    input_text: str = "hello",
    stub_response: Optional[Dict[str, Any]] = None,
) -> Any:
    """Add a Stubber response and invoke the agent through the real client."""
    # Always a fresh dict — the injector hook mutates ``parsed`` in-place
    # and ``parsed`` IS the stubber's response dict.
    resp = stub_response or _stub_response()
    stubber.add_response("invoke_agent", resp)
    return boto_client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=input_text,
    )


def _invoke(
    adapter: BedrockAgentsAdapter,
    boto_client: Any,
    stubber: Stubber,
    *,
    agent_id: str = "agent-1",
    alias_id: str = "alias-1",
    session_id: str = "sess-1",
    input_text: str = "hello",
    stub_response: Optional[Dict[str, Any]] = None,
) -> Any:
    """Shorthand: stub + call invoke_agent."""
    return _call_invoke(
        boto_client,
        stubber,
        agent_id=agent_id,
        alias_id=alias_id,
        session_id=session_id,
        input_text=input_text,
        stub_response=stub_response,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_registers_hooks(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)

        # Verify hooks fire by making a stubbed call
        stubber = Stubber(boto)
        stubber.activate()
        stubber.add_response("invoke_agent", _stub_response())

        fired = {"before": False, "after": False}

        def check_before(**kw):
            fired["before"] = True

        def check_after(**kw):
            fired["after"] = True

        boto.meta.events.register(_mod._BEFORE_HOOK, check_before)
        boto.meta.events.register(_mod._AFTER_HOOK, check_after)

        boto.invoke_agent(agentId="a1", agentAliasId="al1", sessionId="sess-1", inputText="hi")

        assert fired["before"]
        assert fired["after"]
        adapter.disconnect()

    def test_disconnect_unregisters_hooks(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        adapter.disconnect()

        # After disconnect, adapter hooks should not fire.
        # We can verify by calling invoke_agent — the adapter's _before_invoke
        # checks self._connected and returns early, but more importantly the
        # hooks are unregistered from the real event system.
        stubber = Stubber(boto)
        stubber.activate()
        stubber.add_response("invoke_agent", _stub_response())

        # No collector active, no events emitted, no crash
        boto.invoke_agent(agentId="a1", agentAliasId="al1", sessionId="sess-1", inputText="hi")

    def test_connect_returns_target(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        result = adapter.connect(target=boto)
        assert result is boto
        adapter.disconnect()

    def test_connect_without_target_raises(self, mock_client):
        with pytest.raises(ValueError, match="requires a bedrock-agent-runtime"):
            BedrockAgentsAdapter(mock_client).connect(target=None)

    def test_adapter_info(self, mock_client):
        adapter = BedrockAgentsAdapter(mock_client)
        info = adapter.adapter_info()
        assert info.name == "bedrock_agents"
        assert info.adapter_type == "framework"
        assert not info.connected

    def test_connected_flag(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        assert not adapter.adapter_info().connected
        adapter.connect(target=boto)
        assert adapter.adapter_info().connected
        adapter.disconnect()
        assert not adapter.adapter_info().connected

    def test_raises_when_boto3_missing(self, mock_client, monkeypatch):
        monkeypatch.setattr(_mod, "_HAS_BOTO3", False)
        with pytest.raises(ImportError, match="bedrock"):
            BedrockAgentsAdapter(mock_client).connect(target=_make_boto_client())

    def test_disconnect_tolerates_unregister_failure(self, mock_client, monkeypatch):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)

        # Sabotage the event system to simulate failure
        real_unregister = boto.meta.events.unregister

        def exploding_unregister(*a, **kw):
            raise RuntimeError("boom")

        boto.meta.events.unregister = exploding_unregister

        # Should not raise
        adapter.disconnect()
        assert not adapter.is_connected

        # Restore so GC doesn't explode
        boto.meta.events.unregister = real_unregister


# ---------------------------------------------------------------------------
# Agent I/O
# ---------------------------------------------------------------------------


class TestAgentIO:
    def test_input_and_output(self, mock_client):
        injector = _make_injector(output_text="Sunny")
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber, input_text="What is the weather?")
        adapter.disconnect()

        events = uploaded["events"]
        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_id"] == "agent-1"
        assert inp["payload"]["session_id"] == "sess-1"
        assert inp["payload"]["input"] == "What is the weather?"
        assert inp["span_name"] == "bedrock.invoke_agent"

        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "Sunny"
        assert out["payload"]["latency_ms"] is not None
        assert out["span_name"] == "bedrock.invoke_agent"

    def test_content_gating(self, mock_client):
        injector = _make_injector(output_text="classified")
        adapter, uploaded, boto, stubber = _setup(
            mock_client,
            config=CaptureConfig(capture_content=False),
            injector=injector,
        )

        _invoke(adapter, boto, stubber, input_text="secret")
        adapter.disconnect()

        events = uploaded["events"]
        assert "input" not in find_event(events, "agent.input")["payload"]
        assert "output" not in find_event(events, "agent.output")["payload"]

    def test_nested_output_extraction(self, mock_client):
        injector = _make_injector(output_nested="nested text")
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"] == "nested text"

    def test_noop_when_disconnected(self, mock_client):
        adapter = BedrockAgentsAdapter(mock_client)
        # Not connected — calling the hook directly should be a no-op
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        assert not mock_client.traces.upload.called


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


class TestEnvironmentConfig:
    def test_emitted_once_per_agent(self, mock_client):
        injector = _make_injector(output_text="ok")
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber, agent_id="a1")
        _invoke(adapter, boto, stubber, agent_id="a1")
        adapter.disconnect()

        configs = find_events(uploaded["events"], "environment.config")
        assert len(configs) == 1
        assert configs[0]["payload"]["agent_id"] == "a1"

    def test_emitted_per_unique_agent(self, mock_client):
        injector = _make_injector(output_text="ok")
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber, agent_id="a1")
        _invoke(adapter, boto, stubber, agent_id="a2")
        adapter.disconnect()

        configs = find_events(uploaded["events"], "environment.config")
        assert len(configs) == 2

    def test_enable_trace_flag(self, mock_client):
        """enable_trace comes from the request params, not the response."""
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=_make_injector(output_text="ok"))

        # enable_trace is in the request params; the provide-client-params hook
        # receives it. But the adapter reads it from params dict. We pass it
        # as a kwarg to invoke_agent, which boto3 puts into params.
        # Unfortunately enableTrace is not a real InvokeAgent param in the model.
        # The adapter reads it from kwargs["params"]["enableTrace"], which is
        # populated by boto3 from the actual API call parameters.
        # Since enableTrace IS a real param, this works through the real client.
        stubber.add_response("invoke_agent", _stub_response())
        boto.invoke_agent(
            agentId="a1",
            agentAliasId="alias-1",
            sessionId="sess-1",
            inputText="hi",
            enableTrace=True,
        )
        adapter.disconnect()

        cfg = find_event(uploaded["events"], "environment.config")
        assert cfg["payload"]["enable_trace"] is True


# ---------------------------------------------------------------------------
# Trace steps — action groups
# ---------------------------------------------------------------------------


class TestActionGroup:
    def test_action_group_emitted(self, mock_client):
        injector = _make_injector(
            output_text="done",
            trace_steps=[
                {
                    "type": "ACTION_GROUP",
                    "actionGroupName": "MyAction",
                    "actionGroupInput": {"key": "val"},
                    "actionGroupInvocationOutput": {"output": "result"},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "MyAction"
        assert tc["payload"]["tool_type"] == "action_group"
        assert tc["payload"]["input"] == {"key": "val"}
        assert tc["payload"]["output"] == "result"
        assert tc["span_name"] == "bedrock.action_group"

    def test_action_group_content_gating(self, mock_client):
        injector = _make_injector(
            output_text="done",
            trace_steps=[
                {
                    "type": "ACTION_GROUP",
                    "actionGroupName": "A",
                    "actionGroupInput": "secret",
                    "actionGroupInvocationOutput": {"output": "classified"},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(
            mock_client,
            config=CaptureConfig(capture_content=False),
            injector=injector,
        )

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        tc = find_event(uploaded["events"], "tool.call")
        assert "input" not in tc["payload"]
        assert "output" not in tc["payload"]


# ---------------------------------------------------------------------------
# Trace steps — knowledge base
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def test_knowledge_base_emitted(self, mock_client):
        injector = _make_injector(
            output_text="found it",
            trace_steps=[
                {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseId": "kb-99",
                    "knowledgeBaseLookupInput": "search query",
                    "knowledgeBaseLookupOutput": {"retrievedReferences": [{"text": "ref1"}]},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "kb-99"
        assert tc["payload"]["tool_type"] == "knowledge_base_retrieval"
        assert tc["span_name"] == "bedrock.knowledge_base"


# ---------------------------------------------------------------------------
# Trace steps — model invocation
# ---------------------------------------------------------------------------


class TestModelInvocation:
    def test_model_invoke_with_tokens(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "anthropic.claude-3",
                    "modelInvocationOutput": {"usage": {"inputTokens": 100, "outputTokens": 50}},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        events = uploaded["events"]
        me = find_event(events, "model.invoke")
        assert me["payload"]["model"] == "anthropic.claude-3"
        assert me["payload"]["tokens_prompt"] == 100
        assert me["payload"]["tokens_completion"] == 50
        assert me["payload"]["tokens_total"] == 150
        assert me["span_name"] == "bedrock.model"

    def test_cost_record_emitted(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "anthropic.claude-3",
                    "modelInvocationOutput": {"usage": {"inputTokens": 10, "outputTokens": 5}},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 15
        assert cost["payload"]["model"] == "anthropic.claude-3"

    def test_no_tokens_no_cost(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "anthropic.claude-3",
                    "modelInvocationOutput": {},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        assert len(find_events(uploaded["events"], "cost.record")) == 0

    def test_cost_parented_to_model_span(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "m",
                    "modelInvocationOutput": {"usage": {"inputTokens": 1, "outputTokens": 1}},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        events = uploaded["events"]
        me = find_event(events, "model.invoke")
        cost = find_event(events, "cost.record")
        assert cost["span_id"] == me["span_id"]


# ---------------------------------------------------------------------------
# Trace steps — collaborator handoff
# ---------------------------------------------------------------------------


class TestCollaboratorHandoff:
    def test_handoff_emitted(self, mock_client):
        injector = _make_injector(
            output_text="done",
            trace_steps=[
                {
                    "type": "AGENT_COLLABORATOR",
                    "supervisorAgentId": "sup-1",
                    "collaboratorAgentId": "collab-2",
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        h = find_event(uploaded["events"], "agent.handoff")
        assert h["payload"]["from_agent"] == "sup-1"
        assert h["payload"]["to_agent"] == "collab-2"
        assert h["payload"]["reason"] == "supervisor_delegation"
        assert h["span_name"] == "bedrock.handoff"


# ---------------------------------------------------------------------------
# Full invocation (multi-step trace)
# ---------------------------------------------------------------------------


class TestFullInvocation:
    def test_rag_pipeline(self, mock_client):
        injector = _make_injector(
            output_text="AI is...",
            trace_steps=[
                {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseId": "kb-1",
                    "knowledgeBaseLookupInput": "What is AI?",
                    "knowledgeBaseLookupOutput": {"retrievedReferences": [{"text": "doc"}]},
                },
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "anthropic.claude-3",
                    "modelInvocationOutput": {"usage": {"inputTokens": 200, "outputTokens": 100}},
                },
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber, input_text="What is AI?")
        adapter.disconnect()

        events = uploaded["events"]
        assert len(find_events(events, "agent.input")) == 1
        assert len(find_events(events, "agent.output")) == 1
        assert len(find_events(events, "tool.call")) == 1  # KB retrieval
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "cost.record")) == 1

    def test_multiple_invocations(self, mock_client):
        """Two separate invoke_agent calls through the same client."""
        injector = _make_injector(output_text="ok")
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber, agent_id="a1", input_text="q1")
        _invoke(adapter, boto, stubber, agent_id="a1", input_text="q2")
        adapter.disconnect()

        events = uploaded["events"]
        inputs = find_events(events, "agent.input")
        outputs = find_events(events, "agent.output")
        assert len(inputs) == 2
        assert len(outputs) == 2


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id_within_invocation(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {
                    "type": "MODEL_INVOCATION",
                    "foundationModel": "m",
                    "modelInvocationOutput": {"usage": {"inputTokens": 1, "outputTokens": 1}},
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        trace_ids = {e["trace_id"] for e in uploaded["events"]}
        assert len(trace_ids) == 1

    def test_monotonic_sequence_ids(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[
                {"type": "ACTION_GROUP", "actionGroupName": "a"},
                {"type": "MODEL_INVOCATION", "foundationModel": "m", "modelInvocationOutput": {}},
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        seq = [e["sequence_id"] for e in uploaded["events"]]
        assert seq == sorted(seq)

    def test_span_hierarchy(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            trace_steps=[{"type": "ACTION_GROUP", "actionGroupName": "a"}],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        events = uploaded["events"]
        root = find_event(events, "agent.input")["span_id"]
        tc = find_event(events, "tool.call")
        assert tc["parent_span_id"] == root

    def test_nested_orchestration_trace_path(self, mock_client):
        injector = _make_injector(
            output_text="ok",
            nested_trace_steps=[
                {
                    "type": "ACTION_GROUP",
                    "actionGroupName": "Nested",
                }
            ],
        )
        adapter, uploaded, boto, stubber = _setup(mock_client, injector=injector)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "Nested"


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_before_invoke_survives_bad_params(self, mock_client):
        """Calling _before_invoke with missing/bad params should not raise."""
        adapter, _, boto, _ = _setup(mock_client)
        adapter._before_invoke()
        adapter._before_invoke(params=None)
        adapter.disconnect()

    def test_after_invoke_survives_bad_parsed(self, mock_client):
        """Calling _after_invoke with missing/bad parsed should not raise."""
        adapter, _, boto, _ = _setup(mock_client)
        adapter._after_invoke()
        adapter._after_invoke(parsed=None)
        adapter._after_invoke(parsed={"trace": "not_a_dict"})
        adapter.disconnect()

    def test_invoke_with_empty_response(self, mock_client):
        """A stubbed call with no injected data should not crash the adapter."""
        adapter, uploaded, boto, stubber = _setup(mock_client)

        _invoke(adapter, boto, stubber)
        adapter.disconnect()

        # Should still get agent.input and agent.output (output will be None/missing)
        events = uploaded["events"]
        assert len(find_events(events, "agent.input")) == 1
        assert len(find_events(events, "agent.output")) == 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_completion_output_text(self):
        assert _extract_completion({"outputText": "hello"}) == "hello"

    def test_extract_completion_nested(self):
        assert _extract_completion({"output": {"text": "nested"}}) == "nested"

    def test_extract_completion_none(self):
        assert _extract_completion({}) is None

    def test_collect_steps_flat(self):
        steps = _collect_steps({"trace": {"steps": [{"type": "A"}]}})
        assert len(steps) == 1

    def test_collect_steps_nested(self):
        steps = _collect_steps(
            {
                "trace": {"trace": {"orchestrationTrace": {"steps": [{"type": "B"}]}}},
            }
        )
        assert len(steps) == 1

    def test_collect_steps_bad_trace(self):
        assert _collect_steps({"trace": "not_dict"}) == []
        assert _collect_steps({}) == []
