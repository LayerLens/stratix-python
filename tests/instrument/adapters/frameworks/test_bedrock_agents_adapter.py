"""Unit tests for the AWS Bedrock Agents framework adapter.

Mocked at the SDK shape level — no real ``boto3`` runtime needed.
The adapter integrates via boto3 event hooks: ``client.meta.events.register(...)``.

After the typed-event migration (PR #129 follow-up — bundle 4) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests that
want to assert against the model surface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.bedrock_agents import (
    ADAPTER_CLASS,
    BedrockAgentsAdapter,
    instrument_client,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        # Hold strong references to the original typed payloads for
        # the subset of tests that want to assert against the model
        # surface (e.g. ``isinstance(payload, ToolCallEvent)``). The
        # dict view lives on ``events`` and is what most assertions
        # read.
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # Two-arg legacy path: ``emit(event_type, payload_dict)``.
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        # Single-arg typed path: ``emit(payload_model[, privacy_level])``.
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


class _FakeEventSystem:
    """Mimics boto3 client.meta.events register/unregister."""

    def __init__(self) -> None:
        self.handlers: Dict[str, List[Callable[..., Any]]] = {}
        self.unregistered: List[Tuple[str, Callable[..., Any]]] = []

    def register(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def unregister(self, event: str, handler: Callable[..., Any]) -> None:
        self.unregistered.append((event, handler))
        if event in self.handlers and handler in self.handlers[event]:
            self.handlers[event].remove(handler)


class _FakeClient:
    """Mimics a boto3 bedrock-agent-runtime client."""

    def __init__(self) -> None:
        self.meta = _FakeMeta()


class _FakeMeta:
    def __init__(self) -> None:
        self.events = _FakeEventSystem()


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is BedrockAgentsAdapter


def test_lifecycle() -> None:
    a = BedrockAgentsAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = BedrockAgentsAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "bedrock_agents"
    assert info.name == "BedrockAgentsAdapter"
    health = a.health_check()
    assert health.framework_name == "bedrock_agents"


def test_instrument_client_registers_event_hooks() -> None:
    adapter = BedrockAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    client = _FakeClient()
    adapter.instrument_client(client)

    handlers = client.meta.events.handlers
    assert "provide-client-params.bedrock-agent-runtime.InvokeAgent" in handlers
    assert "after-call.bedrock-agent-runtime.InvokeAgent" in handlers


def test_disconnect_unregisters_event_hooks() -> None:
    adapter = BedrockAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    adapter.disconnect()
    assert len(client.meta.events.unregistered) == 2


def test_before_invoke_emits_input_event() -> None:
    """Typed AgentInputEvent: input lives on payload.content.message;
    Bedrock-specific provenance lives on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    # Simulate the boto3 'provide-client-params' event firing.
    adapter._before_invoke_agent(
        params={
            "agentId": "agent-123",
            "agentAliasId": "alias-1",
            "sessionId": "sess-1",
            "inputText": "hello",
            "enableTrace": True,
        }
    )

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types

    inp = next(e for e in stratix.events if e["event_type"] == "agent.input")
    payload = inp["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["message"] == "hello"
    assert payload["content"]["role"] == "human"
    metadata = payload["content"]["metadata"]
    assert metadata["framework"] == "bedrock_agents"
    assert metadata["agent_id"] == "agent-123"
    assert metadata["session_id"] == "sess-1"
    assert metadata["enable_trace"] is True

    # environment.config carries the agent config on attributes.
    cfg = next(e for e in stratix.events if e["event_type"] == "environment.config")
    cfg_payload = cfg["payload"]
    assert cfg_payload["layer"] == "L4a"
    assert cfg_payload["environment"]["type"] == "cloud"
    assert cfg_payload["environment"]["attributes"]["agent_id"] == "agent-123"
    assert cfg_payload["environment"]["attributes"]["agent_alias_id"] == "alias-1"


def test_after_invoke_emits_output_and_processes_trace() -> None:
    """Typed AgentOutputEvent + ToolCallEvent + ModelInvokeEvent +
    CostRecordEvent + AgentHandoffEvent for a complete Bedrock response.
    """
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Simulate the after-call event with a parsed response.
    adapter._after_invoke_agent(
        parsed={
            "outputText": "the answer is 42",
            "sessionId": "sess-1",
            "trace": {
                "steps": [
                    {
                        "type": "ACTION_GROUP",
                        "actionGroupName": "calc",
                        "actionGroupInput": {"x": 1},
                        "actionGroupInvocationOutput": {"output": "ok"},
                    },
                    {
                        "type": "MODEL_INVOCATION",
                        "foundationModel": "anthropic.claude-v2",
                        "modelInvocationOutput": {"usage": {"inputTokens": 100, "outputTokens": 50}},
                    },
                    {
                        "type": "AGENT_COLLABORATOR",
                        "supervisorAgentId": "sup-1",
                        "collaboratorAgentId": "col-1",
                    },
                ]
            },
        }
    )

    types = [e["event_type"] for e in stratix.events]
    assert "agent.output" in types
    assert "tool.call" in types
    assert "model.invoke" in types
    assert "cost.record" in types
    assert "agent.handoff" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    out_payload = out["payload"]
    assert out_payload["layer"] == "L1"
    assert out_payload["content"]["message"] == "the answer is 42"
    assert out_payload["content"]["role"] == "agent"
    assert out_payload["content"]["metadata"]["framework"] == "bedrock_agents"
    assert out_payload["content"]["metadata"]["session_id"] == "sess-1"

    model = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    m_payload = model["payload"]
    assert m_payload["layer"] == "L3"
    assert m_payload["model"]["provider"] == "aws_bedrock"
    assert m_payload["model"]["name"] == "anthropic.claude-v2"
    assert m_payload["model"]["version"] == "unavailable"
    assert m_payload["prompt_tokens"] == 100
    assert m_payload["completion_tokens"] == 50

    tool = next(e for e in stratix.events if e["event_type"] == "tool.call")
    t_payload = tool["payload"]
    assert t_payload["layer"] == "L5a"
    assert t_payload["tool"]["name"] == "calc"
    assert t_payload["tool"]["integration"] == "service"
    assert t_payload["input"]["x"] == 1
    assert t_payload["input"]["framework"] == "bedrock_agents"
    assert t_payload["input"]["tool_type"] == "action_group"

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    cost_payload = cost["payload"]
    assert cost_payload["cost"]["prompt_tokens"] == 100
    assert cost_payload["cost"]["completion_tokens"] == 50
    assert cost_payload["cost"]["tokens"] == 150

    handoff = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    h_payload = handoff["payload"]
    assert h_payload["from_agent"] == "sup-1"
    assert h_payload["to_agent"] == "col-1"
    assert h_payload["handoff_context_hash"].startswith("sha256:")
    assert len(h_payload["handoff_context_hash"]) == 7 + 64


def test_on_tool_use_emits_event() -> None:
    """Typed ToolCallEvent for a manual tool invocation."""
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["latency_ms"] == 12.3
    assert payload["input"]["x"] == 1
    assert payload["input"]["framework"] == "bedrock_agents"
    assert payload["output"] == {"value": 2}


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: canonical sha256:<hex64> handoff_context_hash."""
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "a"
    assert payload["to_agent"] == "b"
    assert payload["handoff_context_hash"].startswith("sha256:")
    assert len(payload["handoff_context_hash"]) == 7 + 64


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call events do NOT fire (handoff still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "agent.handoff" in types


def test_instrument_client_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    client = _FakeClient()
    adapter = instrument_client(client, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY
    # Hooks were registered.
    assert "provide-client-params.bedrock-agent-runtime.InvokeAgent" in client.meta.events.handlers


def test_serialize_for_replay() -> None:
    adapter = BedrockAgentsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "bedrock_agents"
    assert rt.adapter_name == "BedrockAgentsAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 4)
# ---------------------------------------------------------------------------


def test_bedrock_agents_emits_typed_payloads_only() -> None:
    """Every emit site in bedrock_agents lifecycle is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../bedrock_agents/lifecycle.py
    → 0`` acceptance criterion in the typed-events bundle 4 PR.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    # Drive every emission path that does not require boto3.
    adapter._before_invoke_agent(
        params={
            "agentId": "agent-123",
            "agentAliasId": "alias-1",
            "sessionId": "sess-1",
            "inputText": "hello",
            "enableTrace": True,
        }
    )
    adapter._after_invoke_agent(
        parsed={
            "outputText": "answer",
            "sessionId": "sess-1",
            "trace": {
                "steps": [
                    {
                        "type": "ACTION_GROUP",
                        "actionGroupName": "calc",
                        "actionGroupInput": {"x": 1},
                        "actionGroupInvocationOutput": {"output": "ok"},
                    },
                    {
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseId": "kb-1",
                        "knowledgeBaseLookupInput": {"q": "what"},
                        "knowledgeBaseLookupOutput": {"retrievedReferences": []},
                    },
                    {
                        "type": "MODEL_INVOCATION",
                        "foundationModel": "anthropic.claude-v2",
                        "modelInvocationOutput": {"usage": {"inputTokens": 1, "outputTokens": 2}},
                    },
                    {
                        "type": "AGENT_COLLABORATOR",
                        "supervisorAgentId": "sup-1",
                        "collaboratorAgentId": "col-1",
                    },
                ]
            },
        }
    )
    adapter.on_invoke_start(agent_id="agent-456", input_text="more input")
    adapter.on_invoke_end(agent_id="agent-456", output="more output")
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_llm_call(provider="aws_bedrock", model="anthropic.claude-v2", tokens_prompt=1, tokens_completion=2)
    adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert AgentHandoffEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen
    assert CostRecordEvent in types_seen


def test_bedrock_agents_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from bedrock_agents lifecycle emission paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, bedrock_agents lifecycle must
    never trigger that warning. ``filterwarnings("error", ...)``
    converts the warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter._before_invoke_agent(
            params={"agentId": "agent-1", "inputText": "hi"}
        )
        adapter._after_invoke_agent(
            parsed={
                "outputText": "out",
                "trace": {
                    "steps": [
                        {
                            "type": "MODEL_INVOCATION",
                            "foundationModel": "claude",
                            "modelInvocationOutput": {"usage": {"inputTokens": 1, "outputTokens": 1}},
                        },
                        {
                            "type": "AGENT_COLLABORATOR",
                            "supervisorAgentId": "s",
                            "collaboratorAgentId": "c",
                        },
                    ]
                },
            }
        )
        adapter.on_invoke_start(agent_id="a", input_text="x")
        adapter.on_invoke_end(agent_id="a", output="y")
        adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
        adapter.on_llm_call(provider="aws_bedrock", model="claude", tokens_prompt=1, tokens_completion=2)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
