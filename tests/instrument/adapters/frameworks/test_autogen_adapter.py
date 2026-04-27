"""Unit tests for the AutoGen framework adapter.

Mocked at the SDK shape level — no real ``autogen`` runtime needed.

After the typed-event migration (PR #129 follow-up — bundle 1) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests that
want to assert against the model surface.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.autogen import (
    ADAPTER_CLASS,
    AutoGenAdapter,
    instrument_agents,
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


class _FakeAgent:
    """Minimal duck-typed AutoGen ConversableAgent for tests."""

    def __init__(
        self,
        name: str = "agent",
        system_message: Any = None,
        llm_config: Any = None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config

    def send(self, message: Any, recipient: Any, **kwargs: Any) -> Any:
        return None

    def receive(self, message: Any, sender: Any, **kwargs: Any) -> Any:
        return None

    def generate_reply(self, messages: Any = None, sender: Any = None, **kwargs: Any) -> Any:
        return "reply"

    def execute_code_blocks(self, code_blocks: Any) -> Any:
        return "exec result"


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AutoGenAdapter


def test_lifecycle() -> None:
    a = AutoGenAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = AutoGenAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "autogen"
    assert info.name == "AutoGenAdapter"
    health = a.health_check()
    assert health.framework_name == "autogen"


def test_connect_agents_wraps_methods_and_emits_config() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    adapter.connect_agents(agent)

    # Methods replaced.
    assert agent.send.__name__ == "traced_send"
    assert agent.receive.__name__ == "traced_receive"
    assert agent.generate_reply.__name__ == "traced_generate_reply"
    assert agent.execute_code_blocks.__name__ == "traced_execute_code"

    # environment.config emitted once for this agent.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1

    adapter.disconnect()
    # Original methods restored.
    assert agent.send.__name__ == "send"


def test_connect_agents_idempotent() -> None:
    """Calling connect_agents twice with the same agent does not double-wrap."""
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="alice")
    adapter.connect_agents(agent)
    adapter.connect_agents(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1


def test_on_send_emits_handoff() -> None:
    """Typed AgentHandoffEvent: from/to live at the top level; the
    canonical sha256 hash is generated from the message preview.
    """
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    sender = _FakeAgent(name="alice")
    recipient = _FakeAgent(name="bob")
    adapter.on_send(sender=sender, message="hi there", recipient=recipient)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "alice"
    assert evt["payload"]["to_agent"] == "bob"
    # Canonical handoff_context_hash format: sha256:<64 hex chars>.
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")
    assert len(evt["payload"]["handoff_context_hash"]) == 7 + 64


def test_on_receive_emits_input_event_with_agent_role() -> None:
    """The receive boundary is mapped to AgentInputEvent(role=AGENT).

    The previous adapter implementation emitted an ad-hoc
    ``agent.state.change`` payload, which did not satisfy the canonical
    AgentStateChangeEvent ``before_hash`` / ``after_hash`` schema. The
    typed migration maps the receive boundary onto AgentInputEvent
    with ``role=AGENT`` (the message arrives from another agent), and
    framework provenance lives on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    receiver = _FakeAgent(name="bob")
    sender = _FakeAgent(name="alice")
    adapter.on_receive(receiver=receiver, message={"content": "hello"}, sender=sender)

    # At least one agent.input event (with role=AGENT) was emitted.
    inputs = [e for e in stratix.events if e["event_type"] == "agent.input"]
    receive_inputs = [
        e for e in inputs
        if e["payload"]["content"]["metadata"].get("event_subtype") == "message_received"
    ]
    assert len(receive_inputs) == 1
    payload = receive_inputs[0]["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["role"] == "agent"
    assert payload["content"]["metadata"]["agent"] == "bob"
    assert payload["content"]["metadata"]["from_agent"] == "alice"


def test_on_generate_reply_emits_model_invoke() -> None:
    """Typed ModelInvokeEvent: model identity lives at payload.model.*.

    AutoGen-specific provenance (``framework``, ``agent``,
    ``reply_preview``) is carried on ``model.parameters``. Token
    counts use the canonical ``prompt_tokens`` / ``completion_tokens``
    slots.
    """
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    reply = type("Reply", (), {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})()
    adapter.on_generate_reply(agent=agent, messages=[{"role": "user", "content": "hi"}], reply=reply, latency_ms=42.0)

    evt = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    payload = evt["payload"]
    assert payload["layer"] == "L3"
    assert payload["model"]["name"] == "gpt-5"
    assert payload["model"]["provider"] == "openai"  # detected from "gpt"
    assert payload["model"]["version"] == "unavailable"
    assert payload["model"]["parameters"]["agent"] == "alice"
    assert payload["model"]["parameters"]["framework"] == "autogen"
    assert payload["latency_ms"] == 42.0
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    # capture_content default is True → input messages are captured.
    assert payload["input_messages"] is not None


def test_on_execute_code_emits_tool_call_and_environment() -> None:
    """Typed ToolCallEvent + ToolEnvironmentEvent for code execution."""
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice")
    adapter.on_execute_code(agent=agent, code_blocks=[("python", "print(1)")], result="1\n", latency_ms=5.0)

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" in types
    assert "tool.environment" in types
    tool = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = tool["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "code_execution"
    assert payload["tool"]["integration"] == "script"
    assert payload["input"]["code_blocks_count"] == 1
    assert payload["input"]["agent"] == "alice"
    assert payload["latency_ms"] == 5.0
    # tool.environment carries the same execution context on
    # environment.config.
    env = next(e for e in stratix.events if e["event_type"] == "tool.environment")
    assert env["payload"]["layer"] == "L5c"
    assert env["payload"]["environment"]["config"]["execution_type"] == "code_block"
    assert env["payload"]["environment"]["config"]["agent"] == "alice"


def test_on_conversation_start_end_emits_input_output() -> None:
    """Typed AgentInputEvent + AgentOutputEvent for the conversation
    boundary. AutoGen-specific provenance lives on
    MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    initiator = _FakeAgent(name="alice")
    adapter.on_conversation_start(initiator=initiator, message="start")
    adapter.on_conversation_end(final_message="bye", termination_reason="max_rounds")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    payload = out["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["message"] == "bye"
    metadata = payload["content"]["metadata"]
    assert metadata["framework"] == "autogen"
    assert metadata["termination_reason"] == "max_rounds"
    assert metadata["duration_ns"] >= 0


def test_capture_config_gates_l3_model_metadata() -> None:
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = AutoGenAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    adapter.on_generate_reply(agent=agent, reply="hi")
    sender = _FakeAgent(name="alice")
    recipient = _FakeAgent(name="bob")
    adapter.on_send(sender=sender, message="x", recipient=recipient)

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    # handoff (from on_send) is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_instrument_agents_helper() -> None:
    """Top-level convenience wraps multiple agents at once."""
    a = _FakeAgent(name="a")
    b = _FakeAgent(name="b")
    result = instrument_agents(a, b, org_id="test-org")
    assert isinstance(result, list)
    assert len(result) == 2
    # Both wrapped.
    assert a.send.__name__ == "traced_send"
    assert b.send.__name__ == "traced_send"


def test_serialize_for_replay() -> None:
    adapter = AutoGenAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "autogen"
    assert rt.adapter_name == "AutoGenAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 1)
# ---------------------------------------------------------------------------


def test_autogen_emits_typed_payloads_only() -> None:
    """Every emit site in autogen lifecycle is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../autogen/lifecycle.py
    → 0`` acceptance criterion in the typed-events bundle 1 PR.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        ToolEnvironmentEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    sender = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    recipient = _FakeAgent(name="bob")
    adapter.connect_agents(sender, recipient)
    adapter.on_conversation_start(initiator=sender, message="start")
    adapter.on_send(sender=sender, message="hi", recipient=recipient)
    adapter.on_receive(receiver=recipient, message={"content": "hi"}, sender=sender)
    adapter.on_generate_reply(agent=recipient, messages=[], reply="ok")
    adapter.on_execute_code(agent=recipient, code_blocks=[("python", "x")], result="1\n")
    adapter.on_conversation_end(final_message="bye", termination_reason="done")

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
    assert ToolEnvironmentEvent in types_seen


def test_autogen_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from autogen lifecycle emission paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, autogen lifecycle must never
    trigger that warning. ``filterwarnings("error", ...)`` converts
    the warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    sender = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    recipient = _FakeAgent(name="bob")
    adapter.connect_agents(sender, recipient)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter.on_conversation_start(initiator=sender, message="start")
        adapter.on_send(sender=sender, message="hi", recipient=recipient)
        adapter.on_receive(receiver=recipient, message={"content": "hi"}, sender=sender)
        adapter.on_generate_reply(agent=recipient, messages=[], reply="ok")
        adapter.on_execute_code(agent=recipient, code_blocks=[("python", "x")], result="1\n")
        adapter.on_conversation_end(final_message="bye", termination_reason="done")
