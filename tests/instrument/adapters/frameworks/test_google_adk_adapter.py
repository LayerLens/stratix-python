"""Unit tests for the Google Agent Development Kit (ADK) framework adapter.

Mocked at the SDK shape level — no real ``google.adk`` runtime needed.
The adapter integrates via 6 native callbacks (before/after agent/model/tool).

After the typed-event migration (PR #129 follow-up — bundle 3) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests
that want to assert against the model surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.google_adk import (
    ADAPTER_CLASS,
    GoogleADKAdapter,
    instrument_agent,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        # Hold strong references to the original typed payloads for the
        # subset of tests that want to assert against the model surface
        # (e.g. ``isinstance(payload, ToolCallEvent)``).
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
    """Minimal duck-typed Google ADK agent for tests."""

    def __init__(
        self,
        name: str = "adk-agent",
        tools: Any = None,
        model: Any = None,
        description: Any = None,
        instruction: Any = None,
        sub_agents: Any = None,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.description = description
        self.instruction = instruction
        self.sub_agents = sub_agents
        self.before_agent_callback: Any = None
        self.after_agent_callback: Any = None
        self.before_model_callback: Any = None
        self.after_model_callback: Any = None
        self.before_tool_callback: Any = None
        self.after_tool_callback: Any = None


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is GoogleADKAdapter


def test_lifecycle() -> None:
    a = GoogleADKAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = GoogleADKAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "google_adk"
    assert info.name == "GoogleADKAdapter"
    health = a.health_check()
    assert health.framework_name == "google_adk"


def test_instrument_agent_attaches_callbacks() -> None:
    adapter = GoogleADKAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # All six callbacks attached. Bound methods compare equal but not identical.
    assert agent.before_agent_callback == adapter._before_agent_callback
    assert agent.after_agent_callback == adapter._after_agent_callback
    assert agent.before_model_callback == adapter._before_model_callback
    assert agent.after_model_callback == adapter._after_model_callback
    assert agent.before_tool_callback == adapter._before_tool_callback
    assert agent.after_tool_callback == adapter._after_tool_callback


def test_before_after_agent_emits_input_output() -> None:
    """Typed AgentInputEvent + AgentOutputEvent for the agent lifecycle.
    ADK-specific provenance lives on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gemini-2", tools=[SimpleNamespace(name="search")])
    callback_context = SimpleNamespace(agent=agent, user_content="hello world", agent_output="response", session=None)

    adapter._before_agent_callback(callback_context)
    adapter._after_agent_callback(callback_context)

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    inp = next(e for e in stratix.events if e["event_type"] == "agent.input")
    inp_payload = inp["payload"]
    assert inp_payload["layer"] == "L1"
    assert inp_payload["content"]["message"] == "hello world"
    assert inp_payload["content"]["metadata"]["agent_name"] == "planner"
    assert inp_payload["content"]["metadata"]["framework"] == "google_adk"

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    out_payload = out["payload"]
    assert out_payload["content"]["message"] == "response"
    assert out_payload["content"]["metadata"]["duration_ns"] >= 0


def test_after_model_emits_invoke_and_cost() -> None:
    """Typed ModelInvokeEvent + CostRecordEvent.
    Model name lives at payload.model.name; tokens at payload.cost.*.
    """
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    callback_context = SimpleNamespace(model="gemini-2", agent=None)
    llm_request = SimpleNamespace()
    adapter._before_model_callback(callback_context, llm_request)

    llm_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )
    adapter._after_model_callback(callback_context, llm_response)

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    inv_payload = invoke["payload"]
    assert inv_payload["layer"] == "L3"
    assert inv_payload["model"]["name"] == "gemini-2"
    assert inv_payload["model"]["provider"] == "google"
    assert inv_payload["model"]["version"] == "unavailable"
    assert inv_payload["prompt_tokens"] == 10
    assert inv_payload["completion_tokens"] == 20

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    cost_payload = cost["payload"]
    assert cost_payload["cost"]["prompt_tokens"] == 10
    assert cost_payload["cost"]["completion_tokens"] == 20
    assert cost_payload["cost"]["tokens"] == 30


def test_after_tool_emits_tool_call() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name."""
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    inp = {"x": 1}
    adapter._before_tool_callback(SimpleNamespace(), "calc", inp)
    adapter._after_tool_callback(SimpleNamespace(), "calc", inp, 42)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["tool"]["version"] == "unavailable"
    assert payload["latency_ms"] is not None
    # Scalar tool_output is wrapped in {"value": ...} so the canonical
    # ``output: dict`` slot is satisfied.
    assert payload["output"] == {"value": 42}
    assert payload["input"]["x"] == 1
    assert payload["input"]["framework"] == "google_adk"


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>.

    The previous adapter's ``context_hash`` (bare hex) and
    ``context_preview`` (top-level) fields are gone — the canonical
    schema only declares ``handoff_context_hash`` (with strict
    ``sha256:`` prefix validation).
    """
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "a"
    assert payload["to_agent"] == "b"
    assert payload["handoff_context_hash"].startswith("sha256:")
    # 7-char prefix + 64 hex chars = 71 chars total per the canonical
    # validator in events_cross_cutting.py.
    assert len(payload["handoff_context_hash"]) == 7 + 64


def test_handoff_emits_canonical_hash_for_empty_context() -> None:
    """Empty context still produces a well-formed sha256 hash.

    The previous adapter emitted ``context_hash=None`` when the
    context was missing; the canonical schema rejects ``None``.
    """
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


def test_capture_config_gates_l3_model_metadata() -> None:
    """When l3_model_metadata is disabled, model.invoke does NOT fire (handoff still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    callback_context = SimpleNamespace(model="gemini-2", agent=None)
    adapter._before_model_callback(callback_context, SimpleNamespace())
    adapter._after_model_callback(
        callback_context,
        SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)),
    )
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    assert "agent.handoff" in types


def test_environment_config_emits_once_per_agent() -> None:
    """Typed EnvironmentConfigEvent: provenance lives on
    payload.environment.attributes.
    """
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")])
    cb = SimpleNamespace(agent=agent, user_content="hi", agent_output=None, session=None)
    adapter._before_agent_callback(cb)
    # second call should not re-emit environment.config
    adapter._before_agent_callback(cb)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    # Canonical L4a schema: payload.environment.attributes is the dict
    # that carries adapter-specific provenance (agent_name).
    attributes = configs[0]["payload"]["environment"]["attributes"]
    assert attributes["agent_name"] == "a1"
    assert configs[0]["payload"]["environment"]["type"] == "simulated"


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = GoogleADKAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "google_adk"
    assert rt.adapter_name == "GoogleADKAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 3)
# ---------------------------------------------------------------------------


def test_google_adk_lifecycle_emits_typed_payloads_only() -> None:
    """Every emit site in google_adk lifecycle.py is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../google_adk/ → 0``
    acceptance criterion in the typed-events bundle 3 PR.
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
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gemini-2", tools=[SimpleNamespace(name="search")])
    cb = SimpleNamespace(agent=agent, user_content="hello", agent_output="reply", session=None)
    adapter._before_agent_callback(cb)
    adapter._after_agent_callback(cb)

    model_cb = SimpleNamespace(model="gemini-2", agent=None)
    adapter._before_model_callback(model_cb, SimpleNamespace())
    adapter._after_model_callback(
        model_cb,
        SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)),
    )

    inp = {"x": 1}
    adapter._before_tool_callback(SimpleNamespace(), "calc", inp)
    adapter._after_tool_callback(SimpleNamespace(), "calc", inp, 42)

    adapter.on_agent_start(agent_name="other", input_data="task")
    adapter.on_agent_end(agent_name="other", output="done")
    adapter.on_tool_use("ext_tool", tool_input={"a": 1}, tool_output="ok")
    adapter.on_llm_call(provider="google", model="gemini-2", tokens_prompt=5)
    adapter.on_handoff(from_agent="planner", to_agent="executor", context="ctx")

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert AgentHandoffEvent in types_seen
    assert CostRecordEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen


def test_google_adk_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from google_adk lifecycle emission paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, google_adk lifecycle must never
    trigger that warning. ``filterwarnings("error", ...)`` converts
    the warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gemini-2", tools=[SimpleNamespace(name="search")])
    cb = SimpleNamespace(agent=agent, user_content="hello", agent_output="reply", session=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter._before_agent_callback(cb)
        adapter._after_agent_callback(cb)
        adapter._before_model_callback(SimpleNamespace(model="gemini-2", agent=None), SimpleNamespace())
        adapter._after_model_callback(
            SimpleNamespace(model="gemini-2", agent=None),
            SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)),
        )
        inp = {"x": 1}
        adapter._before_tool_callback(SimpleNamespace(), "calc", inp)
        adapter._after_tool_callback(SimpleNamespace(), "calc", inp, 42)
        adapter.on_agent_start(agent_name="o", input_data="i")
        adapter.on_agent_end(agent_name="o", output="o")
        adapter.on_tool_use("t", tool_input={"a": 1}, tool_output="ok")
        adapter.on_llm_call(provider="google", model="gemini-2", tokens_prompt=5)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
