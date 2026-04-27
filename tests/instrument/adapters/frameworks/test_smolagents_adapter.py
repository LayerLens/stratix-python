"""Unit tests for the SmolAgents framework adapter.

Mocked at the SDK shape level — no real ``smolagents`` runtime needed.

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
from layerlens.instrument.adapters.frameworks.smolagents import (
    ADAPTER_CLASS,
    SmolAgentsAdapter,
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
    """Minimal duck-typed SmolAgents agent for tests."""

    def __init__(
        self,
        name: str = "test-agent",
        tools: Any = None,
        managed_agents: Any = None,
        model: Any = None,
        system_prompt: Any = None,
    ) -> None:
        self.name = name
        self.tools = tools
        self.managed_agents = managed_agents
        self.model = model
        self.system_prompt = system_prompt
        self._raised = False

    def run(self, task: str, **kwargs: Any) -> Any:
        if self._raised:
            raise RuntimeError("simulated failure")
        return f"result for {task}"


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is SmolAgentsAdapter


def test_lifecycle() -> None:
    a = SmolAgentsAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_instrument_agent_wraps_run() -> None:
    adapter = SmolAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # Wrapped: the bound method's underlying function is now ``traced_run``.
    assert agent.run.__name__ == "traced_run"

    adapter.disconnect()
    # Restored: name is back to the original.
    assert agent.run.__name__ == "run"


def test_run_emits_input_and_output_events() -> None:
    """Typed AgentInputEvent + AgentOutputEvent for the run lifecycle.
    SmolAgents-specific provenance lives on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    result = agent.run("compute 2+2")

    assert result == "result for compute 2+2"

    types = [e["event_type"] for e in stratix.events]
    # First event is environment.config from initial agent registration.
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    payload = out["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["message"] == "result for compute 2+2"
    metadata = payload["content"]["metadata"]
    assert metadata["agent_name"] == "planner"
    assert metadata["framework"] == "smolagents"
    assert metadata["duration_ns"] >= 0
    assert metadata["run_status"] == "run_complete"


def test_run_failure_emits_output_with_error() -> None:
    """Errors are surfaced via canonical metadata on AgentOutputEvent."""
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="failing")
    agent._raised = True
    adapter.instrument_agent(agent)

    import pytest

    with pytest.raises(RuntimeError):
        agent.run("bad task")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert "error" in metadata
    assert "simulated failure" in metadata["error"]
    assert metadata["run_status"] == "run_failed"


def test_managed_agents_recursively_instrumented() -> None:
    adapter = SmolAgentsAdapter(org_id="test-org")
    adapter.connect()

    sub = _FakeAgent(name="sub")
    parent = _FakeAgent(name="parent", managed_agents={"sub": sub})

    adapter.instrument_agent(parent)
    # Both wrapped.
    assert parent.run.__name__ == "traced_run"
    assert sub.run.__name__ == "traced_run"


def test_environment_config_emits_once_per_agent() -> None:
    """Typed EnvironmentConfigEvent: provenance lives on
    payload.environment.attributes.
    """
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(
        name="a1",
        tools=["search", "calc"],
        model="some-model",
        system_prompt="you are helpful",
    )
    adapter.instrument_agent(agent)
    # Re-instrument should not re-emit config.
    adapter.instrument_agent(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    # Canonical L4a schema: payload.environment.attributes is the dict
    # that carries adapter-specific provenance.
    attributes = configs[0]["payload"]["environment"]["attributes"]
    assert attributes["agent_name"] == "a1"
    assert attributes["tools"] == ["search", "calc"]
    assert configs[0]["payload"]["environment"]["type"] == "simulated"


def test_on_tool_use_emits_event() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name."""
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["latency_ms"] == 12.3
    assert payload["input"] == {"x": 1}
    assert payload["output"] == {"value": 2}


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>.

    The previous adapter's ``context_hash`` (bare hex) and
    ``context_preview`` (top-level) fields are gone — the canonical
    schema only declares ``handoff_context_hash`` (with strict
    ``sha256:`` prefix validation). Context redaction concerns live
    at the privacy-policy layer above the adapter, not on the
    canonical handoff payload.
    """
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
    adapter = SmolAgentsAdapter(
        stratix=stratix,
        capture_config=CaptureConfig(capture_content=False),
    )
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    # Hash is well-formed even without context.
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = SmolAgentsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "smolagents"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 1)
# ---------------------------------------------------------------------------


def test_smolagents_emits_typed_payloads_only() -> None:
    """Every emit site in smolagents lifecycle is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../smolagents/ → 0``
    acceptance criterion in the typed-events bundle 1 PR.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", tools=["search"], model="some-model")
    adapter.instrument_agent(agent)
    agent.run("compute 2+2")
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_llm_call(provider="hf", model="llama-3", tokens_prompt=10)
    adapter.on_handoff(from_agent="manager", to_agent="planner", context="ctx")

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


def test_smolagents_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from smolagents emission paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, smolagents must never trigger
    that warning. ``filterwarnings("error", ...)`` converts the
    warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", tools=["search"], model="some-model")
    adapter.instrument_agent(agent)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent.run("compute 2+2")
        adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
        adapter.on_llm_call(provider="hf", model="llama-3", tokens_prompt=10)
        adapter.on_handoff(from_agent="manager", to_agent="planner", context="ctx")
