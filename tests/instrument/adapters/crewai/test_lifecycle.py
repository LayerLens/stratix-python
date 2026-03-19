"""Tests for STRATIX CrewAI Lifecycle Hooks."""

import pytest

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.crewai.lifecycle import CrewAIAdapter


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockAgent:
    """Mock CrewAI Agent."""

    def __init__(
        self,
        role="researcher",
        goal="research things",
        backstory="a curious mind",
        tools=None,
        allow_delegation=True,
        verbose=True,
        max_iter=5,
        memory=False,
        llm=None,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.allow_delegation = allow_delegation
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory
        self.llm = llm


class MockTask:
    """Mock CrewAI Task."""

    def __init__(self, description="test task", expected_output="result", agent=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


class MockCrew:
    """Mock CrewAI Crew."""

    def __init__(self, agents=None, tasks=None, process="sequential"):
        self.agents = agents or []
        self.tasks = tasks or []
        self.process = process
        self.step_callback = None
        self.task_callback = None

    def kickoff(self):
        return "crew result"


class TestCrewAIAdapter:
    """Tests for CrewAIAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)

        assert adapter._stratix is stratix
        assert adapter.FRAMEWORK == "crewai"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_with_legacy_param(self):
        """Test adapter accepts legacy stratix_instance param."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix_instance=stratix)

        assert adapter._stratix is stratix

    def test_adapter_without_stratix(self):
        """Test adapter works without STRATIX instance."""
        adapter = CrewAIAdapter()

        assert not adapter.has_stratix

    def test_connect_sets_healthy(self):
        """Test connect sets adapter to HEALTHY state."""
        adapter = CrewAIAdapter()
        adapter.connect()

        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        """Test disconnect sets adapter to DISCONNECTED state."""
        adapter = CrewAIAdapter()
        adapter.connect()
        adapter.disconnect()

        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        """Test health_check returns correct health info."""
        adapter = CrewAIAdapter()
        adapter.connect()

        health = adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "crewai"
        assert health.error_count == 0
        assert health.circuit_open is False

    def test_get_adapter_info(self):
        """Test get_adapter_info returns correct metadata."""
        adapter = CrewAIAdapter()

        info = adapter.get_adapter_info()

        assert info.name == "CrewAIAdapter"
        assert info.framework == "crewai"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_serialize_for_replay(self):
        """Test serialize_for_replay returns valid trace."""
        adapter = CrewAIAdapter()

        trace = adapter.serialize_for_replay()

        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "CrewAIAdapter"
        assert trace.framework == "crewai"
        assert trace.trace_id  # non-empty

    def test_instrument_crew(self):
        """Test instrument_crew attaches callback."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent()
        crew = MockCrew(agents=[agent])

        result = adapter.instrument_crew(crew)

        assert result is crew
        assert hasattr(crew, "_stratix_callback")
        assert hasattr(crew, "_stratix_adapter")

    def test_instrument_crew_emits_agent_config(self):
        """Test instrument_crew emits environment.config for each agent."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        agents = [MockAgent(role="researcher"), MockAgent(role="writer")]
        crew = MockCrew(agents=agents)

        adapter.instrument_crew(crew)

        config_events = stratix.get_events("environment.config")
        assert len(config_events) == 2
        roles = {e["payload"]["agent_role"] for e in config_events}
        assert roles == {"researcher", "writer"}

    def test_instrument_crew_records_process_type(self):
        """Test instrument_crew records process type in config events."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        crew = MockCrew(agents=[MockAgent()], process="hierarchical")
        adapter.instrument_crew(crew)

        config_events = stratix.get_events("environment.config")
        assert config_events[0]["payload"]["process_type"] == "hierarchical"

    def test_agent_config_emitted_once(self):
        """Test agent config is only emitted once per agent role."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent(role="researcher")
        crew = MockCrew(agents=[agent])
        adapter.instrument_crew(crew)

        # Emit again for same role
        adapter._emit_agent_config(agent)

        config_events = stratix.get_events("environment.config")
        assert len(config_events) == 1

    def test_on_crew_start_emits_agent_input(self):
        """Test on_crew_start emits agent.input event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_crew_start(crew_input={"topic": "AI safety"})

        events = stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "crewai"
        assert events[0]["payload"]["input"] == {"topic": "AI safety"}

    def test_on_crew_end_emits_agent_output(self):
        """Test on_crew_end emits agent.output event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_crew_start()
        adapter.on_crew_end(crew_output="final result")

        events = stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["output"] == "final result"

    def test_on_crew_end_with_error(self):
        """Test on_crew_end records error."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_crew_start()
        adapter.on_crew_end(error=ValueError("test error"))

        events = stratix.get_events("agent.output")
        assert events[0]["payload"]["error"] == "test error"

    def test_on_task_start_emits_agent_code(self):
        """Test on_task_start emits agent.code event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        adapter.on_task_start(
            task_description="Research AI",
            agent_role="researcher",
            expected_output="report",
            task_order=1,
        )

        events = stratix.get_events("agent.code")
        assert len(events) == 1
        assert events[0]["payload"]["task_description"] == "Research AI"
        assert events[0]["payload"]["agent_role"] == "researcher"
        assert events[0]["payload"]["task_order"] == 1

    def test_on_task_start_gated_by_capture_config(self):
        """Test L2 events are gated by CaptureConfig."""
        stratix = MockStratix()
        # Default config has l2_agent_code=False
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_task_start(task_description="test")

        events = stratix.get_events("agent.code")
        assert len(events) == 0

    def test_on_task_end_emits_state_change(self):
        """Test on_task_end emits agent.state.change (cross-cutting)."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_task_end(task_output="result", agent_role="researcher")

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["agent_role"] == "researcher"

    def test_on_task_end_cross_cutting_always_emitted(self):
        """Test cross-cutting events emitted even with minimal config."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_task_end(task_output="result")

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1

    def test_on_tool_use_emits_tool_call(self):
        """Test on_tool_use emits tool.call event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="search",
            tool_input="query",
            tool_output="results",
            latency_ms=150.0,
        )

        events = stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "search"
        assert events[0]["payload"]["latency_ms"] == 150.0

    def test_on_tool_use_with_error(self):
        """Test on_tool_use records tool errors."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="search",
            error=RuntimeError("timeout"),
        )

        events = stratix.get_events("tool.call")
        assert events[0]["payload"]["error"] == "timeout"

    def test_on_llm_call_emits_model_invoke(self):
        """Test on_llm_call emits model.invoke event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_llm_call(
            provider="openai",
            model="gpt-4",
            tokens_prompt=100,
            tokens_completion=50,
            latency_ms=500.0,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "openai"
        assert events[0]["payload"]["model"] == "gpt-4"
        assert events[0]["payload"]["tokens_prompt"] == 100

    def test_on_delegation_emits_handoff(self):
        """Test on_delegation emits agent.handoff event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_delegation(
            from_agent="manager",
            to_agent="researcher",
            context="research AI safety",
        )

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "manager"
        assert events[0]["payload"]["to_agent"] == "researcher"

    def test_capture_config_minimal_gates_l3_l5(self):
        """Test minimal CaptureConfig disables L3 and L5 events."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_llm_call(model="gpt-4")
        adapter.on_tool_use(tool_name="search")

        assert len(stratix.get_events("model.invoke")) == 0
        assert len(stratix.get_events("tool.call")) == 0

    def test_capture_config_minimal_allows_l1(self):
        """Test minimal CaptureConfig allows L1 events."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_crew_start(crew_input="test")

        assert len(stratix.get_events("agent.input")) == 1

    def test_safe_serialize_dict(self):
        """Test _safe_serialize handles dicts."""
        adapter = CrewAIAdapter()
        assert adapter._safe_serialize({"key": "val"}) == {"key": "val"}

    def test_safe_serialize_none(self):
        """Test _safe_serialize handles None."""
        adapter = CrewAIAdapter()
        assert adapter._safe_serialize(None) is None

    def test_safe_serialize_primitive(self):
        """Test _safe_serialize handles primitives."""
        adapter = CrewAIAdapter()
        assert adapter._safe_serialize("hello") == "hello"
        assert adapter._safe_serialize(42) == 42

    def test_safe_serialize_fallback(self):
        """Test _safe_serialize falls back to str()."""
        adapter = CrewAIAdapter()

        class Custom:
            def __str__(self):
                return "custom_obj"

        assert adapter._safe_serialize(Custom()) == "custom_obj"

    def test_replay_trace_accumulates_events(self):
        """Test that events accumulate for replay."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_crew_start(crew_input="test")
        adapter.on_crew_end(crew_output="result")

        trace = adapter.serialize_for_replay()
        assert len(trace.events) == 2
