"""Tests for STRATIX CrewAI Callback Handler."""

import pytest

from layerlens.instrument.adapters.crewai.lifecycle import CrewAIAdapter
from layerlens.instrument.adapters.crewai.callbacks import STRATIXCrewCallback
from layerlens.instrument.adapters._capture import CaptureConfig


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

    def __init__(self, role="researcher", goal="research", backstory="curious"):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = []
        self.allow_delegation = True


class MockTask:
    """Mock CrewAI Task."""

    def __init__(self, description="task", expected_output="output", agent=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


class MockStepOutput:
    """Mock CrewAI step output."""

    def __init__(self, tool=None, tool_input=None, result=None, delegated_to=None, agent=None):
        self.tool = tool
        self.tool_input = tool_input
        self.result = result
        self.delegated_to = delegated_to
        self.agent = agent


class TestSTRATIXCrewCallback:
    """Tests for STRATIXCrewCallback."""

    def test_callback_initialization(self):
        """Test callback initializes correctly."""
        adapter = CrewAIAdapter()
        callback = STRATIXCrewCallback(adapter=adapter)

        assert callback._adapter is adapter
        assert callback._task_counter == 0
        assert len(callback._seen_agents) == 0

    def test_on_crew_start_routes_to_adapter(self):
        """Test on_crew_start routes to adapter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        callback.on_crew_start(inputs={"topic": "AI"})

        events = stratix.get_events("agent.input")
        assert len(events) == 1

    def test_on_crew_end_routes_to_adapter(self):
        """Test on_crew_end routes to adapter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        adapter.on_crew_start()
        callback.on_crew_end(output="result")

        events = stratix.get_events("agent.output")
        assert len(events) == 1

    def test_on_task_start_increments_counter(self):
        """Test on_task_start increments task counter."""
        adapter = CrewAIAdapter(capture_config=CaptureConfig.full())
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        task1 = MockTask(description="task 1", agent=MockAgent())
        task2 = MockTask(description="task 2", agent=MockAgent(role="writer"))

        callback.on_task_start(task1)
        callback.on_task_start(task2)

        assert callback._task_counter == 2

    def test_on_task_start_emits_with_agent_role(self):
        """Test on_task_start extracts agent role from task."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        agent = MockAgent(role="writer")
        task = MockTask(description="write report", agent=agent)

        callback.on_task_start(task)

        events = stratix.get_events("agent.code")
        assert len(events) == 1
        assert events[0]["payload"]["agent_role"] == "writer"
        assert events[0]["payload"]["task_description"] == "write report"

    def test_on_task_start_handles_none_task(self):
        """Test on_task_start handles None task gracefully."""
        adapter = CrewAIAdapter(capture_config=CaptureConfig.full())
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        # Should not raise
        callback.on_task_start(None)
        assert callback._task_counter == 1

    def test_on_task_end_routes_to_adapter(self):
        """Test on_task_end routes to adapter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        task = MockTask(agent=MockAgent())
        callback.on_task_end(task=task, output="done")

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1

    def test_on_agent_action_caches_seen_agents(self):
        """Test on_agent_action caches first encounter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        agent = MockAgent(role="researcher")
        callback.on_agent_action(agent=agent)

        assert "researcher" in callback._seen_agents

    def test_on_agent_action_emits_config_once(self):
        """Test agent config emitted only on first encounter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        agent = MockAgent(role="researcher")
        callback.on_agent_action(agent=agent)
        callback.on_agent_action(agent=agent)

        config_events = stratix.get_events("environment.config")
        assert len(config_events) == 1

    def test_on_agent_end_emits_state_change(self):
        """Test on_agent_end emits state change."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        agent = MockAgent(role="writer")
        callback.on_agent_end(agent=agent, output="report done")

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["agent_role"] == "writer"

    def test_on_tool_use_routes_to_adapter(self):
        """Test on_tool_use routes to adapter."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        callback.on_tool_use(
            tool_name="search",
            tool_input="query",
            tool_output="results",
        )

        events = stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "search"

    def test_on_llm_call_extracts_model_info(self):
        """Test on_llm_call extracts model from response."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        class MockResponse:
            model = "gpt-4"
            usage = {"prompt_tokens": 100, "completion_tokens": 50}

        callback.on_llm_call(response=MockResponse())

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4"
        assert events[0]["payload"]["tokens_prompt"] == 100

    def test_on_step_extracts_tool_usage(self):
        """Test on_step extracts tool usage from step output."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        step = MockStepOutput(tool="search", tool_input="AI", result="found it")
        callback.on_step(step)

        events = stratix.get_events("tool.call")
        assert len(events) == 1

    def test_on_step_detects_delegation(self):
        """Test on_step detects delegation."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        delegated_agent = MockAgent(role="writer")
        from_agent = MockAgent(role="manager")
        step = MockStepOutput(
            delegated_to=delegated_agent,
            agent=from_agent,
            result="delegated context",
        )
        callback.on_step(step)

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "manager"
        assert events[0]["payload"]["to_agent"] == "writer"

    def test_callback_error_isolation(self):
        """Test callback errors don't propagate."""
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = CrewAIAdapter(stratix=FailingSTRATIX())
        adapter.connect()
        callback = STRATIXCrewCallback(adapter=adapter)

        # None of these should raise
        callback.on_crew_start()
        callback.on_crew_end()
        callback.on_task_start(MockTask())
        callback.on_task_end()
        callback.on_agent_action()
        callback.on_agent_end()
        callback.on_tool_use()
        callback.on_llm_call()
        callback.on_step()
