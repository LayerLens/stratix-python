"""Integration tests for Bedrock Agents adapter using the REAL boto3 SDK.

Ported from ``ateam/tests/adapters/bedrock_agents/test_integration.py``.

These tests verify that BedrockAgentsAdapter correctly captures events
from actual boto3 types and Bedrock Agent constructs -- not mocks.
The SDK must be installed::

    pip install boto3

Tests are skipped if boto3 is not installed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

boto3 = pytest.importorskip("boto3", reason="boto3 not installed")
botocore = pytest.importorskip("botocore", reason="botocore not installed")

from botocore.config import Config as BotoConfig  # noqa: E402
from botocore.session import Session as BotocoreSession  # noqa: E402

from layerlens.instrument.adapters._base.adapter import (  # noqa: E402
    AdapterStatus,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents.lifecycle import (  # noqa: E402
    BedrockAgentsAdapter,
)

# ---------------------------------------------------------------------------
# EventCollector -- real collector, not a mock
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector -- accumulates events for assertions."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.traces_started: int = 0
        self.traces_ended: int = 0

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def start_trace(self, **kwargs: Any) -> str:
        self.traces_started += 1
        return f"trace-{self.traces_started}"

    def end_trace(self, **kwargs: Any) -> None:
        self.traces_ended += 1

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


# ---------------------------------------------------------------------------
# Adapter construction with real boto3 types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and works with real boto3 classes."""

    def test_adapter_detects_boto3_version(self) -> None:
        """connect() should discover the installed boto3 version."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_version == boto3.__version__

    def test_adapter_framework_metadata(self) -> None:
        """Adapter should expose correct framework name and version."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "bedrock_agents"
        assert adapter.VERSION is not None

    def test_adapter_capabilities_include_handoffs(self) -> None:
        """Bedrock Agents adapter must declare TRACE_HANDOFFS for supervisor mode."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()
        info = adapter.get_adapter_info()
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    def test_botocore_session_exists(self) -> None:
        """Verify botocore Session is available (adapter depends on event system)."""
        session = BotocoreSession()
        assert session is not None
        # Verify the event system exists -- this is what the adapter hooks into
        assert hasattr(session, "get_available_events") or hasattr(session, "create_client")

    def test_boto_config_object(self) -> None:
        """BotoConfig is usable -- needed for bedrock-agent-runtime client config."""
        config = BotoConfig(
            region_name="us-east-1",
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        assert config.region_name == "us-east-1"


# ---------------------------------------------------------------------------
# Lifecycle events via on_invoke_start / on_invoke_end
# ---------------------------------------------------------------------------


class TestInvokeLifecycleEvents:
    """Verify agent invoke lifecycle emits correct events."""

    def test_invoke_start_emits_agent_input(self) -> None:
        """on_invoke_start should emit an agent.input event."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_invoke_start(agent_id="AGENT123ABC", input_text="What is EC2?")

        events = collector.get_events("agent.input")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["framework"] == "bedrock_agents"
        assert payload["agent_id"] == "AGENT123ABC"
        assert payload["input"] == "What is EC2?"
        assert "timestamp_ns" in payload

    def test_invoke_roundtrip_captures_duration(self) -> None:
        """on_invoke_start + on_invoke_end should compute duration_ns."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_invoke_start(agent_id="AGENT123ABC", input_text="hello")
        adapter.on_invoke_end(agent_id="AGENT123ABC", output="EC2 is a compute service.")

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        payload = output_events[0]["payload"]
        assert payload["output"] == "EC2 is a compute service."
        assert payload["duration_ns"] >= 0

    def test_invoke_end_with_error(self) -> None:
        """on_invoke_end with error should include error string."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_invoke_start(agent_id="AGENT123ABC", input_text="fail")
        adapter.on_invoke_end(
            agent_id="AGENT123ABC",
            output=None,
            error=RuntimeError("Throttling: Rate exceeded"),
        )

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert "Throttling" in output_events[0]["payload"]["error"]

    def test_disconnected_adapter_emits_nothing(self) -> None:
        """Lifecycle hooks should no-op when adapter is disconnected."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        # NOT calling adapter.connect()

        adapter.on_invoke_start(agent_id="AGENT123ABC", input_text="hello")
        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Tool call events (Action Groups / Knowledge Bases)
# ---------------------------------------------------------------------------


class TestToolCallEvents:
    """Verify tool.call events for Action Groups and KB queries."""

    def test_action_group_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event with correct fields."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="GetInstanceStatus",
            tool_input={"instanceId": "i-0123456789abcdef0"},
            tool_output={"status": "running", "instanceType": "m5.large"},
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "GetInstanceStatus"
        assert payload["tool_input"]["instanceId"] == "i-0123456789abcdef0"
        assert payload["tool_output"]["status"] == "running"

    def test_tool_call_with_error(self) -> None:
        """Tool call with error should include error string."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="QueryKB",
            tool_input={"query": "test"},
            error=ValueError("Knowledge base not found"),
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert "Knowledge base not found" in events[0]["payload"]["error"]

    def test_tool_call_with_latency(self) -> None:
        """Tool call should preserve latency_ms when provided."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="SearchDocs",
            tool_input={"query": "pricing"},
            tool_output={"results": []},
            latency_ms=245.7,
        )

        events = collector.get_events("tool.call")
        assert events[0]["payload"]["latency_ms"] == 245.7

    def test_tool_events_disabled_by_capture_config(self) -> None:
        """When l5a_tool_calls=False, tool events should not be captured."""
        collector = EventCollector()
        config = CaptureConfig(l5a_tool_calls=False)
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_tool_use(tool_name="GetStatus", tool_input={}, tool_output={})

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 0


# ---------------------------------------------------------------------------
# Model invocation events
# ---------------------------------------------------------------------------


class TestModelInvocationEvents:
    """Verify model.invoke events for Bedrock foundation model calls."""

    def test_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="aws_bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            tokens_prompt=150,
            tokens_completion=80,
            latency_ms=1200.0,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert payload["provider"] == "aws_bedrock"
        assert payload["tokens_prompt"] == 150
        assert payload["tokens_completion"] == 80
        assert payload["latency_ms"] == 1200.0

    def test_model_events_disabled_by_capture_config(self) -> None:
        """When l3_model_metadata=False, model events should not be captured."""
        collector = EventCollector()
        config = CaptureConfig(l3_model_metadata=False)
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(model="anthropic.claude-3-haiku-20240307-v1:0")

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 0

    def test_llm_call_with_messages_content_capture(self) -> None:
        """Messages should only be included when capture_content=True."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=True)
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "What is S3?"}],
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" in events[0]["payload"]
        assert events[0]["payload"]["messages"][0]["content"] == "What is S3?"

    def test_llm_call_without_content_capture(self) -> None:
        """Messages should be excluded when capture_content=False."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=False)
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "secret data"}],
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" not in events[0]["payload"]


# ---------------------------------------------------------------------------
# Supervisor -> Collaborator handoff events
# ---------------------------------------------------------------------------


class TestHandoffEvents:
    """Verify agent.handoff events for multi-agent supervisor patterns."""

    def test_handoff_event(self) -> None:
        """on_handoff should emit agent.handoff with from/to agents."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="supervisor-agent",
            to_agent="code-review-collaborator",
            context="Review PR #42",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["from_agent"] == "supervisor-agent"
        assert payload["to_agent"] == "code-review-collaborator"
        assert payload["reason"] == "supervisor_delegation"
        # Context should be hashed (not stored in plaintext)
        expected_hash = hashlib.sha256(b"Review PR #42").hexdigest()
        assert payload["context_hash"] == expected_hash

    def test_handoff_without_context(self) -> None:
        """Handoff without context should set context_hash to None."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(from_agent="supervisor", to_agent="collaborator")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["context_hash"] is None

    def test_handoff_always_emitted_even_with_minimal_config(self) -> None:
        """agent.handoff is cross-cutting and should emit even with minimal config."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = BedrockAgentsAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_handoff(from_agent="a", to_agent="b")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Trace extraction from Bedrock response structures
# ---------------------------------------------------------------------------


class TestTraceExtraction:
    """Verify _process_trace extracts events from Bedrock trace structures."""

    def test_action_group_trace_step(self) -> None:
        """ACTION_GROUP steps should emit tool.call events."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        parsed = {
            "trace": {
                "steps": [
                    {
                        "type": "ACTION_GROUP",
                        "actionGroupName": "GetWeather",
                        "actionGroupInput": {"city": "Seattle"},
                        "actionGroupInvocationOutput": {"output": {"temperature": "62F"}},
                    }
                ]
            }
        }
        adapter._process_trace(parsed)

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "GetWeather"
        assert events[0]["payload"]["tool_type"] == "action_group"

    def test_knowledge_base_trace_step(self) -> None:
        """KNOWLEDGE_BASE steps should emit tool.call events."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        parsed = {
            "trace": {
                "steps": [
                    {
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseId": "KB-001",
                        "knowledgeBaseLookupInput": {"query": "pricing"},
                        "knowledgeBaseLookupOutput": {
                            "retrievedReferences": [{"content": "Standard tier: $10/mo"}]
                        },
                    }
                ]
            }
        }
        adapter._process_trace(parsed)

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "KB-001"
        assert events[0]["payload"]["tool_type"] == "knowledge_base_retrieval"

    def test_model_invocation_trace_step(self) -> None:
        """MODEL_INVOCATION steps should emit model.invoke + cost.record."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        parsed = {
            "trace": {
                "steps": [
                    {
                        "type": "MODEL_INVOCATION",
                        "foundationModel": "anthropic.claude-3-sonnet",
                        "modelInvocationOutput": {
                            "usage": {
                                "inputTokens": 200,
                                "outputTokens": 100,
                            }
                        },
                    }
                ]
            }
        }
        adapter._process_trace(parsed)

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        assert model_events[0]["payload"]["model"] == "anthropic.claude-3-sonnet"
        assert model_events[0]["payload"]["provider"] == "aws_bedrock"

        cost_events = collector.get_events("cost.record")
        assert len(cost_events) == 1
        assert cost_events[0]["payload"]["tokens_prompt"] == 200
        assert cost_events[0]["payload"]["tokens_completion"] == 100
        assert cost_events[0]["payload"]["tokens_total"] == 300

    def test_collaborator_trace_step(self) -> None:
        """AGENT_COLLABORATOR steps should emit agent.handoff."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        parsed = {
            "trace": {
                "steps": [
                    {
                        "type": "AGENT_COLLABORATOR",
                        "supervisorAgentId": "supervisor-1",
                        "collaboratorAgentId": "coder-1",
                    }
                ]
            }
        }
        adapter._process_trace(parsed)

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "supervisor-1"
        assert events[0]["payload"]["to_agent"] == "coder-1"

    def test_empty_trace_emits_nothing(self) -> None:
        """Empty trace should not emit any events."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter._process_trace({})
        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Completion extraction
# ---------------------------------------------------------------------------


class TestCompletionExtraction:
    """Verify _extract_completion handles various Bedrock response shapes."""

    def test_extract_from_output_text(self) -> None:
        adapter = BedrockAgentsAdapter()
        result = adapter._extract_completion({"outputText": "Hello world"})
        assert result == "Hello world"

    def test_extract_from_output_dict(self) -> None:
        adapter = BedrockAgentsAdapter()
        result = adapter._extract_completion({"output": {"text": "response text"}})
        assert result == "response text"

    def test_extract_from_session_attributes(self) -> None:
        adapter = BedrockAgentsAdapter()
        result = adapter._extract_completion({"sessionAttributes": {"key": "value"}})
        assert isinstance(result, str)

    def test_extract_returns_none_for_empty(self) -> None:
        adapter = BedrockAgentsAdapter()
        result = adapter._extract_completion({})
        assert result is None


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle with real SDK."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should not raise."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_serialization_for_replay(self) -> None:
        """serialize_for_replay should produce a ReplayableTrace with events."""
        collector = EventCollector()
        adapter = BedrockAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_invoke_start(agent_id="test", input_text="hello")
        adapter.on_invoke_end(agent_id="test", output="world")

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "BedrockAgentsAdapter"
        assert trace.framework == "bedrock_agents"
        assert len(trace.events) >= 2

    def test_null_stratix_pattern(self) -> None:
        """Adapter should work (no-op) without a stratix instance."""
        adapter = BedrockAgentsAdapter()
        adapter.connect()
        # Should not raise
        adapter.on_invoke_start(agent_id="test", input_text="hello")
        adapter.on_invoke_end(agent_id="test", output="world")
        adapter.on_tool_use(tool_name="test", tool_input={})
        adapter.on_llm_call(model="test")
        adapter.on_handoff(from_agent="a", to_agent="b")
