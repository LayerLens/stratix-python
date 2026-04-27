"""Per-adapter capture/replay round-trip tests.

Cross-pollination audit §2.6 requires that each of the 8 lighter
adapters supports :meth:`BaseAdapter.execute_replay_via_factory`
through the shared
:class:`~layerlens.instrument.adapters._base.replay.ReplayExecutor`.

Each adapter test:

1. Builds an instrumented agent via the adapter's framework-specific
   ``instrument_*`` method.
2. Runs the agent once to capture an original trace.
3. Calls :meth:`execute_replay_via_factory` with a fresh-agent
   factory to replay the trace.
4. Asserts the resulting :class:`ReplayResult` carries the adapter's
   ``org_id`` and recorded the replay's events.

Tests use minimal duck-typed agents — no framework runtimes are
required. The adapter's ``instrument_*`` method wraps the duck-type
just like it would a real framework agent.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from layerlens.instrument.adapters._base import (
    ReplayResult,
    DivergenceKind,
)
from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter
from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter
from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter
from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter
from layerlens.instrument.adapters.frameworks.llama_index import LlamaIndexAdapter
from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter
from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter
from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentAdapter

_ORG_ID = "tenant-replay-tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an awaitable on a fresh loop (3.9 compat)."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Agno
# ---------------------------------------------------------------------------


class _AgnoLikeAgent:
    def __init__(self, name: str = "agno-agent", output: str = "ok") -> None:
        self.name = name
        self.tools: Any = None
        self.model: Any = None
        self.team: Any = None
        self._output = output

    def run(self, message: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(content=f"{self._output}:{message}")

    async def arun(self, message: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(content=f"{self._output}:{message}")


def test_agno_replay_round_trip_returns_replay_result() -> None:
    adapter = AgnoAdapter(org_id=_ORG_ID)
    adapter.connect()
    # Capture an original run.
    original = _AgnoLikeAgent(name="alpha")
    adapter.instrument_agent(original)
    original.run("hello")
    trace = adapter.serialize_for_replay()
    assert any(evt["event_type"] == "agent.input" for evt in trace.events)

    def factory() -> _AgnoLikeAgent:
        return _AgnoLikeAgent(name="alpha")

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "agno"
    assert result.succeeded is True
    # Replay should at least re-emit agent.input + agent.output.
    types = [evt["event_type"] for evt in result.captured_events]
    assert "agent.input" in types
    assert "agent.output" in types


def test_agno_replay_async_factory() -> None:
    adapter = AgnoAdapter(org_id=_ORG_ID)
    adapter.connect()
    original = _AgnoLikeAgent(name="alpha")
    adapter.instrument_agent(original)
    original.run("hello")
    trace = adapter.serialize_for_replay()

    async def factory() -> _AgnoLikeAgent:
        await asyncio.sleep(0)
        return _AgnoLikeAgent(name="alpha")

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert result.succeeded is True
    assert result.org_id == _ORG_ID


# ---------------------------------------------------------------------------
# OpenAI Agents
# ---------------------------------------------------------------------------


class _OpenAIAgentLike:
    def __init__(self, name: str = "openai-agent") -> None:
        self.name = name

    async def arun(self, inputs: Any) -> str:
        return f"openai-out:{inputs}"


def test_openai_agents_replay_round_trip() -> None:
    adapter = OpenAIAgentsAdapter(org_id=_ORG_ID)
    adapter.connect()
    # Manually emit an agent.input event so the trace has shape — the
    # SDK is not installed in test environment.
    adapter.on_run_start(agent_name="openai-agent", input_data="hello")
    adapter.on_run_end(agent_name="openai-agent", output="hello-result")
    trace = adapter.serialize_for_replay()

    def factory() -> _OpenAIAgentLike:
        return _OpenAIAgentLike(name="openai-agent")

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "openai_agents"


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class _LlamaWorkflowLike:
    def __init__(self) -> None:
        self.name = "llama-workflow"

    async def arun(self, inputs: Any) -> str:
        return f"llama-out:{inputs}"


def test_llama_index_replay_round_trip() -> None:
    adapter = LlamaIndexAdapter(org_id=_ORG_ID)
    adapter.connect()
    # Seed a trace event (LlamaIndex emits via dispatcher in real use).
    adapter.emit_dict_event("agent.input", {"input": "hello"})
    trace = adapter.serialize_for_replay()

    def factory() -> _LlamaWorkflowLike:
        return _LlamaWorkflowLike()

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "llama_index"


# ---------------------------------------------------------------------------
# Google ADK
# ---------------------------------------------------------------------------


class _ADKAgentLike:
    def __init__(self, name: str = "adk-agent") -> None:
        self.name = name

    def run(self, inputs: Any) -> str:
        return f"adk-out:{inputs}"


def test_google_adk_replay_round_trip() -> None:
    adapter = GoogleADKAdapter(org_id=_ORG_ID)
    adapter.connect()
    adapter.emit_dict_event(
        "agent.input",
        {"agent_name": "adk-agent", "input": "hello"},
    )
    trace = adapter.serialize_for_replay()

    def factory() -> _ADKAgentLike:
        return _ADKAgentLike(name="adk-agent")

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "google_adk"
    assert result.outputs == "adk-out:hello"


# ---------------------------------------------------------------------------
# Strands
# ---------------------------------------------------------------------------


class _StrandsAgentLike:
    def __init__(self, name: str = "strands-agent") -> None:
        self.name = name

    def __call__(self, inputs: Any) -> str:
        return f"strands-out:{inputs}"


def test_strands_replay_round_trip() -> None:
    adapter = StrandsAdapter(org_id=_ORG_ID)
    adapter.connect()
    original = _StrandsAgentLike()
    adapter.instrument_agent(original)
    # Emit something so the trace has events.
    adapter.emit_dict_event("agent.input", {"input": "hello"})
    trace = adapter.serialize_for_replay()

    def factory() -> _StrandsAgentLike:
        return _StrandsAgentLike(name="strands-agent")

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "strands"


# ---------------------------------------------------------------------------
# PydanticAI
# ---------------------------------------------------------------------------


class _PydanticAIAgentLike:
    def __init__(self, name: str = "pyai-agent") -> None:
        self.name = name

    async def run(self, user_prompt: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(data=f"pyai-out:{user_prompt}")

    def run_sync(self, user_prompt: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(data=f"pyai-out:{user_prompt}")


def test_pydantic_ai_replay_round_trip() -> None:
    adapter = PydanticAIAdapter(org_id=_ORG_ID)
    adapter.connect()
    original = _PydanticAIAgentLike()
    adapter.instrument_agent(original)
    original.run_sync("hello")
    trace = adapter.serialize_for_replay()

    def factory() -> _PydanticAIAgentLike:
        return _PydanticAIAgentLike()

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "pydantic_ai"
    assert result.succeeded is True


# ---------------------------------------------------------------------------
# SmolAgents
# ---------------------------------------------------------------------------


class _SmolAgentLike:
    def __init__(self, name: str = "smol-agent") -> None:
        self.name = name
        self.managed_agents: Any = None

    def run(self, task: Any, **kwargs: Any) -> str:
        return f"smol-out:{task}"


def test_smolagents_replay_round_trip() -> None:
    adapter = SmolAgentsAdapter(org_id=_ORG_ID)
    adapter.connect()
    original = _SmolAgentLike()
    adapter.instrument_agent(original)
    original.run("compute primes")
    trace = adapter.serialize_for_replay()

    def factory() -> _SmolAgentLike:
        return _SmolAgentLike()

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "smolagents"


# ---------------------------------------------------------------------------
# MS Agent Framework
# ---------------------------------------------------------------------------


class _MSAgentChatLike:
    def __init__(self, name: str = "ms-chat") -> None:
        self.name = name

    async def invoke(self, **kwargs: Any) -> Any:
        # MS Agent Framework's invoke yields ChatMessageContent items —
        # for the replay test we just return a single item.
        async def _gen() -> Any:
            for item in [SimpleNamespace(content="ms-out", role="assistant")]:
                yield item

        return _gen()


def test_ms_agent_framework_replay_round_trip() -> None:
    adapter = MSAgentAdapter(org_id=_ORG_ID)
    adapter.connect()
    adapter.emit_dict_event(
        "agent.input",
        {"agent_name": "ms-chat", "input": "hello"},
    )
    trace = adapter.serialize_for_replay()

    def factory() -> _MSAgentChatLike:
        return _MSAgentChatLike()

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert isinstance(result, ReplayResult)
    assert result.org_id == _ORG_ID
    assert result.framework == "ms_agent_framework"


# ---------------------------------------------------------------------------
# Cross-tenant isolation across all 8 adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_cls,framework",
    [
        (AgnoAdapter, "agno"),
        (OpenAIAgentsAdapter, "openai_agents"),
        (LlamaIndexAdapter, "llama_index"),
        (GoogleADKAdapter, "google_adk"),
        (StrandsAdapter, "strands"),
        (PydanticAIAdapter, "pydantic_ai"),
        (SmolAgentsAdapter, "smolagents"),
        (MSAgentAdapter, "ms_agent_framework"),
    ],
)
def test_replay_result_carries_adapter_org_id(
    adapter_cls: Any,
    framework: str,
) -> None:
    """Per-tenant binding propagates to the ReplayResult on every adapter."""
    adapter_a = adapter_cls(org_id="tenant-A")
    adapter_b = adapter_cls(org_id="tenant-B")
    adapter_a.connect()
    adapter_b.connect()

    adapter_a.emit_dict_event("agent.input", {"input": "hello"})
    trace_a = adapter_a.serialize_for_replay()

    class _FakeAgent:
        name = "fake"
        managed_agents: Any = None
        tools: Any = None
        model: Any = None
        team: Any = None

        def run(self, *args: Any, **kwargs: Any) -> str:
            return "fake-out"

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            return "fake-out"

        async def invoke(self, **kwargs: Any) -> Any:
            async def _gen() -> Any:
                for item in []:
                    yield item

            return _gen()

    def factory() -> Any:
        return _FakeAgent()

    result_a = _run(adapter_a.execute_replay_via_factory(trace_a, factory))
    result_b = _run(adapter_b.execute_replay_via_factory(trace_a, factory))

    assert result_a.org_id == "tenant-A"
    assert result_b.org_id == "tenant-B"
    assert result_a.framework == framework
    assert result_b.framework == framework

    # Every captured event on each result must carry that result's org_id.
    for evt in result_a.captured_events:
        assert evt.get("org_id") == "tenant-A"
    for evt in result_b.captured_events:
        assert evt.get("org_id") == "tenant-B"


# ---------------------------------------------------------------------------
# Honest divergence detection on a deliberate mismatch
# ---------------------------------------------------------------------------


def test_agno_replay_surfaces_divergence_when_replay_omits_event() -> None:
    """If the replay doesn't emit a tool.call the original had, divergence is recorded."""
    adapter = AgnoAdapter(org_id=_ORG_ID)
    adapter.connect()
    # Original run with a tool.call event.
    adapter.emit_dict_event("agent.input", {"input": "hello"})
    adapter.emit_dict_event("tool.call", {"tool_name": "search"})
    adapter.emit_dict_event("agent.output", {"output": "ok"})
    trace = adapter.serialize_for_replay()

    # Replay agent that only emits agent.input + agent.output (no tool).
    class _NoToolAgent:
        name = "alpha"
        tools = None
        model = None
        team = None

        def run(self, message: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(content="ok")

    def factory() -> _NoToolAgent:
        return _NoToolAgent()

    result = _run(adapter.execute_replay_via_factory(trace, factory))
    assert result.succeeded is True
    # The replay emits agent.input + agent.output + agent.state.change but no tool.call.
    assert not result.is_exact
    kinds = {d.kind for d in result.divergences}
    # Either a missing event (tool.call is absent at index 1) or
    # event_type_mismatch (replay's index 1 != original's tool.call).
    assert (
        DivergenceKind.MISSING_EVENT in kinds
        or DivergenceKind.EVENT_TYPE_MISMATCH in kinds
    )
