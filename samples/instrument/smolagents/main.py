"""Sample: instrument a SmolAgents agent with the LayerLens adapter.

This sample is intentionally **offline** — it does not require the
``smolagents`` runtime, an OpenAI key, or network access. It builds a
duck-typed ``Agent`` (the same shape SmolAgents exposes), wraps it via
``SmolAgentsAdapter.instrument_agent``, and runs ``agent.run()``. The
adapter emits ``environment.config`` + ``agent.input`` + ``agent.output``
events into an in-process recording sink, then prints them so you can
see what would ship to atlas-app under real conditions.

For a real end-to-end run against the SmolAgents runtime, install the
extra and replace ``_FakeAgent`` with ``smolagents.CodeAgent`` /
``smolagents.ToolCallingAgent``::

    pip install 'layerlens[smolagents]'
    # Then swap _FakeAgent for the real one and configure HttpEventSink.

Required environment for the offline sample: none.

Run::

    python -m samples.instrument.smolagents.main
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.smolagents import (
    SmolAgentsAdapter,
)


class _FakeAgent:
    """Duck-typed SmolAgents agent for the offline demo.

    Mirrors the surface ``SmolAgentsAdapter.instrument_agent`` looks at:
    ``name``, ``run(task)``, ``tools``, ``model``, ``system_prompt``.
    """

    def __init__(self) -> None:
        self.name = "demo-agent"
        self.tools = ["search", "calc"]
        self.model = "offline-mock"
        self.system_prompt = "You are a helpful assistant."

    def run(self, task: str) -> str:
        # In a real CodeAgent, this would compile + execute Python that
        # invokes ``self.tools`` and an LLM. For the demo we return a
        # deterministic string so the sample is reproducible offline.
        return f"echo: {task}"


class _RecordingClient:
    """Stand-in for the LayerLens client. Captures events for inspection."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, "payload": payload})


def main() -> int:
    client = _RecordingClient()
    adapter = SmolAgentsAdapter(
        stratix=client,
        capture_config=CaptureConfig.standard(),
    )
    adapter.connect()

    agent = _FakeAgent()
    adapter.instrument_agent(agent)

    try:
        result = agent.run("What is 2 + 2?")
        print(f"Agent output: {result}")
    finally:
        adapter.disconnect()

    print(f"\nEmitted {len(client.events)} event(s):")
    for evt in client.events:
        agent_name = evt["payload"].get("agent_name", "<n/a>")
        print(f"  - {evt['event_type']:>22}  agent={agent_name}")

    print(
        "\nReplace _FakeAgent with smolagents.CodeAgent and add an "
        "HttpEventSink to ship telemetry to the LayerLens dashboard."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
