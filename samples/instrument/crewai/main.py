"""Sample: instrument a CrewAI one-task crew with the LayerLens adapter.

Builds a single-agent, single-task crew, wraps it with ``CrewAIAdapter``, and
runs ``kickoff()``. Each crew kickoff emits ``agent.input``, ``model.invoke``,
``tool.call`` (if any), ``agent.handoff`` (when delegation occurs), and
``agent.output`` events. This sample collects events with a small in-process
``EventSink`` and prints a summary so you can verify wiring without standing
up a backend.

Required environment:

* ``OPENAI_API_KEY`` — used by the underlying CrewAI LLM (default; CrewAI
  honours the standard env var).

Run::

    pip install 'layerlens[crewai]' openai
    python -m samples.instrument.crewai.main

To ship the same events to atlas-app instead of printing, swap the
``_PrintSink`` below for a real transport sink (``HttpEventSink`` /
``OTLPHttpSink``) once those land.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import CaptureConfig, EventSink
from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter


class _PrintSink(EventSink):
    """Tiny in-process sink that records and prints every event.

    Stand-in for the real HTTP/OTLP transport sinks shipped in M3+.
    Useful for samples and local diagnosis.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        self.events.append(
            {"event_type": event_type, "payload": payload, "timestamp_ns": timestamp_ns}
        )
        print(f"[event] {event_type}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print(
            "crewai not installed. Install with:\n"
            "    pip install 'layerlens[crewai]' openai",
            file=sys.stderr,
        )
        return 2

    sink = _PrintSink()
    adapter = CrewAIAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    researcher = Agent(
        role="Math Tutor",
        goal="Answer arithmetic questions concisely.",
        backstory="A concise math tutor who replies with a single number.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="What is 2 + 2? Reply with just the number.",
        agent=researcher,
        expected_output="A single integer.",
    )
    crew = Crew(agents=[researcher], tasks=[task], verbose=False)

    try:
        instrumented = adapter.instrument_crew(crew)
        result = instrumented.kickoff()
        print(f"\nCrew result: {result}")
    finally:
        adapter.disconnect()

    print(f"\nCaptured {len(sink.events)} event(s):")
    for evt in sink.events:
        print(f"  - {evt['event_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
