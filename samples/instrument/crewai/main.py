"""Sample: instrument a CrewAI one-task crew with the LayerLens adapter.

Builds a single-agent, single-task crew, wraps it with ``CrewAIAdapter``, and
runs ``kickoff()``. Each crew kickoff emits ``agent.input``, ``model.invoke``,
``tool.call`` (if any), and ``agent.output`` events that ship to atlas-app
via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by the underlying CrewAI LLM (the default is
  OpenAI; CrewAI honours the standard env var).
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[crewai,providers-openai]'
    python -m samples.instrument.crewai.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from crewai import Crew, Task, Agent
    except ImportError:
        print(
            "crewai not installed. Install with:\n"
            "    pip install 'layerlens[crewai,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="crewai",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

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
        print(f"Result: {result}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
