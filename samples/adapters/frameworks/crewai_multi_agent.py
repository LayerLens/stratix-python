"""Sample: CrewAI multi-agent crew instrumented with layerlens."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter


def main() -> None:
    try:
        from crewai import Crew, Task, Agent  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install crewai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run CrewAI against a live LLM.")
        return

    researcher = Agent(
        role="researcher",
        goal="find one interesting fact",
        backstory="curious",
        allow_delegation=False,
    )
    writer = Agent(role="writer", goal="summarize in one line", backstory="terse", allow_delegation=False)
    task = Task(description="Produce one line about the moon.", agent=researcher, expected_output="a one-liner")
    crew = Crew(agents=[researcher, writer], tasks=[task])

    CrewAIAdapter().connect(crew)
    with capture_events("crewai"):
        print(crew.kickoff())


if __name__ == "__main__":
    main()
