# CrewAI framework adapter

`layerlens.instrument.adapters.frameworks.crewai.CrewAIAdapter` instruments
[CrewAI](https://github.com/joaomdmoura/crewai) crews — multi-agent
collaborations with explicit role assignments and task delegation.

## Install

```bash
pip install 'layerlens[crewai]'
```

Pulls `crewai>=0.30,<0.90`.

## Quick start

```python
from crewai import Agent, Crew, Task

from layerlens.instrument.adapters.frameworks.crewai import (
    CrewAIAdapter,
    instrument_crew,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="crewai")
adapter = CrewAIAdapter()
adapter.add_sink(sink)
adapter.connect()

# build a one-task crew
researcher = Agent(role="Researcher", goal="Answer", backstory="...")
task = Task(description="What is 2 + 2?", agent=researcher, expected_output="A number")
crew = Crew(agents=[researcher], tasks=[task])

instrumented = adapter.instrument_crew(crew)
result = instrumented.kickoff()

adapter.disconnect()
sink.close()
```

The `instrument_crew(crew)` convenience helper wraps the whole flow above:

```python
instrumented = instrument_crew(crew)  # connects + wraps in one call
result = instrumented.kickoff()
```

## What's wrapped

`adapter.instrument_crew(crew)` patches the following on the underlying
crew + agent objects:

- `Crew.kickoff` — emits `agent.input` + `environment.config` at start,
  `agent.output` at completion.
- `Agent.execute_task` — emits `agent.action` + `agent.code` (when enabled)
  per task.
- `Agent._invoke_tool` — emits `tool.call` per tool invocation.
- `Agent.llm.call` — emits `model.invoke` per LLM call.
- Crew delegation events via `CrewDelegationTracker` — emits `agent.handoff`
  on `Agent.delegate` calls.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First `kickoff` of an instrumented crew. |
| `agent.input` | L1 | Start of every `kickoff`. |
| `agent.output` | L1 | End of every `kickoff` and per-task. |
| `agent.code` | L2 | When `CaptureConfig.l2_agent_code` is true; one per agent. |
| `agent.action` | L4a | Per task execution. |
| `agent.state.change` | cross-cutting | When agent memory or context changes. |
| `agent.handoff` | L4a | When one agent delegates to another. |
| `tool.call` | L5a | Per tool invocation inside a task. |
| `model.invoke` | L3 | Per LLM call from any crew agent. |

## CrewAI specifics

- **Multi-agent attribution**: every event payload includes `agent_id`,
  `agent_role`, and (when present) `task_id` so the platform can reconstruct
  who-did-what across a crew.
- **Memory tracking**: when a `memory_service` is passed to
  `CrewAIAdapter(memory_service=...)`, agent short-term memory writes emit
  `agent.state.change` with the memory diff.
- **Sequential vs hierarchical**: works for both `Process.sequential` and
  `Process.hierarchical`. Hierarchical delegation is captured via the
  delegation tracker.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = CrewAIAdapter(capture_config=CaptureConfig.standard())

# Add agent.code (the prompt template / system message of each agent).
adapter = CrewAIAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l2_agent_code=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
    ),
)
```

## BYOK

CrewAI agents instantiate their own LLM clients (LangChain or LiteLLM under
the hood). The CrewAI adapter does not own those keys. For platform-managed
BYOK see `docs/adapters/byok.md` (atlas-app M1.B).

## Backward compatibility

```python
from layerlens.instrument.adapters.frameworks.crewai import STRATIXCrewCallback
```

`STRATIXCrewCallback` is an alias for `LayerLensCrewCallback` and will be
removed in v2.0.
