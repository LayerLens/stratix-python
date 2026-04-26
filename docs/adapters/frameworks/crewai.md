# CrewAI framework adapter

`layerlens.instrument.adapters.frameworks.crewai.CrewAIAdapter` instruments
[CrewAI](https://github.com/joaomdmoura/crewai) crews — multi-agent
collaborations with explicit role assignments and task delegation.

## Install

```bash
pip install 'layerlens[crewai]'
```

Pulls `crewai>=0.30,<0.90`. CrewAI 0.30+ requires Python 3.10 or newer
and pins `pydantic = "^2"`, so the adapter declares
`requires_pydantic = PydanticCompat.V2_ONLY`. Importing it under
Pydantic v1 raises a clear error rather than failing inside CrewAI's
own model layer.

## Quick start

```python
from crewai import Agent, Crew, Task

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.crewai import (
    CrewAIAdapter,
    instrument_crew,
)

adapter = CrewAIAdapter(capture_config=CaptureConfig.standard())
adapter.connect()

# build a one-task crew
researcher = Agent(role="Researcher", goal="Answer", backstory="...")
task = Task(description="What is 2 + 2?", agent=researcher, expected_output="A number")
crew = Crew(agents=[researcher], tasks=[task])

instrumented = adapter.instrument_crew(crew)
result = instrumented.kickoff()

adapter.disconnect()
```

The `instrument_crew(crew)` convenience helper wraps `connect` +
`instrument_crew` + return:

```python
from layerlens.instrument.adapters.frameworks.crewai import instrument_crew

instrumented = instrument_crew(crew)  # connects + wraps in one call
result = instrumented.kickoff()
```

## What's wrapped

`adapter.instrument_crew(crew)` patches the following on the underlying
crew + agent objects:

- `Crew.kickoff` — emits `agent.input` + `environment.config` at start,
  `agent.output` at completion.
- Per-task `step_callback` — emits `agent.code` (when enabled) and
  routes tool / delegation events.
- `Agent` tool invocations — emit `tool.call` per tool use.
- LLM calls — emit `model.invoke`.
- Crew delegation events via `CrewDelegationTracker` — emit
  `agent.handoff` whenever one agent delegates to another.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First time each agent role is seen on a crew. |
| `agent.input` | L1 | Start of every `kickoff`. |
| `agent.output` | L1 | End of every `kickoff`. |
| `agent.code` | L2 | When `CaptureConfig.l2_agent_code` is true; per task start. |
| `agent.state.change` | cross-cutting | Per task completion and per agent end. |
| `agent.handoff` | cross-cutting | When one agent delegates to another. |
| `tool.call` | L5a | Per tool invocation inside a task. |
| `model.invoke` | L3 | Per LLM call from any crew agent. |
| `cost.record` | cross-cutting | Per task completion when token usage is reported. |

## CrewAI specifics

- **Multi-agent attribution**: every event payload carries
  `framework="crewai"`, `agent_role`, and (when present) `task_order`
  so the platform can reconstruct who-did-what across a crew.
- **Memory tracking**: when a `memory_service` is passed to
  `CrewAIAdapter(memory_service=...)`, the adapter's
  `inject_memory_context` and `store_task_result` helpers retrieve and
  persist procedural memories around each task.
- **Sequential vs hierarchical**: works for both `Process.sequential`
  and `Process.hierarchical`. Hierarchical delegation is captured by
  the `CrewDelegationTracker`.

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

Cross-cutting events (`agent.handoff`, `cost.record`) are always
emitted regardless of the capture config — see
`ALWAYS_ENABLED_EVENT_TYPES` in `adapters/_base/capture.py`.

## BYOK

CrewAI agents instantiate their own LLM clients (LangChain or LiteLLM
under the hood). The CrewAI adapter does not own those keys. Keep your
provider keys in the standard env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.) — atlas-app's BYOK layer applies separately
to the platform-owned model gateway.

## Backward compatibility

```python
from layerlens.instrument.adapters.frameworks.crewai import STRATIXCrewCallback
```

`STRATIXCrewCallback` is a deprecation alias for `LayerLensCrewCallback`
preserved for users coming from the legacy ``stratix`` (ateam) namespace.
It will be removed in v2.0.
