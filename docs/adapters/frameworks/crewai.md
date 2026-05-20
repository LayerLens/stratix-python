# CrewAI adapter

Instruments [CrewAI](https://github.com/crewAIInc/crewAI) crews via CrewAI's
typed event bus (CrewAI ≥ 1.0). Earlier 0.x event-bus versions are also
handled via the dispatcher fallback.

## Install

```bash
pip install layerlens[crewai]
```

Pulls `crewai>=0.30.0`. CrewAI 0.30+ is Pydantic v2-only
(`requires_pydantic="2"` on the adapter).

## Usage

```python
from crewai import Agent, Crew, Task
from layerlens.instrument.adapters.frameworks import CrewAIAdapter

adapter = CrewAIAdapter(client=layerlens_client)
adapter.connect()                # registers handlers on CrewAI's event bus

crew = Crew(agents=[...], tasks=[...])
crew.kickoff()

adapter.disconnect()             # tears down handlers when done
```

## Event surface

- `agent.start` / `agent.end` per agent step.
- `task.start` / `task.end` per Crew task.
- `tool.call` for every tool invocation, with the tool name + arguments.
- `model.invoke` for the underlying LLM calls (provider-aware via the
  CrewAI agent's `llm` attribute).
- `agent.handoff` when CrewAI delegates between agents.

Thread-safety: CrewAI dispatches handlers across threads, so the adapter
manages collector and span state on the instance rather than via
ContextVars.

## Sample

[`samples/instrument/crewai/example.py`](../../../samples/instrument/crewai/example.py)

## Compat

- CrewAI 0.30+ (Pydantic v2-only)
- Python 3.9+
