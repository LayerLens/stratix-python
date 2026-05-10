# Agno framework adapter

`layerlens.instrument.adapters.frameworks.agno.AgnoAdapter` instruments
[Agno](https://github.com/agno-agi/agno) agents — single-agent and
multi-agent teams — by wrapping `Agent.run()` and `Agent.arun()`.

## Install

```bash
pip install 'layerlens[agno]'
```

Pulls `agno>=0.1,<1.0`. Requires Python 3.10+.

## Quick start

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter, instrument_agent
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="agno")
adapter = AgnoAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), instructions="Be concise.")
adapter.instrument_agent(agent)

response = agent.run("What is 2 + 2?")

adapter.disconnect()
sink.close()
```

`instrument_agent(agent)` is the one-liner equivalent.

## What's wrapped

`adapter.instrument_agent(agent)` patches the following on each Agent:

- `run` — sync entry point. Emits `agent.input` + `agent.output` and any
  inner `model.invoke` / `tool.call` events.
- `arun` — async entry point. Same semantics.
- `_run_tool` — emits `tool.call` per tool invocation (when present in the
  Agno version).
- Model adapter hooks — emit `model.invoke` per LLM call.

`disconnect()` restores all originals.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First `run` per agent (`lifecycle.py:462`). |
| `agent.input` | L1 | Beginning of every `run` / `arun` (`lifecycle.py:289`). |
| `agent.output` | L1 | End of every `run` / `arun` (`lifecycle.py:324`). |
| `agent.handoff` | L4a | When a team agent delegates to a sub-agent (`lifecycle.py:267,398`). |
| `agent.state.change` | cross-cutting | Run completion / failure (`lifecycle.py:325`). |
| `tool.call` | L5a | Per tool invocation (`lifecycle.py:246,358`). |
| `model.invoke` | L3 | Per LLM call (`lifecycle.py:216,388`). |
| `cost.record` | cross-cutting | Per LLM call when `result.metrics` / `result.usage` are populated (`lifecycle.py:228`). |

## Agno specifics

- **Teams**: Agno supports multi-agent teams via `Team(agents=[...])`.
  Each team member must be instrumented individually with
  `adapter.instrument_agent(team_member)` — or call
  `instrument_agent(team)` and the convenience helper recurses.
- **Reasoning agents**: Agno's `reasoning=True` mode is not specially
  instrumented — intermediate reasoning steps are not surfaced as
  separate events. The reasoning chain is observable only via the
  inputs/outputs already captured on `agent.input` / `agent.output` and
  any inner `model.invoke` / `tool.call` events the underlying agent
  emits.
- **Storage backends**: Agno session storage (Postgres, sqlite, Redis,
  etc.) emits `agent.state.change` on every save.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = AgnoAdapter(capture_config=CaptureConfig.standard())

# Heavy: enable all capture layers (L2 agent.code currently has no
# Agno-specific emission sites — reasoning is not specially instrumented).
adapter = AgnoAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l2_agent_code=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
    ),
)
```

## BYOK

Agno model adapters (`OpenAIChat`, `AnthropicClaude`, etc.) read their own
credentials. The Agno adapter does not own them. For platform-managed
BYOK see `docs/adapters/byok.md` (atlas-app M1.B).
