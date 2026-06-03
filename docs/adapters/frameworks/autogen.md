# AutoGen adapter

Instruments [AutoGen](https://github.com/microsoft/autogen) agents and teams via
AutoGen's structured event logging API (autogen-core ≥ 0.4).

## Install

```bash
pip install layerlens[autogen]
```

Pulls `autogen-agentchat>=0.4.0` (and `autogen-core` as a transitive dep).

## Usage

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from layerlens.instrument.adapters.frameworks import AutoGenAdapter

adapter = AutoGenAdapter(client=layerlens_client)
adapter.connect()                # attaches a logging.Handler to autogen_core.events

async def run():
    team = RoundRobinGroupChat([agent_a, agent_b])
    await team.run(task="...")

asyncio.run(run())
adapter.disconnect()             # removes the handler and flushes the trace
```

## Event surface

The adapter listens for AutoGen's structured event classes and emits:

- `model.invoke` for `LLMCallEvent` and `LLMStreamEndEvent` (provider-aware,
  pulls the model name from the response payload).
- `tool.call` for `ToolCallEvent`, including tool name and arguments.
- `agent.message` for `MessageEvent` between participants.
- `agent.error` for `MessageDroppedEvent`, `MessageHandlerExceptionEvent`,
  and `AgentConstructionExceptionEvent`.
- `conversation.ended` per topic/session when the trace tears down, with the
  participant set, message count, and turn count.

Thread-safety: AutoGen dispatches log events from any thread, so the adapter
holds the collector and run state on the instance rather than via ContextVars.

## Sample

[`samples/instrument/autogen/example.py`](../../../samples/instrument/autogen/example.py)

## Compat

- autogen-agentchat 0.4+ (autogen-core 0.4+)
- Python 3.9+
