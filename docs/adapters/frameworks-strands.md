# AWS Strands framework adapter

`layerlens.instrument.adapters.frameworks.strands.StrandsAdapter` instruments
[AWS Strands](https://github.com/strands-agents/sdk-python) agents by
wrapping `Agent.__call__` and `Agent.invoke`.

## Install

```bash
pip install 'layerlens[strands]'
```

Pulls `strands-agents>=0.1,<1.0`. Requires Python 3.10+. AWS credentials
must be provisioned the standard way (env, IAM role, profile) since Strands
runs against Bedrock under the hood.

## Quick start

```python
from strands import Agent

from layerlens.instrument.adapters.frameworks.strands import (
    StrandsAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="strands")
adapter = StrandsAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = Agent(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
adapter.instrument_agent(agent)

response = agent("What is 2 + 2?")

adapter.disconnect()
sink.close()
```

`instrument_agent(agent)` is the convenience helper.

## What's wrapped

`adapter.instrument_agent(agent)` wraps both invocation surfaces:

- `__call__` — the primary entry point (`agent("question")`).
- `invoke` — alternative entry point present in some Strands versions.

Both wrappers emit lifecycle events around the call and capture inner
`tool.call` and `model.invoke` events from Strands' internal callback
hooks. `disconnect()` restores the originals.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First wrap of each agent. |
| `agent.input` | L1 | Beginning of every `__call__` / `invoke`. |
| `agent.output` | L1 | End of every `__call__` / `invoke`. |
| `agent.action` | L4a | Per intermediate reasoning loop iteration. |
| `agent.handoff` | L4a | Multi-agent collaboration handoffs. |
| `tool.call` | L5a | Per Strands tool invocation. |
| `model.invoke` | L3 | Per LLM call (Strands routes these through Bedrock). |

## Strands specifics

- **Bedrock-native**: every `model.invoke` payload includes the Bedrock
  `modelId` and the conversation `inferenceConfig`. Token usage is parsed
  from the Bedrock response shape.
- **Tools**: Strands tools registered via the `@tool` decorator surface
  their function name and JSON schema in `tool.call.tool_schema`.
- **Loops**: Strands runs a reasoning loop (think → act → observe). Each
  loop iteration emits an `agent.action` with `loop_index` and a copy of
  the conversation state.
- **Multi-agent**: Strands supports orchestrator/worker patterns; cross-agent
  delegation emits `agent.handoff` with `source_agent` + `target_agent`.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = StrandsAdapter(capture_config=CaptureConfig.standard())

# Drop conversation content for compliance.
adapter = StrandsAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

Strands authenticates against AWS using the standard boto3 credential
chain (env / profile / IAM role). There's no separate API key. The Bedrock
model used by the agent is configured at construction time via the
`model` parameter.
