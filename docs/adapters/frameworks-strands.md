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
| `environment.config` | L4a | First wrap of each agent (`lifecycle.py:430`). |
| `agent.input` | L1 | Beginning of every `__call__` / `invoke` (`lifecycle.py:272`). |
| `agent.output` | L1 | End of every `__call__` / `invoke` (`lifecycle.py:307`). |
| `agent.state.change` | cross-cutting | Mid-run state mutations and run completion (`lifecycle.py:249,308`). |
| `tool.call` | L5a | Per Strands tool invocation (`lifecycle.py:223,341`). |
| `model.invoke` | L3 | Per LLM call — Strands routes through Bedrock (`lifecycle.py:188,371`). |
| `cost.record` | cross-cutting | Per LLM call when token usage is present (`lifecycle.py:209`). |

## Strands specifics

- **Bedrock-native**: every `model.invoke` payload includes the Bedrock
  `modelId` and the conversation `inferenceConfig`. Token usage is parsed
  from the Bedrock response shape.
- **Tools**: Strands tools registered via the `@tool` decorator surface
  their function name and JSON schema in `tool.call.tool_schema`.
- **Loops**: Strands runs a reasoning loop (think → act → observe). Each
  iteration is observable via the inner `model.invoke` and `tool.call`
  events the adapter captures from Strands' callback hooks — there is no
  separate per-iteration loop event.
- **State changes**: mid-run state mutations emit `agent.state.change`
  (`lifecycle.py:249`) and run completion emits a terminal
  `agent.state.change` alongside `agent.output` (`lifecycle.py:308`).

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
