# AWS Bedrock Agents framework adapter

`layerlens.instrument.adapters.frameworks.bedrock_agents.BedrockAgentsAdapter`
instruments AWS Bedrock Agent runtime calls by registering boto3 event hooks
and parsing the `InvokeAgent` response stream's `trace` blocks.

## Install

```bash
pip install 'layerlens[bedrock-agents]'
```

Pulls `boto3>=1.34`. AWS credentials and region must be configured the
standard way (env vars, IAM role, profile).

## Quick start

```python
import boto3

from layerlens.instrument.adapters.frameworks.bedrock_agents import (
    BedrockAgentsAdapter,
    instrument_client,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="bedrock_agents")
adapter = BedrockAgentsAdapter()
adapter.add_sink(sink)
adapter.connect()

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
adapter.instrument_client(client)

response = client.invoke_agent(
    agentId="ABCDEFGHIJ",
    agentAliasId="TSTALIASID",
    sessionId="my-session",
    inputText="What is 2+2?",
)
# Iterate the response stream — trace events are captured automatically.
for chunk in response["completion"]:
    pass

adapter.disconnect()
sink.close()
```

`instrument_client(client)` is the convenience helper.

## What's wrapped

`adapter.instrument_client(client)` registers two boto3 event hooks on the
provided `bedrock-agent-runtime` client:

- `provide-client-params.bedrock-agent-runtime.InvokeAgent` — fires before
  the request goes out. Captures `agentId`, `sessionId`, `inputText`,
  emits `agent.input` and `environment.config` on first agent encounter.
- `after-call.bedrock-agent-runtime.InvokeAgent` — fires after the response
  comes back. Walks the `trace` blocks in the streamed events and emits
  `model.invoke` / `tool.call` / `agent.action` per trace step.

`disconnect()` unregisters both hooks.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First `InvokeAgent` per `agentId`. |
| `agent.input` | L1 | Beginning of every `InvokeAgent`. |
| `agent.output` | L1 | End of every `InvokeAgent` (after stream consumption). |
| `agent.action` | L4a | Per `orchestrationTrace.modelInvocationInput` block. |
| `agent.handoff` | L4a | Per cross-agent collaboration step. |
| `tool.call` | L5a | Per `actionGroupInvocationInput` / `knowledgeBaseLookupInput` block. |
| `model.invoke` | L3 | Per `modelInvocationOutput` block (with token usage). |

## Bedrock Agents specifics

- **Action groups**: each `actionGroup` invocation maps to a `tool.call`
  with `tool_name = "{actionGroupName}::{apiPath}"` and the typed
  parameters in the payload.
- **Knowledge bases**: every KB lookup emits a `tool.call` with
  `tool_name = "knowledge_base::{knowledgeBaseId}"` and the rendered
  query + retrieved citations.
- **Multi-agent collaboration**: when a supervisor agent delegates to a
  collaborator, an `agent.handoff` event is emitted with both agent IDs.
- **Session attributes**: passed through into `agent.input` payloads as
  `session_attributes`.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = BedrockAgentsAdapter(capture_config=CaptureConfig.standard())

# Compliance: drop user input/output content but keep tool/model metadata.
adapter = BedrockAgentsAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

Bedrock Agents bills directly to your AWS account via your IAM identity.
There's no separate API key to manage. The model used by the agent is
configured server-side in the agent definition.
