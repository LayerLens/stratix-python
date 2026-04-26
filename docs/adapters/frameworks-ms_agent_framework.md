# Microsoft Agent Framework adapter

`layerlens.instrument.adapters.frameworks.ms_agent_framework.MSAgentAdapter`
instruments [Microsoft Agent Framework](https://learn.microsoft.com/en-us/semantic-kernel/agents/)
(Semantic Kernel Agents) by wrapping `AgentChat.invoke()` and
`AgentGroupChat.invoke()`.

## Install

```bash
pip install 'layerlens[ms-agent-framework]'
```

Pulls `semantic-kernel>=1.0,<2.0` (Semantic Kernel hosts the agents API).
Requires Python 3.10+.

## Quick start

```python
import asyncio
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    MSAgentAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="ms_agent_framework")
adapter = MSAgentAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(ai_model_id="gpt-4o-mini"),
    name="answerer",
    instructions="Be concise.",
)
adapter.instrument_chat(agent)

async def main() -> None:
    async for response in agent.invoke("What is 2+2?"):
        print(response.content)

asyncio.run(main())

adapter.disconnect()
sink.close()
```

`instrument_agent(chat)` is the convenience helper.

## What's wrapped

`adapter.instrument_chat(chat_or_agent)` wraps the framework's invocation
surfaces:

- `invoke` — async generator returning the agent's responses.
- `invoke_stream` — async generator returning streaming chunks (when
  present in the installed version).

Both wrappers emit lifecycle events around the call and capture inner
`tool.call` and `model.invoke` events from the underlying Semantic Kernel
filters. `disconnect()` restores the originals.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First wrap of each chat. |
| `agent.input` | L1 | Beginning of every `invoke` / `invoke_stream`. |
| `agent.output` | L1 | End of every invocation (per response). |
| `agent.action` | L4a | Per intermediate step. |
| `agent.handoff` | L4a | Per `AgentGroupChat` speaker turn. |
| `tool.call` | L5a | Per plugin function invocation. |
| `model.invoke` | L3 | Per LLM call. |

## MS Agent Framework specifics

- **`AgentChat` vs `AgentGroupChat`**: both support the same
  `invoke()` signature; group chats additionally emit `agent.handoff`
  on each speaker turn.
- **Plugins**: Semantic Kernel plugin functions surface as `tool.call` —
  the plugin name + function name combine into `tool_name`.
- **Multi-agent terminations**: configurable termination strategies
  emit `agent.action` with `terminate_reason` when a group chat ends.
- **Streaming**: `invoke_stream` emits one consolidated `model.invoke`
  on stream completion; per-chunk text is accumulated.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = MSAgentAdapter(capture_config=CaptureConfig.standard())

# Drop content for compliance.
adapter = MSAgentAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

Microsoft Agent Framework uses Semantic Kernel connectors
(`OpenAIChatCompletion`, `AzureChatCompletion`, etc.) for model access.
The adapter does not own those credentials. For platform-managed BYOK
see `docs/adapters/byok.md` (atlas-app M1.B).
