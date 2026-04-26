# AutoGen framework adapter

`layerlens.instrument.adapters.frameworks.autogen.AutoGenAdapter` instruments
[Microsoft AutoGen](https://github.com/microsoft/autogen) `ConversableAgent`
objects, capturing message exchange, LLM calls, code execution, and group-chat
turns.

## Install

```bash
pip install 'layerlens[autogen]'
```

Pulls `pyautogen>=0.2,<0.5`.

## Quick start

```python
from autogen import AssistantAgent, UserProxyAgent

from layerlens.instrument.adapters.frameworks.autogen import (
    AutoGenAdapter,
    instrument_agents,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="autogen")
adapter = AutoGenAdapter()
adapter.add_sink(sink)
adapter.connect()

assistant = AssistantAgent(name="assistant", llm_config={"model": "gpt-4o-mini"})
user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)

adapter.connect_agents(assistant, user)
user.initiate_chat(assistant, message="What is 2+2?", max_turns=1)

adapter.disconnect()
sink.close()
```

`instrument_agents(*agents)` is the one-line equivalent of the three-line
adapter setup above.

## What's wrapped

`adapter.connect_agents(*agents)` monkey-patches the following on each
`ConversableAgent`:

- `send` — emits `agent.input` for the outgoing message.
- `receive` — emits `agent.output` for the incoming message.
- `generate_reply` — emits `model.invoke` and any `tool.call` events.
- `execute_code_blocks` — emits `agent.code` and `tool.call` for code
  execution (when present).

The originals are stashed on the adapter and restored on `disconnect()`.
A `GroupChatTracer` wires similar hooks onto `GroupChatManager`, and a
`HumanProxyTracer` adds `agent.handoff` semantics for human-in-the-loop
proxies.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First time each agent is seen. |
| `agent.input` | L1 | Every `send`. |
| `agent.output` | L1 | Every `receive`. |
| `agent.action` | L4a | Per `generate_reply` decision. |
| `agent.code` | L2 | When `execute_code_blocks` runs and `l2_agent_code` is enabled. |
| `agent.handoff` | L4a | Group-chat speaker selection / human handoff. |
| `agent.state.change` | cross-cutting | Conversation history mutations. |
| `tool.call` | L5a | Per function-call inside `generate_reply`. |
| `model.invoke` | L3 | Per LLM call. |

## AutoGen specifics

- **Multi-agent attribution**: `agent_name`, `recipient_name`, and
  `message_seq` (a monotonic counter) are included on every event so the
  full chat can be reconstructed in order.
- **Group chats**: `GroupChatTracer` registers as a callback on
  `GroupChatManager`, capturing the speaker-selection turns. Pass a
  `GroupChatManager` to `connect_agents` alongside the participants.
- **Code execution**: when an agent runs code blocks, the language and
  truncated code body emit `agent.code` (only if
  `CaptureConfig.l2_agent_code` is enabled).

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = AutoGenAdapter(capture_config=CaptureConfig.standard())

# Production-light: skip the verbose code-execution events.
adapter = AutoGenAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l4a_environment_config=True,
        l5a_tool_calls=True,
        l2_agent_code=False,
    ),
)
```

## BYOK

AutoGen reads its `llm_config` to instantiate provider clients. The adapter
does not own those keys. For platform-managed BYOK see
`docs/adapters/byok.md` (atlas-app M1.B).
