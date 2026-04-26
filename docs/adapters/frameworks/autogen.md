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

from layerlens import LayerLens
from layerlens.instrument.adapters.frameworks.autogen import (
    AutoGenAdapter,
    instrument_agents,
)

client = LayerLens()  # picks up LAYERLENS_STRATIX_API_KEY / _BASE_URL

adapter = AutoGenAdapter(stratix=client)
adapter.connect()

assistant = AssistantAgent(name="assistant", llm_config={"model": "gpt-4o-mini"})
user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)

adapter.connect_agents(assistant, user)
user.initiate_chat(assistant, message="What is 2+2?", max_turns=1)

adapter.disconnect()
```

`instrument_agents(assistant, user, stratix=client)` is the one-line
equivalent of the `AutoGenAdapter(...).connect()` + `connect_agents(...)`
sequence above.

For an offline / no-API-key demonstration, see
`samples/instrument/autogen/main.py` — it wires the adapter against an
in-process `EventSink` and a duck-typed agent so you can inspect the
emitted event stream without any external services.

## What's wrapped

`adapter.connect_agents(*agents)` monkey-patches the following on each
`ConversableAgent`:

- `send` — emits `agent.handoff` for the outgoing message.
- `receive` — emits `agent.state.change` for the incoming message.
- `generate_reply` — emits `model.invoke` (with token usage when
  available).
- `execute_code_blocks` — emits `tool.call` and `tool.environment` for
  the code execution.

The originals are stashed on the adapter and restored on `disconnect()`.
A `GroupChatTracer` wires similar hooks onto `GroupChatManager`, and a
`HumanProxyTracer` traces `get_human_input` for human-in-the-loop
proxies (emitting `agent.input` events with `role: "HUMAN"`).

## Events emitted

| Event                | Layer          | When                                                                     |
|----------------------|----------------|--------------------------------------------------------------------------|
| `environment.config` | L4a            | First time each agent is seen by `connect_agents`.                       |
| `agent.input`        | L1             | `on_conversation_start` and human-input requests.                        |
| `agent.output`       | L1             | `on_conversation_end` and group-chat termination.                        |
| `agent.handoff`      | cross-cutting  | Every `send` (carries `from_agent`, `to_agent`, `message_seq`).          |
| `agent.state.change` | cross-cutting  | Every `receive` (carries `agent`, `from_agent`, `message_preview`).      |
| `agent.code`         | L2             | Group-chat speaker selection (via `GroupChatTracer.on_speaker_selected`).|
| `tool.call`          | L5a            | Each `execute_code_blocks` call (`tool_name: "code_execution"`).         |
| `tool.environment`   | L5c            | Each `execute_code_blocks` call (execution context details).             |
| `model.invoke`       | L3             | Each `generate_reply` (with token usage / latency / messages).           |

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
