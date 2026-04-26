# AG-UI protocol adapter

`layerlens.instrument.adapters.protocols.agui.AGUIAdapter` instruments
the [AG-UI (Agent-User Interaction) protocol](https://github.com/ag-ui-protocol/ag-ui)
by intercepting the SSE event stream between an agent backend and a
frontend client.

## Install

```bash
pip install 'layerlens[protocols-agui]'
```

Pulls `ag-ui>=0.1`. Requires Python 3.10+.

## Quick start

```python
from layerlens.instrument.adapters.protocols.agui import AGUIAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="agui")
adapter = AGUIAdapter()
adapter.add_sink(sink)
adapter.connect()

# Each AG-UI SSE event becomes a LayerLens event:
adapter.on_agui_event(
    "TEXT_MESSAGE_START",
    payload={"thread_id": "thread-1", "message_id": "msg-1"},
)
adapter.on_agui_event(
    "TEXT_MESSAGE_CONTENT",
    payload={"thread_id": "thread-1", "content": "Hello"},
)
adapter.on_agui_event(
    "TEXT_MESSAGE_END",
    payload={"thread_id": "thread-1"},
)

adapter.disconnect()
sink.close()
```

In production the adapter is wired in as ASGI/WSGI middleware around the
agent's SSE handler — see `agui/middleware.py`.

## What's wrapped

`AGUIAdapter.on_agui_event(event_type, payload)` is the single hook the
host calls per SSE event. The adapter routes the event to the appropriate
Stratix event type via `event_mapper.map_agui_to_stratix`.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `protocol.stream.event` | L6b | Per AG-UI SSE event (gated by `l6b_protocol_streams` for high-frequency content). |
| `agent.state.change` | cross-cutting | Per `STATE_SNAPSHOT` / `STATE_DELTA` / lifecycle event. |
| `tool.call` | L5a | Per `TOOL_CALL_START` / `TOOL_CALL_END`. |

## AG-UI specifics

- **Text message buffering**: `TEXT_MESSAGE_START` → `TEXT_MESSAGE_CONTENT*`
  → `TEXT_MESSAGE_END` is buffered into a single `full_text` field on the
  end event. Per-chunk events are gated by `CaptureConfig.l6b_protocol_streams`.
- **Payload hashing**: every emitted `protocol.stream.event` contains a
  sha256 of the original payload, so the platform can verify reproducibility
  without storing the full body.
- **State diffing**: `STATE_DELTA` events compute `before_hash` /
  `after_hash` and update an in-memory cache for the thread.
- **Custom events**: AG-UI's `CUSTOM_EVENT` type is preserved verbatim in
  the `payload_summary` field (truncated to 200 chars).

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Standard preset captures lifecycle + tool events but suppresses the
# high-frequency text-content stream.
adapter = AGUIAdapter(capture_config=CaptureConfig.standard())

# Full content (debug builds only).
adapter = AGUIAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l5a_tool_calls=True,
        l6b_protocol_streams=True,
    ),
)
```

## BYOK

AG-UI is transport-only. There are no model API keys involved at the
protocol layer.
