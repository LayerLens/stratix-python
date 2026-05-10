# AG-UI protocol instrumentation sample

End-to-end demo of `AGUIAdapter` — simulates an AG-UI (Agent-User
Interface) SSE event stream entirely in-memory. The sample drives the
adapter through a text-message stream, a state delta, and a tool call,
and ships the resulting LayerLens events to atlas-app via
`HttpEventSink`.

## Prerequisites

```bash
pip install 'layerlens[protocols-agui]'
```

(The `protocols-agui` extra is empty — no peer library is required.
The sample exercises the AG-UI event mapper directly.)

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.agui.main
```

## Expected output

The sample enables `CaptureConfig.l6b_protocol_streams` so per-chunk
text events are captured, then drives nine AG-UI events through
`on_agui_event`. The event mapper in
`adapters/protocols/agui/event_mapper.py` maps each AG-UI event to
its LayerLens event-type string:

| AG-UI event             | Emitted as              |
|-------------------------|-------------------------|
| `TEXT_MESSAGE_START`    | `protocol.stream.event` |
| `TEXT_MESSAGE_CONTENT`  | `protocol.stream.event` |
| `TEXT_MESSAGE_END`      | `protocol.stream.event` |
| `STATE_DELTA`           | `agent.state.change`    |
| `TOOL_CALL_START`       | `tool.call`             |
| `TOOL_CALL_END`         | `protocol.stream.event` |

The sample prints a one-line summary and the sink's `batches_sent`
counter on the way out.

## What this demonstrates

- `AGUIAdapter.connect()` brings the adapter to `HEALTHY` with no peer
  AG-UI library installed.
- Mutating `CaptureConfig.standard()` to set `l6b_protocol_streams =
  True` so per-chunk text-message events are not filtered out.
- A three-chunk text-message stream (`TEXT_MESSAGE_START` →
  `TEXT_MESSAGE_CONTENT × 3` → `TEXT_MESSAGE_END`).
- A `STATE_DELTA` event carrying a JSON patch.
- A `TOOL_CALL_START` / `TOOL_CALL_END` pair.
- `HttpEventSink` batching with `max_batch=10`, `flush_interval_s=1.0`.
- Clean shutdown via `adapter.disconnect()` + `sink.close()` in a
  `finally` block.

The full AG-UI event-to-LayerLens mapping table is in
[`docs/adapters/protocols-agui.md`](../../../docs/adapters/protocols-agui.md).
