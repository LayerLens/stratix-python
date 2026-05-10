# A2A protocol instrumentation sample

End-to-end demo of `A2AAdapter` — simulates a Google A2A (Agent-to-Agent)
exchange entirely in-memory, with no real A2A server or network call.
The sample walks an agent-card registration, a task submission, a single
SSE stream event, and a task completion through the adapter hook surface.

## Prerequisites

```bash
pip install 'layerlens[protocols-a2a]'
```

(The `protocols-a2a` extra only pulls `httpx` for the HTTP sink. No
peer A2A library is required to run the sample — the hook methods are
called directly.)

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.a2a.main
```

## Expected output

The sample emits four protocol events through `HttpEventSink`, then
prints a one-line summary and the sink's batch-send count. Event-type
strings come straight from `_vendored/events_protocol.py`:

| Event                      | Source hook                              |
|----------------------------|------------------------------------------|
| `protocol.agent_card`      | `adapter.register_agent_card(...)`       |
| `protocol.task.submitted`  | `adapter.on_task_submitted(...)`         |
| `protocol.stream.event`    | `adapter.on_stream_event(...)`           |
| `protocol.task.completed`  | `adapter.on_task_completed(...)`         |

## What this demonstrates

- `A2AAdapter.connect()` brings the adapter to `HEALTHY` with no real
  A2A SDK present.
- `register_agent_card` emits `protocol.agent_card` from a synthetic
  agent-card dict (skills, capabilities, well-known URL).
- `on_task_submitted` records an outbound task for `task-001`.
- `on_stream_event` records an in-flight SSE progress update for the
  same task.
- `on_task_completed` closes the task with an artifact payload.
- `HttpEventSink` batches events (`max_batch=10`, `flush_interval_s=1.0`)
  and the sample reads `sink.stats()["batches_sent"]` on the way out.
- `adapter.disconnect()` and `sink.close()` run in a `finally` block, so
  the sample is safe to interrupt.

The full mapping from each hook to the canonical event name is in
[`docs/adapters/protocols-a2a.md`](../../../docs/adapters/protocols-a2a.md).
