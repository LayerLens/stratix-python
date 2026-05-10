# MCP Extensions protocol instrumentation sample

End-to-end demo of `MCPExtensionsAdapter` — walks the four
Model Context Protocol 2025 extensions in-memory: tool calls,
structured-output validation, elicitation request/response, and
async-task lifecycle. The sample requires no real MCP server.

## Prerequisites

```bash
pip install 'layerlens[protocols-mcp]'
```

The `protocols-mcp` extra pulls `mcp>=0.9` on Python 3.10+, but the
sample only exercises adapter hook methods — it does not import the
peer `mcp` package.

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.mcp.main
```

## Expected output

The sample drives six adapter hook calls. Event-type strings come
straight from `_vendored/events_protocol.py`:

| Event                              | Source hook                                |
|------------------------------------|--------------------------------------------|
| `tool.call`                        | `adapter.on_tool_call(...)`                |
| `protocol.tool.structured_output`  | `adapter.on_structured_output(...)`        |
| `protocol.elicitation.request`     | `adapter.on_elicitation_request(...)`      |
| `protocol.elicitation.response`    | `adapter.on_elicitation_response(...)`     |
| `protocol.async_task` (×2)         | `adapter.on_async_task(...)` (start + end) |

The sample prints a one-line summary and the sink's `batches_sent`
counter on the way out.

## What this demonstrates

- `MCPExtensionsAdapter.connect()` brings the adapter to `HEALTHY`.
- A `tool.call` event for a `weather.get` invocation with latency.
- A `protocol.tool.structured_output` event with the schema and
  `validation_passed=True`.
- An elicitation request/response pair (`elic-1`) — the server-side
  prompt and the client reply, both with the same `prompt_id`.
- A two-step async task lifecycle: `running` then `completed`, with the
  completion event carrying `duration_ms=1500.0`.
- `HttpEventSink` batching with `max_batch=10`, `flush_interval_s=1.0`.
- Clean shutdown via `adapter.disconnect()` + `sink.close()` in a
  `finally` block.

The full mapping from each hook to the canonical event name is in
[`docs/adapters/protocols-mcp.md`](../../../docs/adapters/protocols-mcp.md).
