# MCP Extensions protocol adapter

`layerlens.instrument.adapters.protocols.mcp.MCPExtensionsAdapter` instruments
the [Model Context Protocol](https://modelcontextprotocol.io/) extensions
introduced in 2025: elicitation, structured tool outputs, async tasks,
MCP apps, and OAuth 2.1 authentication.

The base MCP tool/resource calls are already covered by the host's
runtime (e.g. the `mcp` Python SDK); this adapter focuses on the
extensions that introduce new shapes the framework does not natively
trace.

## Install

```bash
pip install 'layerlens[protocols-mcp]'
```

Pulls `mcp>=0.9`. Requires Python 3.10+.

## Quick start

```python
from layerlens.instrument.adapters.protocols.mcp import MCPExtensionsAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="mcp_extensions")
adapter = MCPExtensionsAdapter()
adapter.add_sink(sink)
adapter.connect()

# Trace a tool call
adapter.on_tool_call(
    tool_name="search",
    input_data={"q": "weather in NYC"},
    output_data={"result": "..."},
    latency_ms=42.5,
)

# Trace structured output validation
adapter.on_structured_output(
    tool_name="search",
    output={"result": "..."},
    schema={"type": "object", "$id": "search-result-v1"},
    validation_passed=True,
)

adapter.disconnect()
sink.close()
```

## What's wrapped

`MCPExtensionsAdapter` exposes a set of `on_*` hooks. The host MCP server
calls them at the appropriate extension points:

- `on_tool_call(tool_name, input_data, output_data, error, latency_ms)` —
  per tool invocation.
- `on_structured_output(tool_name, output, schema, validation_passed,
  validation_errors)` — per structured-output validation.
- `on_elicitation_request(prompt_id, prompt, schema)` — when the server
  prompts the user for structured input.
- `on_elicitation_response(prompt_id, response, valid)` — user reply.
- `on_async_task(task_id, status, ...)` — long-running tool task lifecycle.
- `on_mcp_app_invocation(app_id, ...)` — interactive UI components
  invoked as tools.
- `on_auth_event(event_type, ...)` — OAuth 2.1 / OpenID Connect events
  inside an MCP session.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `tool.call` | L5a | Per `on_tool_call`. |
| `protocol.structured_output` | L4a | Per `on_structured_output`. |
| `protocol.elicitation_request` | L4a | Per `on_elicitation_request`. |
| `protocol.elicitation_response` | L4a | Per `on_elicitation_response`. |
| `protocol.async_task` | L4a | Per `on_async_task`. |
| `protocol.mcp_app_invocation` | L4a | Per `on_mcp_app_invocation`. |
| `protocol.auth_event` | cross-cutting | Per `on_auth_event`. |
| `policy.violation` | cross-cutting | When `validation_passed=False`. |

## MCP specifics

- **Schema hashing**: structured-output schemas are sha256-hashed and
  the hash is stored on the event. If the schema declares `$id`, that
  identifier is captured separately.
- **Output hashing**: tool outputs are likewise sha256-hashed for
  reproducibility verification without storing the full body.
- **Procedural memory**: when a `memory_service=` is passed to the
  constructor, recurring tool patterns are stored as procedural memory
  entries (`memory_type="procedural"`, importance 0.4). Failures are
  swallowed.
- **Async tasks**: `on_async_task` accepts `status` values
  (`pending|running|completed|failed`) and computes duration on the
  terminal transition.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# All MCP extension events are L4a / L5a — covered by standard.
adapter = MCPExtensionsAdapter(capture_config=CaptureConfig.standard())
```

## BYOK

OAuth 2.1 client credentials live with the host MCP server. The adapter
does not own them. For platform-managed BYOK see
`docs/adapters/byok.md`.
