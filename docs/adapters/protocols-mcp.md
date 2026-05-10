# MCP Extensions protocol adapter

> **Canonical source:** [`protocols/mcp/adapter.py`](../../src/layerlens/instrument/adapters/protocols/mcp/adapter.py)
> and [`_vendored/events_protocol.py`](../../src/layerlens/instrument/_vendored/events_protocol.py).
> Every event-name string and method signature in this doc matches the literal
> `event_type` default and Python signature at source.

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
- `on_elicitation_request(elicitation_id, server_name, schema, title)` —
  when an MCP server initiates a request for structured user input
  (`adapter.py:212-218`).
- `on_elicitation_response(elicitation_id, action, response, latency_ms)` —
  user reply (`adapter.py:237-243`). `action` is `"submit"` or `"cancel"`.
- `on_async_task(async_task_id, status, ...)` — long-running tool task
  lifecycle (`adapter.py:259-267`).
- `on_mcp_app_invocation(app_id, component_type, interaction_result, ...)` —
  interactive UI components invoked as tools (`adapter.py:292-299`).
- `on_auth_event(auth_type, success, details)` — OAuth 2.1 / OpenID Connect
  events inside an MCP session (`adapter.py:319-324`). Surfaced as
  `environment.config` (not a `protocol.*` event) with `auth_event` and
  `auth_success` attributes on the payload.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `tool.call` | L5a | Per `on_tool_call`. |
| `protocol.tool.structured_output` | L5a | Per `on_structured_output` (`events_protocol.py:413-415`). |
| `protocol.elicitation.request` | L5a | Per `on_elicitation_request` (`events_protocol.py:336-338`). |
| `protocol.elicitation.response` | L5a | Per `on_elicitation_response` (`events_protocol.py:375-378`). |
| `protocol.async_task` | cross-cutting | Per `on_async_task` (`events_protocol.py:236-238`). |
| `protocol.mcp_app.invocation` | L5a | Per `on_mcp_app_invocation` (`events_protocol.py:457-459`). |
| `environment.config` | L4 | Per `on_auth_event` — carries `auth_event` and `auth_success` in `attributes` (`adapter.py:331-338`). |

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
