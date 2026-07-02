# Snowflake Cortex Agents adapter

Instruments [Snowflake Cortex Agents](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents)
— Snowflake's governed, server-side data agents (GA November 2025). Agents are
invoked through the `agent:run` REST API, which streams its response back as
server-sent events (SSE). Unlike the in-process framework adapters, there is no
Python object to hook, so this adapter observes the SSE stream directly.

## Install

```bash
pip install layerlens[snowflake]
```

Pulls `httpx>=0.27.0` for the `agent:run` client. Snowflake authentication
(programmatic access token, OAuth, or key-pair JWT) is the caller's
responsibility — the adapter only attaches the token as a bearer credential.

## Usage

There are two ways to use the adapter.

### Instrumented invoke

Let the adapter make the `agent:run` call and turn the stream into a trace:

```python
import os
from layerlens.instrument.adapters.frameworks import SnowflakeCortexAgentsAdapter

adapter = SnowflakeCortexAgentsAdapter(client=layerlens_client)
adapter.connect(
    account_url="https://ACCOUNT.snowflakecomputing.com",
    auth_token=os.environ["SNOWFLAKE_TOKEN"],
    agent="MY_DB.MY_SCHEMA.MY_AGENT",   # omit to use the stateless /api/v2/cortex/agent:run
)

final = adapter.run("What were Q3 sales by region?")

adapter.disconnect()
```

`run()` accepts a plain string or a full `messages` array, plus optional
`model`, `tools`, `thread_id`, and `parent_message_id`. It returns the final
`response` event payload.

### Bring your own call

If you already POST to `agent:run` yourself (via `httpx`, `requests`, or the
Snowflake connector's REST transport), hand the adapter the SSE stream — either
the raw lines or pre-parsed `(event, data)` tuples:

```python
import httpx

adapter.connect(agent="MY_DB.MY_SCHEMA.MY_AGENT")  # ingest-only, no transport needed

with httpx.stream("POST", url, headers=headers, json=body) as resp:
    final = adapter.ingest_stream(resp.iter_lines(), request=body)
```

## What gets emitted

Each `run` / `ingest_stream` call becomes a single trace:

| SSE event | Emitted event |
|---|---|
| request messages | `agent.input` (root span) — the user question, model, thread id |
| `response.tool_use` + `response.tool_result` | one `tool.call` per tool, with input and output |
| `response.tool_result.analyst.delta` | folded into the tool's `tool.call` — generated SQL, explanation, `query_id`, row count |
| `response` → `metadata.usage.tokens_consumed[]` | `model.invoke` + `cost.record` per model |
| `response.text.delta` / `response.thinking.delta` | assistant text and reasoning on `agent.output` |
| `error` | `agent.error` with `code`, `message`, `request_id` |

Message text, tool input/output, and generated SQL are only captured when
`CaptureConfig.capture_content` is enabled. Structural signals (tool names,
status, token counts, and the Cortex Analyst **row count**) are always emitted.
